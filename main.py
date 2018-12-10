import argparse
import csv
import itertools
import os
import sys
import time

import numpy as np
import torch
from inferno.trainers.basic import Trainer
from inferno.trainers.callbacks.base import Callback
from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger
from torch import nn
from tqdm import tqdm
from torch.autograd import Variable
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import _use_shared_memory
from DickNet import *

INPUT_DIM = 40
batch_times = []
train_losses = []
val_losses = []

val_strings = []
val_labels = []

attentions = []

train_steps = 0
val_steps = 0

def load_data(name, args, test=False):
	# Load the numpy files
	features = np.load(os.path.join(args.data_directory, '{}.npy'.format(name)), encoding="latin1")
	if test:
		labels = None
	else:
		labels = np.load(os.path.join(args.data_directory, 'nb_{}_transcripts.npy'.format(name)), encoding="latin1")
	return features, labels


def build_charset(utterances):
	# Create a character set
	chars = set(itertools.chain.from_iterable(utterances))
	chars = list(chars)
	chars.sort()
	return chars


def make_charmap(charset):
	# Create the inverse character map
	return {c: i for i, c in enumerate(charset)}


def map_characters(utterances, charmap):
	# Convert transcripts to ints
	ints = [np.array([charmap[c] for c in u], np.int32) for u in utterances]
	return ints


def calc_output_len(inlens,kernel_size=3,stride=1,padding=1,dilation=1):
	return ((np.array(inlens) + 2 * padding - dilation * (kernel_size - 1) - 1)/stride + 1).astype(int)

def full_run_lens(lens):
	lens = calc_output_len(lens,kernel_size=3,stride=1,padding=1)
	lens = calc_output_len(lens,kernel_size=3,stride=1,padding=1)
	lens = calc_output_len(lens,kernel_size=3,stride=2,padding=1)
	lens = calc_output_len(lens,kernel_size=3,stride=1,padding=1)
	lens = calc_output_len(lens,kernel_size=3,stride=1,padding=1)
	lens = calc_output_len(lens,kernel_size=3,stride=2,padding=1)
	lens = calc_output_len(lens,kernel_size=3,stride=1,padding=1)
	lens = calc_output_len(lens,kernel_size=3,stride=1,padding=1)
	lens = calc_output_len(lens,kernel_size=3,stride=2,padding=1)
	return lens

class SpeechDataset(Dataset):
	# Dataset for utterances and transcripts
	def __init__(self, features, transcripts):
		self.features = [torch.from_numpy(x).float() for x in features]
		if transcripts:
			self.transcripts = [torch.from_numpy(x + 1).long() for x in transcripts]  # +1 for start/end token
			assert len(self.features) == len(self.transcripts)
		else:
			self.transcripts = None

	def __len__(self):
		return len(self.features)

	def __getitem__(self, item):
		if self.transcripts:
			return self.features[item], self.transcripts[item]
		else:
			return self.features[item], None


def speech_collate_fn(batch):
	n = len(batch)

	# allocate tensors for lengths
	if _use_shared_memory:
		ulens = torch.IntStorage._new_shared(n).new(n)
		llens = torch.IntStorage._new_shared(n).new(n)
	else:
		ulens = torch.IntTensor(n)
		llens = torch.IntTensor(n)

	# calculate lengths
	for i, data in enumerate(batch):
		u, l = data[0], data[1]
		# +1 to account for start/end token
		ulens[i] = u.size(0)
		if l is None:
			llens[i] = 1
		else:
			llens[i] = l.size(0) + 1

	# calculate max length
	umax = int(ulens.max())
	lmax = int(llens.max())

	# allocate tensors for data based on max length
	if _use_shared_memory:
		uarray = torch.FloatStorage._new_shared(umax * n).new(umax, n, INPUT_DIM).zero_()
		l1array = torch.LongStorage._new_shared(lmax * n).new(lmax, n).zero_()
		l2array = torch.LongStorage._new_shared(lmax * n).new(lmax, n).zero_()
	else:
		uarray = torch.FloatTensor(umax, n, INPUT_DIM).zero_()
		l1array = torch.LongTensor(lmax, n).zero_()
		l2array = torch.LongTensor(lmax, n).zero_()

	# collate data tensors into pre-allocated arrays
	for i, data in enumerate(batch):
		u, l = data[0], data[1]
		uarray[:u.size(0), i, :] = u
		if l is not None:
			l1array[1:l.size(0) + 1, i] = l
			l2array[:l.size(0), i] = l

	return uarray, ulens, l1array, llens, l2array


class SequenceShuffle(nn.Module):
	# Performs pooling for pBLSTM
	def forward(self, seq):
		assert isinstance(seq, PackedSequence)
		padded, lens = pad_packed_sequence(seq)  # (L, BS, D)
		padded = padded.transpose(0, 1)
		if padded.size(1) % 2 > 0:
			padded = padded[:, :-1, :]
		padded = padded.contiguous()
		padded = padded.view(padded.size(0), padded.size(1) // 2, 2 * padded.size(2))
		padded = padded.transpose(0, 1)
		newlens = np.array(lens) // 2
		newseq = pack_padded_sequence(padded, newlens)
		return newseq


class AdvancedLSTM(nn.LSTM):
	# Class for learning initial hidden states when using LSTMs
	def __init__(self, *args, **kwargs):
		super(AdvancedLSTM, self).__init__(*args, **kwargs)
		bi = 2 if self.bidirectional else 1
		self.h0 = nn.Parameter(torch.FloatTensor(bi, 1, self.hidden_size).zero_())
		self.c0 = nn.Parameter(torch.FloatTensor(bi, 1, self.hidden_size).zero_())

	def initial_state(self, n):
		return (
			self.h0.expand(-1, n, -1).contiguous(),
			self.c0.expand(-1, n, -1).contiguous()
		)

	def forward(self, input, hx=None):
		if hx is None:
			n = input.batch_sizes[0]
			hx = self.initial_state(n)
		return super(AdvancedLSTM, self).forward(input, hx=hx)


class pLSTM(AdvancedLSTM):
	# Pyramidal LSTM
	def __init__(self, *args, **kwargs):
		super(pLSTM, self).__init__(*args, **kwargs)
		self.shuffle = SequenceShuffle()

	def forward(self, input, hx=None):
		return super(pLSTM, self).forward(self.shuffle(input), hx=hx)


class EncoderModel(nn.Module):
	# Encodes utterances to produce keys and values
	def __init__(self, args):
		super(EncoderModel, self).__init__()
		self.rnns = nn.ModuleList()
		self.rnns.append(AdvancedLSTM(INPUT_DIM, args.encoder_dim, bidirectional=True))
		self.rnns.append(pLSTM(args.encoder_dim * 4, args.encoder_dim, bidirectional=True))
		self.rnns.append(pLSTM(args.encoder_dim * 4, args.encoder_dim, bidirectional=True))
		self.rnns.append(pLSTM(args.encoder_dim * 4, args.encoder_dim, bidirectional=True))
		self.key_projection = nn.Linear(args.encoder_dim * 2, args.key_dim)
		self.value_projection = nn.Linear(args.encoder_dim * 2, args.value_dim)

	def forward(self, utterances, utterance_lengths):
		h = utterances

		# Sort and pack the inputs
		sorted_lengths, order = torch.sort(utterance_lengths, 0, descending=True)
		_, backorder = torch.sort(order, 0)
		h = h[:, order, :]
		h = pack_padded_sequence(h, sorted_lengths.data.cpu().numpy())

		# RNNs
		for rnn in self.rnns:
			h, _ = rnn(h)

		# Unpack and unsort the sequences
		h, output_lengths = pad_packed_sequence(h)
		h = h[:, backorder, :]
		output_lengths = torch.from_numpy(np.array(output_lengths))
		if backorder.data.is_cuda:
			output_lengths = output_lengths.cuda()
		output_lengths = output_lengths[backorder.data]

		# Apply key and value
		keys = self.key_projection(h)
		values = self.value_projection(h)

		return keys, values, output_lengths


# class ConvEncoder(nn.Module):
# 	def __init__(self,args,ninp=40,nout=128):
# 		super(ConvEncoder, self).__init__()
# 		self.convnet = []
# 		self.convnet.append(nn.Conv1d(in_channels=ninp,out_channels=64,kernel_size=3,stride=1,padding=1))
# 		self.convnet.append(nn.BatchNorm1d(64))
# 		self.convnet.append(nn.LeakyReLU())
# 		self.convnet.append(nn.Conv1d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1))
# 		self.convnet.append(nn.BatchNorm1d(64))
# 		self.convnet.append(nn.LeakyReLU())
# 		self.convnet.append(nn.Conv1d(in_channels=64,out_channels=64,kernel_size=3,stride=2,padding=1))
# 		self.convnet.append(nn.BatchNorm1d(64))
# 		self.convnet.append(nn.LeakyReLU())
# 		self.convnet.append(nn.Conv1d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1))
# 		self.convnet.append(nn.BatchNorm1d(128))
# 		self.convnet.append(nn.LeakyReLU())
# 		self.convnet.append(nn.Conv1d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1))
# 		self.convnet.append(nn.BatchNorm1d(128))
# 		self.convnet.append(nn.LeakyReLU())
# 		self.convnet.append(nn.Conv1d(in_channels=128,out_channels=128,kernel_size=3,stride=2,padding=1))
# 		self.convnet.append(nn.BatchNorm1d(128))
# 		self.convnet.append(nn.LeakyReLU())
# 		self.convnet.append(nn.Conv1d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1))
# 		self.convnet.append(nn.BatchNorm1d(256))
# 		self.convnet.append(nn.LeakyReLU())
# 		self.convnet.append(nn.Conv1d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1))
# 		self.convnet.append(nn.BatchNorm1d(256))
# 		self.convnet.append(nn.LeakyReLU())
# 		self.convnet.append(nn.Conv1d(in_channels=256,out_channels=256,kernel_size=3,stride=2,padding=1))
# 		self.convnet.append(nn.BatchNorm1d(256))
# 		self.convnet.append(nn.LeakyReLU())
# 		self.convnet = nn.Sequential(*self.convnet)

# 		self.key_proj = nn.Linear(256,nout)
# 		self.val_proj = nn.Linear(256,nout)

# 		self.ninp = ninp
# 		self.nout = nout
# 		self.cuda = args.cuda
# 		self.init_weights()

# 	def init_weights(self):
# 		init_range = 0.1
# 		self.key_proj.bias.data.fill_(0)
# 		# self.val_proj.bias.data.fill_(0)
# 		torch.nn.init.xavier_uniform_(self.key_proj.weight.data)
# 		# torch.nn.init.xavier_uniform(self.val_proj.weight.data)
#
#
#	def forward(self,input,lens):
#		var = input.permute(1,2,0)
#		var = self.convnet(var)
#		var = var.permute(0,2,1)
#		lens = full_run_lens(lens)
#		keys = self.key_proj(var).permute(1,0,2)
#		values = self.val_proj(var).permute(1,0,2)
#		lens = torch.from_numpy(lens)
#		if self.cuda:
#			lens = lens.cuda()
#		return keys,values,lens

def sample_gumbel(shape, eps=1e-10, out=None):
	"""
	Sample from Gumbel(0, 1)
	based on
	https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
	(MIT license)
	"""
	U = out.resize_(shape).uniform_() if out is not None else torch.rand(shape)
	return - torch.log(eps - torch.log(U + eps))


def gumbel_argmax(logits, dim):
	# Draw from a multinomial distribution efficiently
	return torch.max(logits + sample_gumbel(logits.size(), out=logits.data.new()), dim)[1]


class AdvancedLSTMCell(nn.LSTMCell):
	# Extend LSTMCell to learn initial state
	def __init__(self, *args, **kwargs):
		super(AdvancedLSTMCell, self).__init__(*args, **kwargs)
		self.h0 = nn.Parameter(torch.FloatTensor(1, self.hidden_size).zero_())
		self.c0 = nn.Parameter(torch.FloatTensor(1, self.hidden_size).zero_())

	def initial_state(self, n):
		return (
			self.h0.expand(n, -1).contiguous(),
			self.c0.expand(n, -1).contiguous()
		)


def output_mask(maxlen, lengths):
	"""
	Create a mask on-the-fly
	:param maxlen: length of mask
	:param lengths: length of each sequence
	:return: mask shaped (maxlen, len(lengths))
	"""
	lens = lengths.unsqueeze(0)
	ran = torch.arange(0, maxlen, 1, out=lengths.new()).unsqueeze(1)
	mask = ran < lens
	return mask


def calculate_attention(keys, mask, queries,cnn=False):
	"""
	Attention calculation
	:param keys: (N, L, key_dim)
	:param mask: (N, L)
	:param queries: (N, key_dim)
	:return: attention (N, L)
	"""
	energy = torch.bmm(keys, queries.unsqueeze(2)).squeeze(2) * mask  # (N, L)
	energy = energy - (1 - mask) * 1e4  # subtract large number from padded region
	emax = torch.max(energy, 1)[0].unsqueeze(1)  # (N, L)
	eval = torch.exp(energy - emax) * mask  # (N, L)
	attn = eval / (eval.sum(1).unsqueeze(1))  # (N, L)
	return attn


def calculate_context(attn, values):
	"""
	Context calculation
	:param attn:  (N, L)
	:param values: (N, L, value_dim)
	:return: Context (N, value_dim)
	"""
	ctx = torch.bmm(attn.unsqueeze(1), values).squeeze(1)  # (N, value_dim)
	return ctx


class DecoderModel(nn.Module):
	# Speller/Decoder
	def __init__(self, args, vocab_size):
		super(DecoderModel, self).__init__()
		self.embedding = nn.Embedding(vocab_size + 1, args.decoder_dim)
		self.input_rnns = nn.ModuleList()
		self.input_rnns.append(AdvancedLSTMCell(args.decoder_dim + args.value_dim, args.decoder_dim))
		self.input_rnns.append(AdvancedLSTMCell(args.decoder_dim, args.decoder_dim))
		self.input_rnns.append(AdvancedLSTMCell(args.decoder_dim, args.decoder_dim))
		self.query_projection = nn.Linear(args.decoder_dim, args.key_dim)
		self.char_projection = nn.Sequential(
			nn.Linear(args.decoder_dim+args.value_dim, args.decoder_dim),
			nn.LeakyReLU(),
			nn.Linear(args.decoder_dim, vocab_size+1)
		)
		self.force_rate = args.teacher_force_rate
		self.char_projection[-1].weight = self.embedding.weight  # weight tying

	def forward_pass(self, input_t, keys, values, mask, ctx, input_states):
		# Embed the previous character
		embed = self.embedding(input_t)
		# Concatenate embedding and previous context
		ht = torch.cat((embed, ctx), dim=1)
		# Run first set of RNNs
		new_input_states = []
		for rnn, state in zip(self.input_rnns, input_states):
			ht, newstate = rnn(ht, state)
			new_input_states.append((ht, newstate))
		# Calculate query
		query = self.query_projection(ht)
		# Calculate attention
		attn = calculate_attention(keys=keys, mask=mask, queries=query)
		# Calculate context
		ctx = calculate_context(attn=attn, values=values)
		# Concatenate hidden state and context
		ht = torch.cat((ht, ctx), dim=1)
		# Run projection
		logit = self.char_projection(ht)
		# Sample from logits
		generated = gumbel_argmax(logit, 1)  # (N,)
		return logit, generated, ctx, attn, new_input_states

	def forward(self, inputs, input_lengths, keys, values, utterance_lengths, future=0):
		mask = Variable(output_mask(values.size(0), utterance_lengths).transpose(0, 1)).float()
		values = values.transpose(0, 1)
		keys = keys.transpose(0, 1)
		t = inputs.size(0)
		n = inputs.size(1)

		# Initial states
		input_states = [rnn.initial_state(n) for rnn in self.input_rnns]

		# Initial context
		h0 = input_states[-1][0]
		query = self.query_projection(h0)
		attn = calculate_attention(keys, mask, query)
		ctx = calculate_context(attn, values)

		# Decoder loop
		logits = []
		attns = []
		generateds = []
		for i in range(t):
			# Use forced or generated inputs
			if len(generateds) > 0 and self.force_rate < 1 and self.training:
				input_forced = inputs[i]
				input_gen = generateds[-1]
				input_mask = Variable(input_forced.data.new(*input_forced.size()).bernoulli_(self.force_rate))
				input_t = (input_mask * input_forced) + ((1 - input_mask) * input_gen)
			else:
				input_t = inputs[i]
			# Run a single timestep
			logit, generated, ctx, attn, input_states = self.forward_pass(
				input_t=input_t, keys=keys, values=values, mask=mask, ctx=ctx,
				input_states=input_states
			)
			# Save outputs
			logits.append(logit)
			attns.append(attn)
			generateds.append(generated)

		# For future predictions
		if future > 0:
			assert len(generateds) > 0
			input_t = generateds[-1]
			for _ in range(future):
				# Run a single timestep
				logit, generated, ctx, attn, input_states = self.forward_pass(
					input_t=input_t, keys=keys, values=values, mask=mask, ctx=ctx,
					input_states=input_states
				)
				# Save outputs
				logits.append(logit)
				attns.append(attn)
				generateds.append(generated)
				# Pass generated as next input
				input_t = generated

		# Combine all the outputs
		logits = torch.stack(logits, dim=0)  # (L, N, Vocab Size)
		attns = torch.stack(attns, dim=0)  # (L, N, T)
		generateds = torch.stack(generateds, dim=0)  # (L, N)
		return logits, attns, generateds


class Seq2SeqModel(nn.Module):
	# Tie encoder and decoder together
	def __init__(self, args, vocab_size):
		super(Seq2SeqModel, self).__init__()
		self.encoder = EncoderModel(args)
		if args.cnn:
			self.encoder = DenseNet()
		self.decoder = DecoderModel(args, vocab_size=vocab_size)
		self._state_hooks = {}
		self.args = args

	def forward(self, utterances, utterance_lengths, chars, char_lengths, future=0):
		keys, values, lengths = self.encoder(utterances, utterance_lengths)
		if self.args.cuda:
			lengths = lengths.cuda()
		logits, attns, generated = self.decoder(chars, char_lengths, keys, values, lengths, future=future)
		self._state_hooks['attention'] = attns.permute(1, 0, 2).unsqueeze(1).detach().cpu().numpy()
		return logits, generated, char_lengths, self._state_hooks['attention']


def decode_output(output, charset):
	# Convert ints back to strings
	chars = []
	for o in output:
		if o == 0:
			break
		chars.append(charset[o - 1])
	return "".join(chars)


def generate_transcripts(args, model, loader, charset):
    # Create and yield transcripts
    for uarray, ulens, l1array, llens, l2array in loader:
        if args.cuda:
            uarray = uarray.cuda()
            ulens = ulens.cuda()
            l1array = l1array.cuda()
            llens = llens.cuda()
        uarray = Variable(uarray)
        ulens = Variable(ulens)
        l1array = Variable(l1array)
        llens = Variable(llens)
        start = time.time()
        logits, generated, lens, attns = model(
            uarray, ulens, l1array, llens,
            future=args.generator_length)
        end = time.time()
        print("Inference took {}".format(end-start))
        attentions = attns
        generated = generated.data.cpu().numpy()  # (L, BS)
        n = uarray.size(1)
        for i in range(n):
            transcript = decode_output(generated[:, i], charset)
            yield transcript


def write_transcripts(path, args, model, loader, charset):
	# Write the Kaggle CSV file
	model.eval()
	os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
	with open(path, 'w', newline='') as f:
		w = csv.writer(f)
		transcripts = generate_transcripts(args, model, loader, charset)
		w.writerow(['Id', 'Predicted'])
		for i, t in enumerate(transcripts):
			print(i, t)
			w.writerow([i, t])


def make_loader(features, labels, args, shuffle=True, batch_size=64):
	# Build the DataLoaders
	kwargs = {'pin_memory': True, 'num_workers': args.num_workers} if args.cuda else {}
	dataset = SpeechDataset(features, labels)
	loader = DataLoader(dataset, collate_fn=speech_collate_fn, shuffle=shuffle, batch_size=batch_size, **kwargs)
	return loader, dataset


class SequenceCrossEntropy(nn.CrossEntropyLoss):
	# Customized CrossEntropyLoss
	def __init__(self, *args, **kwargs):
		super(SequenceCrossEntropy, self).__init__(*args, reduce=False, **kwargs)

	def forward(self, prediction, target):
		logits, generated, sequence_lengths, attns = prediction
		maxlen = logits.size(0)
		mask = Variable(output_mask(maxlen, sequence_lengths.data)).float()
		logits = logits * mask.unsqueeze(2)
		losses = super(SequenceCrossEntropy, self).forward(logits.view(-1, logits.size(2)), target.view(-1))
		loss = torch.sum(mask.view(-1) * losses) / logits.size(1)
		return loss


class SubmissionCallback(Callback):
	# Periodically write Kaggle submissions during training
	def __init__(self, args, loader, charset):
		super(SubmissionCallback, self).__init__()
		self.args = args
		self.loader = loader
		self.charset = charset

	def end_of_validation_run(self, **_):
		step = self.trainer.iteration_count
		with open(os.path.join(self.args.save_directory, 'model-{:012d}.pt'.format(step)), 'wb') as f:
			torch.save(self.trainer.model.state_dict(), f)
			print("Saved mutha fucka")
		np.save('attentions.npy', np.array(attentions))
		write_transcripts(
			path=os.path.join(self.args.save_directory, 'submission-{:012d}.csv'.format(step)),
			model=self.trainer.model,
			loader=self.loader,
			charset=self.charset,
			args=self.args
		)


class EpochTimer(Callback):
	"""
	Callback that prints the elapsed time per epoch
	"""

	def __init__(self):
		super(EpochTimer, self).__init__()
		self.start_time = None

	def begin_of_training_run(self, **_kwargs):
		self.start_time = time.time()

	def begin_of_epoch(self, **_kwargs):
		self.start_time = time.time()

	def end_of_epoch(self, epoch_count, **_kwargs):
		assert self.start_time is not None
		end_time = time.time()
		elapsed = end_time - self.start_time
		print("Epoch {} elapsed: {}".format(epoch_count, elapsed))
		self.start_time = None


class IterationTimer(Callback):
	"""
	Callback that prints the elapsed time per batch
	"""

	def __init__(self):
		super(IterationTimer, self).__init__()
		self.start_time = None

	def begin_of_training_iteration(self, **_kwargs):
		self.start_time = time.time()

	def end_of_training_iteration(self, **_kwargs):
		assert self.start_time is not None
		end_time = time.time()
		elapsed = end_time - self.start_time
		print("Iteration elapsed: {}ms".format(int(1000 * elapsed)))
		self.start_time = None


def train(model, dataloader, criterion, optimizer, args):
	global train_steps
	train_strings = []
	train_labels = []
	train_attentions = []
	train_losses = []
	model.train()
	i = iter(dataloader)
	for i in range(0, len(dataloader.features), args.batch_size):
		end = min(i + args.batch_size, len(dataloader.features))
		data = dataloader.features[i:end]
		labels = dataloader.transcripts[i:end]
		uarray, ulens, l1array, llens, l2array = speech_collate_fn((data, labels))
		start = time.time()
		optimizer.zero_grad()
		logits, generated, char_lengths, attention = model(uarray, ulens, l1array, llens)
		loss = criterion((logits, generated, char_lengths), l2array)
		loss.backward()
		optimizer.step()
		train_losses.append(loss.item())
		end = time.time()
		batch_times.append(end - start)




def run(args):
    print("Pytorch version {}".format(torch.__version__))
    print("Loading Data")
    trainfeats, trainwords = load_data('train', args)
    devfeats, devwords = load_data('dev', args)
    testfeats, _ = load_data('test', args, True)
    print("Building Charset")
    charset = build_charset(np.concatenate((trainwords, devwords), axis=0))
    charmap = make_charmap(charset)
    charcount = len(charset)
    print("Mapping Characters")
    trainchars = map_characters(trainwords, charmap)
    devchars = map_characters(devwords, charmap)
    print("Building Loader")
    dev_loader, dev_data = make_loader(devfeats, devchars, args, shuffle=True, batch_size=args.batch_size)
    train_loader, train_data = make_loader(trainfeats, trainchars, args, shuffle=True, batch_size=args.batch_size)
    test_loader, test_data = make_loader(testfeats, None, args, shuffle=False, batch_size=args.batch_size)
    print("Building Model")
    model = Seq2SeqModel(args, vocab_size=charcount)
    print("Running")
    trainer = Trainer()
    if args.load:
        if os.path.exists(os.path.join(args.save_directory, trainer._checkpoint_filename)):
            trainer.load(from_directory=args.save_directory)
            print(trainer.model.state_dict())
            model.load_state_dict(trainer.model.state_dict())
            print("model loaded")
            if args.cuda:
                model = model.cuda()
    if False:
        optimizer = torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-5)
        for epoch in range(args.epochs):
            train(model, train_data, SequenceCrossEntropy(), optimizer, args)

    if args.run_inference:
        print("Running inference")
    else:
        trainer = Trainer(model) \
            .build_criterion(SequenceCrossEntropy) \
            .build_optimizer('Adam', lr=args.lr, weight_decay=args.weight_decay) \
            .validate_every((1, 'epochs')) \
            .save_every((1, 'epochs')) \
            .save_to_directory(args.save_directory) \
            .set_max_num_epochs(args.epochs) \
            .build_logger(TensorboardLogger(log_scalars_every=(1, 'iteration'),
                                            log_images_every=(20, 'iteration')),
                                            # log_histograms_every=(np.inf, 'iteration')),
                          log_directory=args.save_directory)
        trainer.logger.observe_state('attention')

        # Bind loaders
        trainer.bind_loader('train', train_loader, num_inputs=4, num_targets=1)
        trainer.bind_loader('validate', dev_loader, num_inputs=4, num_targets=1)
        trainer.register_callback(SubmissionCallback(
            args=args,
            charset=charset,
            loader=dev_loader,
        ))
        trainer.register_callback(EpochTimer)
        trainer.register_callback(IterationTimer)

        if args.cuda:
            trainer.cuda()

        # write_transcripts(
        # path=os.path.join(args.save_directory, 'submission.csv'),
        # args=args, model=model, loader=test_loader, charset=charset)
        # Go!
        trainer.fit()
        trainer.save()
        model = trainer.model
    write_transcripts(
        path=os.path.join(args.save_directory, 'submission.csv'),
        args=args, model=model, loader=dev_loader, charset=charset)


def main(argv):
    # Training settings
    parser = argparse.ArgumentParser(description='HW4 Baseline')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size')
    parser.add_argument('--save-directory', type=str, default='output/hw4-LAS-baseline/v1', help='output directory')
    parser.add_argument('--data-directory', type=str, default='./data', help='data directory')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs')
    parser.add_argument('--num-workers', type=int, default=2, metavar='N', help='number of workers')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--run-inference', action='store_true', default=False, help='Just run inference')
    parser.add_argument('--cnn',action='store_true',default=False,help='Use CNN')
    parser.add_argument('--load',action='store_true',default=False,help='Load model')

    parser.add_argument('--lr', type=float, default=1e-3, metavar='N', help='lr')
    parser.add_argument('--weight-decay', type=float, default=1e-5, metavar='N', help='weight decay')
    parser.add_argument('--teacher-force-rate', type=float, default=0.7, metavar='N', help='teacher forcing rate')

    parser.add_argument('--encoder-dim', type=int, default=256, metavar='N', help='hidden dimension')
    parser.add_argument('--decoder-dim', type=int, default=512, metavar='N', help='hidden dimension')
    parser.add_argument('--value-dim', type=int, default=128, metavar='N', help='hidden dimension')
    parser.add_argument('--key-dim', type=int, default=128, metavar='N', help='hidden dimension')
    parser.add_argument('--generator-length', type=int, default=250, metavar='N', help='maximum length to generate')

    args = parser.parse_args(argv)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    run(args)



if __name__ == '__main__':
	main(sys.argv[1:])
