import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import _use_shared_memory
# Emb instead of oneHot  ----> done?
# need to mask attention ----> done?
# maybe implement psi    ----> ????
# LSTMCELL is faster     ----> don't wanna do
# instead of concatenation, you could use conv1d strides in Decoder
cuda = torch.cuda.is_available()


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
        newseq = nn.utils.rnn.pack_padded_sequence(padded, newlens)
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
    def __init__(self, ninp=40, encoder_dim=256, key_dim=128, value_dim=128):
        super(EncoderModel, self).__init__()
        self.rnns = nn.ModuleList()
        self.rnns.append(AdvancedLSTM(ninp, encoder_dim, bidirectional=True))
        self.rnns.append(pLSTM(encoder_dim * 4, encoder_dim, bidirectional=True))
        self.rnns.append(pLSTM(encoder_dim * 4, encoder_dim, bidirectional=True))
        self.rnns.append(pLSTM(encoder_dim * 4, encoder_dim, bidirectional=True))
        self.key_projection = nn.Linear(encoder_dim * 2, key_dim)
        self.value_projection = nn.Linear(encoder_dim * 2, value_dim)

    def forward(self, utterances, utterance_lengths):
        h = utterances.permute(1, 0, 2)

        # Sort and pack the inputs
        sorted_lengths, order = torch.sort(utterance_lengths, 0, descending=True)
        _, backorder = torch.sort(order, 0)
        h = h[:, order, :]
        h = nn.utils.rnn.pack_padded_sequence(h, sorted_lengths.data.cpu().numpy())

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
        h = h.permute(1, 0, 2)
        # Apply key and value
        keys = self.key_projection(h)
        values = self.value_projection(h)

        return keys, values, output_lengths


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


def createMasks(lens,maxseq):
	x = np.arange(maxseq).reshape(1,maxseq)
	y = np.array(lens).reshape(len(lens),1)
	rmasks = Variable(torch.from_numpy((x < y).astype(int)))
	if cuda:
		rmasks = rmasks.cuda()
	return rmasks


class SeqCrossEntropyLoss(torch.nn.Module):
	def __init__(self):
		super().__init__()
	def forward(self,outputs,labels,lens):
		output_list = []
		label_list = []
		for (output,label,seqlen) in zip(outputs,labels,lens):
			output_list.append(output[:seqlen])
			label_list.append(label[:seqlen])
		outputs = torch.cat(output_list)
		labels = torch.cat(label_list)
		loss = torch.nn.functional.cross_entropy(outputs,labels,size_average=False)
		return loss



class LockedDropout(nn.Module):
	def __init__(self):
		super().__init__()
	def forward(self,x,dropout=0.5):
		if not self.training or not dropout:
			return x
		m = x.data.new(x.size(0),1,x.size(2)).bernoulli_(1-dropout)
		mask = Variable(m,requires_grad=False)/(1-dropout)
		mask = mask.expand_as(x)
		return mask * x

# class Encoder(nn.Module):
# 	def __init__(self,ninp=40,nhid=256,nout=128,bidirectional=True,hdrop=0.3):
# 		super(Encoder, self).__init__()
# 		self.lockdrop = LockedDropout()

# 		#encoders
# 		self.rnns=[]
# 		mult = 2 if bidirectional else 1
# 		#input decode LSTM

# 		#pyramidal LSTM
# 		self.rnns.append(nn.LSTM(ninp,nhid,1,dropout=0,bidirectional=bidirectional,batch_first=True))
# 		self.rnns.append(nn.LSTM(nhid*mult*2,nhid,1,dropout=0,bidirectional=bidirectional,batch_first=True))
# 		self.rnns.append(nn.LSTM(nhid*mult*2,nhid,1,dropout=0,bidirectional=bidirectional,batch_first=True))

# 		self.rnns = nn.ModuleList(self.rnns)

# 		#key,value projection
# 		self.key_proj = nn.Linear(nhid*mult,nout)
# 		self.val_proj = nn.Linear(nhid*mult,nout)

# 		self.nhid = nhid
# 		self.ninp = ninp
# 		self.nout = nout
# 		self.hdrop = hdrop
# 		self.init_weights()

# 	def init_weights(self):
# 		init_range = 0.1
# 		self.key_proj.bias.data.fill_(0)
# 		# self.val_proj.bias.data.fill_(0)
# 		torch.nn.init.xavier_uniform_(self.key_proj.weight.data)
# 		# torch.nn.init.xavier_uniform(self.val_proj.weight.data)


# 	def forward(self,input,lens):
# 		var = input
# 		var = nn.utils.rnn.pack_padded_sequence(var,lens,batch_first=True)
# 		var,_ = self.rnns[0](var)
# 		var,lens = nn.utils.rnn.pad_packed_sequence(var,batch_first=True)
# 		var = var.contiguous()
# 		var = self.lockdrop(var,0.2).contiguous()

# 		#second pyramid op
# 		if var.size()[1] % 2 == 1:
# 			var = var[:,:-1,:].contiguous()
# 		var = var.view((int(var.size()[0]),int(var.size()[1]/2),int(var.size()[2]*2)))
# 		lens = [int(l / 2) for l in lens]
# 		var = nn.utils.rnn.pack_padded_sequence(var,lens,batch_first=True)
# 		var,_ = self.rnns[1](var)
# 		var,lens = nn.utils.rnn.pad_packed_sequence(var,batch_first=True)
# 		var = var.contiguous()
# 		var = self.lockdrop(var,0.1).contiguous()

# 		#third pyramid op
# 		if var.size()[1] % 2 == 1:
# 			var = var[:,:-1,:].contiguous()
# 		var = var.view((int(var.size()[0]),int(var.size()[1]/2),int(var.size()[2]*2)))
# 		lens = [int(l / 2) for l in lens]
# 		var = nn.utils.rnn.pack_padded_sequence(var,lens,batch_first=True)
# 		var,_ = self.rnns[2](var)
# 		var,lens = nn.utils.rnn.pad_packed_sequence(var,batch_first=True)
# 		var = var.contiguous()
# 		#var = self.lockdrop(var,self.hdrop).contiguous()

# 		keys = self.key_proj(var)
# 		values = self.val_proj(var)


# 		return keys,values,lens

class ALSTMCell(nn.LSTMCell):
	def __init__(self,*args,**kwargs):
		super(ALSTMCell,self).__init__(*args,**kwargs)
		self.h0 = nn.Parameter(torch.FloatTensor(1,self.hidden_size).zero_())
		self.c0 = nn.Parameter(torch.FloatTensor(1,self.hidden_size).zero_())

	def init_state(self,n):
		return (self.h0.expand(n,-1).contiguous(),self.c0.expand(n,-1).contiguous())

class Decoder(nn.Module):
	def __init__(self,ninp=34,ncontext=128,nhid=512):
		super(Decoder, self).__init__()
		self.attention = Attention()

		self.emb = nn.Embedding(ninp,ncontext)

		self.rnn_inith = torch.nn.ParameterList()
		self.rnn_initc = torch.nn.ParameterList()

		# self.rnns = nn.LSTM(2*ncontext,nhid,num_layers=3)
		self.rnns = torch.nn.ModuleList()
		self.rnns.append(ALSTMCell(2*ncontext,nhid))
		self.rnns.append(ALSTMCell(nhid,nhid))
		self.rnns.append(ALSTMCell(nhid,nhid))

		self.query_proj = nn.Linear(nhid,ncontext)

		self.char_proj = []
		self.char_proj.append(nn.Linear(nhid + ncontext,ncontext))
		self.char_proj.append(nn.LeakyReLU())
		self.char_proj.append(nn.Linear(ncontext,ninp))
		self.char_proj = nn.Sequential(*self.char_proj)
		self.char_proj[-1].weight = self.emb.weight

		self.ninp = ninp
		self.ncontext = ncontext
		self.nhid = nhid
		# self.init_weights()
		if cuda:
			self.rnn_inith = self.rnn_inith.cuda()
			self.rnn_initc = self.rnn_initc.cuda()

	# def init_weights(self):
	# 	torch.nn.init.xavier_uniform(self.query_proj.weight.data)
	# 	# torch.nn.init.xavier_uniform(self.char_proj.weight.data)
	# 	# self.char_proj.bias.data = torch.from_numpy(np.load('lb.npy')).float()
	# 	for l in self.char_proj:
	# 		if 
	# 		torch.nn.init.xavier_uniform(l.weight.data)
	# 	self.char_proj[1].bias.data = torch.from_numpy(np.load('lb34.npy')).float()
	# 	self.char_proj[0].bias.data.fill_(0)
	# 	#self.char_proj[1].bias.data.fill_(0)


	def forward_pass(self,keys,values,lens,input_t,context,input_states):
		embed = self.emb(input_t)
		hidden_t = torch.cat((embed,context),dim=1)
		n_input_states =[]
		for rnn,state in zip(self.rnns,input_states):
			hidden_t,n_state = rnn(hidden_t,state)
			n_input_states.append((hidden_t,n_state))
		query = self.query_proj(hidden_t).squeeze(1)
		attention,context = self.attention(keys,values,lens,query)
		hidden_t = torch.cat((hidden_t,context),dim=1)
		logit = self.char_proj(hidden_t)
		gen_t = gumbel_argmax(logit,1)
		return logit,gen_t,context,attention,n_input_states


	def forward(self,keys,values,lens,inputs,teacher_force_rate=0.9,future=0):
		outputs = []
		attentions = []
		bsz = inputs.size(0)
		max_transcript_len = inputs.size(1)
		input_states =[rnn.init_state(bsz) for rnn in self.rnns]
		# cell = [self.cell.expand(bsz,-1).contiguous() for _ in range(3)]
		# cell = [c.repeat(bsz,1) for c in self.rnn_initc]
		h0 = input_states[-1][0]
		if cuda:
			h0 = h0.cuda()
		# output_embed = self.emb(sos)
		query = self.query_proj(h0)
		attention,context = self.attention(keys,values,lens,query)

		logits = []
		attentions = []
		generated = []
		for i in range(max_transcript_len):
			if len(generated) > 0 and teacher_force_rate < 1 and self.training:
				input_forced = inputs[:,i]
				input_gen = generated[-1]
				input_mask = Variable(input_forced.data.new(*input_forced.size()).bernoulli_(self.teacher_force_rate))
				input_t = (input_mask * input_forced) + ((1 - input_mask) * input_gen)
			else:
				input_t = inputs[:,i]
			logit,g_t,context,attention,input_states = self.forward_pass(keys,values,lens,input_t,context,input_states)
			logits.append(logit)
			attentions.append(attention)
			generated.append(g_t)

		if future > 0:
			assert len(generated) > 0
			input_t = generated[-1]
			for _ in range(future):
				logit,g_t,context,attention,input_states = self.forward_pass(keys,values,lens,input_t,context,input_states)
				logits.append(logit)
				attentions.append(attention)
				generated.append(g_t)
				input_t = g_t
		logits = torch.stack(logits,dim=0)
		attentions = torch.stack(attentions,dim=0)
		generated = torch.stack(generated,dim=0)
		return logits,attentions,generated

class Attention(nn.Module):
	def __init__(self):
		super(Attention, self).__init__()
		self.softmax = nn.Softmax(dim=-1)
	def forward(self,keys,values,lens,query):
		maxseq = max(lens)
		masks = createMasks(lens,maxseq).float()
		keys = keys.transpose(1,2)
		energy = torch.bmm(query.unsqueeze(1),keys).squeeze(1) * masks
		
		energy = energy - (1 - masks) * 1e6
		energy_max = torch.max(energy,1)[0].unsqueeze(1)
		energy = torch.exp(energy - energy_max) * masks
		attention = energy / (energy.sum(1).unsqueeze(1))
		context = torch.bmm(attention.unsqueeze(1),values).squeeze(1)
		return attention,context
