import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
cuda = torch.cuda.is_available()


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


class ConvEncoder(nn.Module):
	def __init__(self,ninp=40,nout=128):
		super(ConvEncoder, self).__init__()
		self.convnet = []
		self.convnet.append(nn.Conv1d(in_channels=ninp,out_channels=64,kernel_size=3,stride=1,padding=1))
		self.convnet.append(nn.BatchNorm1d(64))
		self.convnet.append(nn.LeakyReLU())
		self.convnet.append(nn.Conv1d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1))
		self.convnet.append(nn.BatchNorm1d(64))
		self.convnet.append(nn.LeakyReLU())
		self.convnet.append(nn.Conv1d(in_channels=64,out_channels=64,kernel_size=3,stride=2,padding=1))
		self.convnet.append(nn.BatchNorm1d(64))
		self.convnet.append(nn.LeakyReLU())
		self.convnet.append(nn.Conv1d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1))
		self.convnet.append(nn.BatchNorm1d(128))
		self.convnet.append(nn.LeakyReLU())
		self.convnet.append(nn.Conv1d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1))
		self.convnet.append(nn.BatchNorm1d(128))
		self.convnet.append(nn.LeakyReLU())
		self.convnet.append(nn.Conv1d(in_channels=128,out_channels=128,kernel_size=3,stride=2,padding=1))
		self.convnet.append(nn.BatchNorm1d(128))
		self.convnet.append(nn.LeakyReLU())
		self.convnet.append(nn.Conv1d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1))
		self.convnet.append(nn.BatchNorm1d(256))
		self.convnet.append(nn.LeakyReLU())
		self.convnet.append(nn.Conv1d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1))
		self.convnet.append(nn.BatchNorm1d(256))
		self.convnet.append(nn.LeakyReLU())
		self.convnet.append(nn.Conv1d(in_channels=256,out_channels=256,kernel_size=3,stride=2,padding=1))
		self.convnet.append(nn.BatchNorm1d(256))
		self.convnet.append(nn.LeakyReLU())
		self.convnet = nn.Sequential(*self.convnet)

		self.key_proj = nn.Linear(256,nout)
		self.val_proj = nn.Linear(256,nout)

		self.ninp = ninp
		self.nout = nout
		self.init_weights()

	def init_weights(self):
		init_range = 0.1
		self.key_proj.bias.data.fill_(0)
		# self.val_proj.bias.data.fill_(0)
		torch.nn.init.xavier_uniform_(self.key_proj.weight.data)
		# torch.nn.init.xavier_uniform(self.val_proj.weight.data)


	def forward(self,input,lens):
		var = self.convnet(input)
		var = var.transpose(2,1)
		lens = np.array(lens)//8 + 1
		keys = self.key_proj(var)
		values = self.val_proj(var)


		return keys,values,lens

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


	def forward(self,keys,values,lens,inputs,teacher_force_rate=1,future=0):
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
		print(masks.shape)
		print(keys.shape)
		energy = torch.bmm(query.unsqueeze(1),keys).squeeze(1) * masks

		energy = energy - (1 - masks) * 1e6
		energy_max = torch.max(energy,1)[0].unsqueeze(1)
		energy = torch.exp(energy - energy_max) * masks
		attention = energy / (energy.sum(1).unsqueeze(1))
		context = torch.bmm(attention.unsqueeze(1),values).squeeze(1)
		return attention,context
