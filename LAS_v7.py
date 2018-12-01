import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
# Emb instead of oneHot  ----> done?
# need to mask attention ----> done?
# maybe implement psi    ----> ????
# LSTMCELL is faster     ----> don't wanna do
# instead of concatenation, you could use conv1d strides in Decoder
cuda = torch.cuda.is_available()


def createMasks(lens,maxseq):
	x = np.arange(maxseq).reshape(1,maxseq)
	y = np.array(lens).reshape(len(lens),1)
	rmasks = Variable(torch.from_numpy((x < y).astype(int)))
	if cuda:
		rmasks = rmasks.cuda()
	return rmasks

def oneHotVar(inputx,endim=34):
	if type(inputx) is Variable:
		inputx = inputx.data
	input_type = type(inputx)
	# print(inputx)
	bsz = inputx.size()[0]
	slen = inputx.size()[1]
	inputx = inputx.unsqueeze(2).type(torch.LongTensor)
	onehot = Variable(torch.LongTensor(bsz,slen,endim).zero_().scatter_(-1,inputx,1))
	return onehot


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

class Encoder(nn.Module):
	def __init__(self,ninp=40,nhid=256,nout=128,bidirectional=True,hdrop=0.3):
		super(Encoder, self).__init__()
		self.lockdrop = LockedDropout()

		#encoders
		self.rnns=[]
		mult = 2 if bidirectional else 1
		#input decode LSTM

		#pyramidal LSTM
		self.rnns.append(nn.LSTM(ninp,nhid,1,dropout=0,bidirectional=bidirectional,batch_first=True))
		self.rnns.append(nn.LSTM(nhid*mult*2,nhid,1,dropout=0,bidirectional=bidirectional,batch_first=True))
		self.rnns.append(nn.LSTM(nhid*mult*2,nhid,1,dropout=0,bidirectional=bidirectional,batch_first=True))

		self.rnns = nn.ModuleList(self.rnns)

		#key,value projection
		self.key_proj = nn.Linear(nhid*mult,nout)
		# self.val_proj = nn.Linear(nhid*mult,nout)

		self.nhid = nhid
		self.ninp = ninp
		self.nout = nout
		self.hdrop = hdrop
		self.init_weights()

	def init_weights(self):
		init_range = 0.1
		self.key_proj.bias.data.fill_(0)
		# self.val_proj.bias.data.fill_(0)
		torch.nn.init.xavier_uniform(self.key_proj.weight.data)
		# torch.nn.init.xavier_uniform(self.val_proj.weight.data)


	def forward(self,input,lens):
		var = input
		var = nn.utils.rnn.pack_padded_sequence(var,lens,batch_first=True)
		var,_ = self.rnns[0](var)
		var,lens = nn.utils.rnn.pad_packed_sequence(var,batch_first=True)
		var = var.contiguous()
		var = self.lockdrop(var,0.2).contiguous()

		#second pyramid op
		if var.size()[1] % 2 == 1:
			var = var[:,:-1,:].contiguous()
		var = var.view((int(var.size()[0]),int(var.size()[1]/2),int(var.size()[2]*2)))
		lens = [int(l / 2) for l in lens]
		var = nn.utils.rnn.pack_padded_sequence(var,lens,batch_first=True)
		var,_ = self.rnns[1](var)
		var,lens = nn.utils.rnn.pad_packed_sequence(var,batch_first=True)
		var = var.contiguous()
		var = self.lockdrop(var,0.1).contiguous()

		#third pyramid op
		if var.size()[1] % 2 == 1:
			var = var[:,:-1,:].contiguous()
		var = var.view((int(var.size()[0]),int(var.size()[1]/2),int(var.size()[2]*2)))
		lens = [int(l / 2) for l in lens]
		var = nn.utils.rnn.pack_padded_sequence(var,lens,batch_first=True)
		var,_ = self.rnns[2](var)
		var,lens = nn.utils.rnn.pad_packed_sequence(var,batch_first=True)
		var = var.contiguous()
		#var = self.lockdrop(var,self.hdrop).contiguous()

		listener_feature = self.key_proj(var)
		return listener_feature,lens


class Decoder(nn.Module):
	def __init__(self,ninp=34,ncontext=128,nhid=512):
		super(Decoder, self).__init__()
		self.attention = Attention()

		self.emb = nn.Embedding(ninp,ncontext)

		self.rnn_inith = torch.nn.ParameterList()
		self.rnn_initc = torch.nn.ParameterList()

		# self.rnns = nn.LSTM(2*ncontext,nhid,num_layers=3)
		self.rnns = torch.nn.ModuleList()
		self.rnns.append(torch.nn.LSTMCell(2*ncontext,nhid))
		self.rnn_inith.append(torch.nn.Parameter(torch.rand(1,nhid)))
		self.rnn_initc.append(torch.nn.Parameter(torch.rand(1,nhid)))

		self.rnns.append(torch.nn.LSTMCell(nhid,nhid))
		self.rnn_inith.append(torch.nn.Parameter(torch.rand(1,nhid)))
		self.rnn_initc.append(torch.nn.Parameter(torch.rand(1,nhid)))

		self.rnns.append(torch.nn.LSTMCell(nhid,nhid))
		self.rnn_inith.append(torch.nn.Parameter(torch.rand(1,nhid)))
		self.rnn_initc.append(torch.nn.Parameter(torch.rand(1,nhid)))

		self.query_proj = nn.Linear(nhid,ncontext)

		self.char_proj = []
		self.char_proj.append(nn.Linear(nhid + ncontext,ncontext))
		self.char_proj.append(nn.Linear(ncontext,ninp))
		self.char_proj = nn.ModuleList(self.char_proj)

		# self.char_proj = nn.Linear(2*ncontext,ninp)

		self.lReLU = nn.LeakyReLU()
		self.softmax = nn.LogSoftmax(dim=-1)

		self.emb.weight = self.char_proj[1].weight

		self.ninp = ninp
		self.ncontext = ncontext
		self.nhid = nhid
		self.init_weights()
		if cuda:
			self.rnn_inith = self.rnn_inith.cuda()
			self.rnn_initc = self.rnn_initc.cuda()

	def init_weights(self):
		torch.nn.init.xavier_uniform(self.query_proj.weight.data)
		# torch.nn.init.xavier_uniform(self.char_proj.weight.data)
		# self.char_proj.bias.data = torch.from_numpy(np.load('lb.npy')).float()
		for l in self.char_proj:
			torch.nn.init.xavier_uniform(l.weight.data)
		self.char_proj[1].bias.data = torch.from_numpy(np.load('lb34.npy')).float()
		self.char_proj[0].bias.data.fill_(0)
		#self.char_proj[1].bias.data.fill_(0)


	def forward(self,bsz,slen,listener_feature,lens,ground_truth=None,teacher_force_rate=1):
		if ground_truth is None:
			teacher_force_rate = 0
		else:
			ground_truth = Variable(ground_truth.data,requires_grad=False)

		teacher_force = np.random.random_sample() < teacher_force_rate
		outputs = []
		attentions = []
		hidden = [h.repeat(bsz,1) for h in self.rnn_inith]
		cell = [c.repeat(bsz,1) for c in self.rnn_initc]
		sos = Variable(torch.zeros(bsz).long().fill_(1))
		if cuda:
			sos = sos.cuda()
		output_embed = self.emb(sos)
		query = self.query_proj(hidden[-1]).unsqueeze(1)
		attention, context = self.attention(query,listener_feature,lens)
		rnn_input = torch.cat([output_embed,context],dim=-1)
		for step in range(slen):
			for (j,rnn) in enumerate(self.rnns):
				hidden[j],cell[j] = rnn(rnn_input,(hidden[j],cell[j]))
				rnn_input = hidden[j]
			output = hidden[-1]
			query = self.query_proj(output).unsqueeze(1)
			attention, context = self.attention(query,listener_feature,lens)
			output = torch.cat([output,context],dim=-1)
			for (i,layer) in enumerate(self.char_proj):
				if i ==len(self.char_proj) - 1:
					output = layer(output)
				else:
					output = self.lReLU(layer(output))
			output = self.softmax(output)
			#print(output)
			outputs.append(output.unsqueeze(1))
			attentions.append(attention.unsqueeze(1))
			if teacher_force:
				output_embed = self.emb(ground_truth[:,step].long()).float()
				rnn_input = torch.cat([output_embed,context],dim=-1)
			else:
				_,index = torch.max(output,-1)
				output_embed = self.emb(index).float()
				rnn_input = torch.cat([output_embed,context],dim=-1)
		return outputs, attentions

class Attention(nn.Module):
	def __init__(self):
		super(Attention, self).__init__()
		self.softmax = nn.Softmax(dim=-1)
	def forward(self,query,listener_feature,lens):
		maxseq = max(lens)
		masks = createMasks(lens,maxseq).float()
		# make query of from (n,a) to (n,1,a)
		# query = query.view(query.size()[0],1,query.size()[1])
		# query is already (n,1,a)........
		# get energy key is (n,l,a) needs to be (n,a,l) --> produces (n,1,l) --> squeeze (n,l)
		energy = torch.bmm(query,listener_feature.transpose(1,2)).squeeze(dim=1)
		#softmax for attention
		attention = self.softmax(energy)
		attention = attention * masks
		attention = attention/(attention.sum(1).unsqueeze(1) + 1e-9)

		# attention = attention.unsqueeze(1)
		#get context
		# context = torch.bmm(attention,value)



		context = torch.sum(listener_feature*attention.unsqueeze(2).repeat(1,1,listener_feature.size(2)),dim=1)
		#context is shape (n,1,b)
		return attention,context
		#return attention,context
