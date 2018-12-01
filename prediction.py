from torch.autograd import Variable
from LAS_v6 import *
from data.label_dict import *
import os
import operator
import numpy as np
import torch
import torch.nn as nn
cuda = torch.cuda.is_available()
root_f_name = str(os.path.dirname(os.path.realpath(__file__)))
test_inp = np.load(root_f_name+'/data/test.npy')
output_file_name = 'logits.npy'

batch_size = 1
validation_data = np.load(root_f_name + '/data/dev.npy')
validation_transcripts = np.load(root_f_name + '/data/dev_labels.npy')
criterion = nn.CrossEntropyLoss(reduce=False)





def translate(predseq):
	rstr = ""
	# print(predseq.size())
	_,index = torch.max(predseq,dim=-1)
	for ch in index:
		if LABEL_DICTIONARY[ch] == "@":
			break
		else:
			rstr += LABEL_DICTIONARY[ch]
	return rstr

def translate_logits(predlog):
	rstr = ""
	for ch in predlog:
		if LABEL_DICTIONARY[ch] == "@":
			break
		else:
			rstr += LABEL_DICTIONARY[ch]
	return rstr

def convertDict(state_dict):
	new_state_dict = OrderedDict()
	for k,v in state_dict.items():
		name = k[7:]
		new_state_dict = v
	return new_state_dict


def setup():
	file_name_enc = root_f_name + '/hahaenc11.pt'
	file_name_dec = root_f_name + '/hahadec11.pt'
	model_enc_state = torch.load(file_name_enc,map_location=lambda storage, loc: storage)
	model_dec_state = torch.load(file_name_dec,map_location=lambda storage, loc: storage)
	modelEncoder = Encoder()
	modelDecoder = Decoder()
	modelEncoder.load_state_dict(model_enc_state)
	modelDecoder.load_state_dict(model_dec_state)
	return modelEncoder,modelDecoder


def prep_data(data):
	max_seq_len = 0
	max_t_seq_len = 0
	mapping = {}
	for i in range(len(data)):
		tlen = len(data[i])
		mapping[i] = tlen
		if tlen > max_seq_len:
			max_seq_len = tlen
	sorted_tuples = sorted(mapping.items(),key=operator.itemgetter(1),reverse=True)
	rdata = [[[0]*40]*max_seq_len]*len(data)
	rdata_lens = []
	for i in range(len(sorted_tuples)):
		rdata[i] = (data[sorted_tuples[i][0]]).tolist()+(max_seq_len-sorted_tuples[i][1])*[[0]*40]
		rdata_lens.append(sorted_tuples[i][1])
	rdata = np.array(rdata)
	rdata_lens = np.array(rdata_lens)
	# print(sorted_tuples)
	return rdata,rdata_lens, sorted_tuples


def prep_dataval(data,labels):
	max_seq_len = 0
	max_t_seq_len = 0
	mapping = {}
	mapping_t = {}
	for i in range(len(data)):
		tlen = len(data[i])
		t_t_len = len(labels[i])
		mapping[i] = tlen
		mapping_t[i] = t_t_len
		if tlen > max_seq_len:
			max_seq_len = tlen
		if t_t_len > max_t_seq_len:
			max_t_seq_len = t_t_len
	# if max_seq_len % 2 == 1:
	# 	max_seq_len += 1
	sorted_tuples = sorted(mapping.items(),key=operator.itemgetter(1),reverse=True)

	rdata = [[[0]*40]*max_seq_len]*len(data)
	# masks = [[0]*max_seq_len]*len(data)
	rdata_lens = []
	rtrans = [[0]*max_t_seq_len]*len(labels)
	rtrans_lens = []
	rlabels = []
	for i in range(len(sorted_tuples)):
		if max_seq_len - sorted_tuples[i][1] == 0:
			rdata[i] = (data[sorted_tuples[i][0]]).tolist()
		else:
			rdata[i] = (data[sorted_tuples[i][0]]).tolist()+(max_seq_len-sorted_tuples[i][1])*[[0]*40]
		# masks[i][:max_seq_len-sorted_tuples[i][1]] = (max_seq_len-sorted_tuples[i][1])*[1]
		rdata_lens.append(sorted_tuples[i][1])
		if max_t_seq_len - mapping_t[sorted_tuples[i][0]] == 0:
			rtrans[i] = labels[sorted_tuples[i][0]]
		else:
			rtrans[i] = (labels[sorted_tuples[i][0]]).tolist() + (max_t_seq_len - mapping_t[sorted_tuples[i][0]])*[0]
		rtrans_lens.append(mapping_t[sorted_tuples[i][0]]-1)
		rlabels.extend(labels[sorted_tuples[i][0]][1:])
	rdata = np.array(rdata)
	rdata_lens = np.array(rdata_lens)
	rtrans = np.array(rtrans)
	rtrans_lens = np.array(rtrans_lens)
	rlabels = np.array(rlabels)
	# masks = np.array(masks)

	return rdata,rdata_lens,rtrans,rlabels,rtrans_lens


def getbatch(data,i,batch_size):
	batch_size = min(batch_size,data.shape[0] - i)
	xd,xl,usort= prep_data(data[i:i+batch_size])
	xd = Variable(torch.from_numpy(xd)).float().contiguous()
	# masks = torch.from_numpy(masks)
	if cuda:
		xd = xd.cuda()
	return xd,xl,usort

def getbatchval(data,labels,i,batch_size):
	batch_size = min(batch_size,data.shape[0] - i)
	xd,xl,td,ty,tl = prep_dataval(data[i:i+batch_size],labels[i:i+batch_size])
	xd = Variable(torch.from_numpy(xd)).float().contiguous()
	tx = Variable(torch.from_numpy(td)).long().contiguous()
	ty = Variable(torch.from_numpy(ty)).long().contiguous()
	# masks = torch.from_numpy(masks)
	if cuda:
		xd = xd.cuda()
		tx = tx.cuda()
		ty = ty.cuda()
	return xd,xl,tx,ty,tl


def validate():
	modelEncoder.eval()
	modelDecoder.eval()
	total_loss = 0
	att = []
	outs= []
	for i in range(0,validation_data.shape[0],batch_size):
		print(i)
		xd,xl,tx,ty,tl = getbatchval(validation_data,validation_transcripts,i,batch_size)
		k,v,ul = modelEncoder(xd,xl)
		ps,att = modelDecoder(tx.size(0),tx.size(1)-1,k,v,ul,ground_truth=tx[:,1:])
		# ps = unpack_concatenate_predseq(ps,tl)
		outs = torch.cat(ps,dim=1).data
		ps = torch.cat(ps,dim=1).view(-1,34)
		ty = tx[:,1:].contiguous().view(-1)
		masks = createMasks(tl,max(tl)).view(-1).float()
		loss = criterion(ps,ty)
		loss = (loss*masks).sum()
		total_loss += loss.data
	att = torch.cat(att,dim=1)
	np.save('outputs.npy',outs.cpu().numpy())
	np.save('attentions.npy',att.data.cpu().numpy())
	total_loss = total_loss[0] / validation_data.shape[0]
	print("Validation Loss: " + str(total_loss))
	return total_loss

def writeData(logits,filename="submission.csv"):
	f = open(filename,"w")
	f.write("Id,Predicted\n")
	for i,s in enumerate(logits):
		f.write(str(i) + "," + s + "\n")
	f.close()

def prediction():
	modelEncoder, modelDecoder = setup()
	modelEncoder.eval()
	modelDecoder.eval()
	preds = np.empty(len(test_inp),dtype=object)
	i = 0
	while i < len(test_inp):
		print(i)
		xd,xl,usort = getbatch(test_inp,i,batch_size)
		keys,values,ul = modelEncoder(xd,xl)
		nkeys,nvalues = [],[]
		genouts = []
		lens = []
		for j in range(25):
			nkeys.append(keys)
			nvalues.append(values)
			ps, att = modelDecoder(len(xl),500,keys,values,ul,ground_truth=None)
			ps = torch.cat(ps,dim=1)
			_,logits = torch.max(ps,dim=-1)
			# print(preds)
			logits[0,-1] = 2
			idx = logits[0].data.numpy().tolist().index(2)
			genouts.append(logits)
			lens.append(idx)
		genouts = torch.cat(genouts,dim=0)
		nkeys = torch.cat(nkeys,dim=0)
		nvalues = torch.cat(nvalues,dim=0)

		masks = createMasks(lens,500).view(-1).float()
		ps,att = modelDecoder(25,500,nkeys,nvalues,ul,ground_truth=genouts)
		ps = torch.cat(ps,dim=1).view(-1,34)
		loss = criterion(ps,genouts.view(-1))
		perseq = (masks*loss).view(-1,500).sum(1)
		_,select = torch.min(perseq,dim=0)
		rstr = translate_logits(genouts[select].data[0])
		print(rstr)
		preds[i] = rstr

		# ps,att = modelDecoder(len(xl),200,keys,values,ul)
		# ps = torch.cat(ps,dim=1)
		# # print(ps)
		# for k in range(len(usort)):	
		# 	index = i + usort[k][0]
		# 	preds[index] = translate(ps[k].data)



		i+= batch_size
	writeData(preds)
	# np.save(output_file_name,preds)

modelEncoder,modelDecoder = setup()
# validate()
prediction()
