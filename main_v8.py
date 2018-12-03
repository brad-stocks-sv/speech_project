#from model import *
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import operator
from LAS_v8 import *
# import gc
cuda = torch.cuda.is_available()
batch_size = 32
grad_clip = 0.25
log_step = 50

LSTM_encoder = True


def repackage_hidden(h):
    if isinstance(h, Variable):
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


# def compute_loss(logits,target,length):
# 	logits_flat = logits.view(-1,logits.size(-1))
# 	logits_probs_flat = torch.nn.functional.log_softmax(logits_flat)
# 	target_flat = target.view(-1,1)
# 	loss_flat = -torch.gather(log_probs_flat,dim=1,index=target_flat)
# 	losses = losses_flat.view(*target.size())
# 	mask =_sequence_mask(sequence_length=length,max_len=target.size(1))
# 	losses = losses * mask.float()
# 	loss = losses.sum()/length.float().sum()
# 	return loss


def unpack_concatenate_predseq(predseq, lens):
	retseq = []
	for i,pred in enumerate(predseq):
		retseq.append(pred[:lens[i],:])
	ret = torch.cat(retseq,dim=0)
	return ret.view(-1,ret.size()[-1])

def translate_2(predseq,labelseq,lens):
	rstr = ""
	labelstr = ""
	for p,l,seqlen in zip(predseq,labelseq,lens):
		p = p[:seqlen]
		l = l[:seqlen]
		_,index = torch.max(p,dim=-1)
		for charx,chary in zip(index,l):
			rstr += LABEL_DICTIONARY[charx]
			labelstr += LABEL_DICTIONARY[chary]
	print("PREDICTED: \n" + rstr)
	print("LABEL SEQUENCE: \n" + labelstr)



def translate(predseq,labelseq):
	rstr = ""
	labelstr = ""
	_,index = torch.max(predseq,dim=-1)
	for p,l in zip(index,labelseq):
		rstr += LABEL_DICTIONARY[p]
		labelstr += LABEL_DICTIONARY[l]
	print("PREDICTED SEQUENCE: \n" + rstr)
	print("LABEL SEQUENCE: \n" + labelstr)

def get_train_data():
	current_dir = os.path.dirname(os.path.realpath(__file__))
	data_path = current_dir + '/data'
	training_data = np.load(data_path+'/train.npy')
	training_labels = np.load(data_path+'/train_labels.npy')
	return training_data,training_labels

def get_validation_data():
	current_dir = os.path.dirname(os.path.realpath(__file__))
	data_path = current_dir + '/data'
	dev_data = np.load(data_path+'/dev.npy')
	dev_labels = np.load(data_path+'/dev_labels.npy')
	return dev_data,dev_labels

def prep_data(data,labels):
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


def getbatch(data,labels,i,batch_size):
	bsz = min(batch_size,data.shape[0] - i)
	xd,xl,td,ty,tl = prep_data(data[i:i+bsz],labels[i:i+bsz])
	xd = Variable(torch.from_numpy(xd)).float().contiguous()
	tx = Variable(torch.from_numpy(td)).long().contiguous()
	ty = Variable(torch.from_numpy(ty)).long().contiguous()
	xl = Variable(torch.from_numpy(xl))
	# masks = torch.from_numpy(masks)
	if cuda:
		xd = xd.cuda()
		tx = tx.cuda()
		ty = ty.cuda()
		xl = xl.cuda()
	return xd,xl,tx,ty,tl



if LSTM_encoder:
	modelEncoder = EncoderModel()
else:
	modelEncoder = ConvEncoder()
modelDecoder = Decoder()
print(modelEncoder)
print(modelDecoder)
if cuda:
	modelEncoder = modelEncoder.cuda()
	modelDecoder = modelDecoder.cuda()

training_data,training_transcripts = get_train_data()
validation_data, validation_transcripts = get_validation_data()
optimizer = torch.optim.Adam(list(modelEncoder.parameters())+ list(modelDecoder.parameters()),lr=1e-3,weight_decay=1e-5)
criterion = nn.CrossEntropyLoss(reduce=False)



def validate():
	modelEncoder.eval()
	modelDecoder.eval()
	total_loss = 0
	all_attentions = []
	all_generated= []
	for i in range(0,validation_data.shape[0],batch_size):
		print(i)
		acoustic_features,acoustic_lens,full_labels,_,label_lens = getbatch(validation_data,validation_transcripts,i,batch_size)
		if len(acoustic_lens) == 0:
			continue
		keys,values,enc_lens = modelEncoder(acoustic_features,acoustic_lens)
		logits,attentions,generated = modelDecoder(keys,values,enc_lens,full_labels[:,:-1])
		masks = createMasks(label_lens,max(label_lens)).float().unsqueeze(2)
		logits = (logits.transpose(1,0) * masks).contiguous()
		loss = criterion(logits.view(-1,logits.size(2)),full_labels[:,1:].contiguous().view(-1))
		loss = torch.sum(masks.view(-1) * loss)/logits.size(1)
		total_loss += loss.item()
		all_attentions.append(attentions.cpu().detach().numpy())
		all_generated.append(generated.cpu().detach().numpy())
	#all_attentions = torch.cat(all_attentions,dim=0)
	#outputs = torch.cat(outputs,dim=0)
	np.save('generated.npy',all_generated)
	np.save('attentions.npy',all_attentions)
	total_loss = total_loss / validation_data.shape[0]
	print("Validation Loss: " + str(total_loss))
	return total_loss

def train():
	modelEncoder.train()
	modelDecoder.train()
	total_loss = 0
	print(training_data.shape[0])
	for i in range(0,training_data.shape[0],batch_size):
		print(i)
		# print(batch_size)
		optimizer.zero_grad()
		acoustic_features,acoustic_lens,full_labels,labels,label_lens = getbatch(training_data,training_transcripts,i,batch_size)
		if len(acoustic_lens) == 0:
			continue
		# print(acoustic_features.size())
		# print(acoustic_lens)
		# print(full_labels.size())
		# print(label_lens)
		keys,values,enc_lens = modelEncoder(acoustic_features,acoustic_lens)
		logits,attentions,generated = modelDecoder(keys,values,enc_lens,full_labels[:,:-1])
		masks = createMasks(label_lens,max(label_lens)).float().unsqueeze(2)
		logits = (logits.transpose(1,0) * masks).contiguous()
		loss = criterion(logits.contiguous().view(-1,logits.size(2)),full_labels[:,1:].contiguous().view(-1))
		loss = torch.sum(masks.view(-1) * loss)/logits.size(1)
		loss.backward()
		torch.nn.utils.clip_grad_norm(list(modelEncoder.parameters())+list(modelDecoder.parameters()),grad_clip)
		optimizer.step()
		total_loss += loss.item()
		if i % log_step == 0 and i > 0:
			curr_loss = total_loss / log_step
			print(curr_loss)
			total_loss = 0
			#translate(ps.data,ty.data)


save_fileenc = "encoder.pt"
save_filedec = "decoder.pt"
epoch = 100
best_loss = validate()

for e in range(epoch):
        train()
        curr_loss = validate()
        if best_loss > curr_loss:
                with open(save_fileenc,'wb') as f:
                        torch.save(modelEncoder.state_dict(),f)
                with open(save_filedec,'wb') as f:
                        torch.save(modelDecoder.state_dict(),f)
                best_loss = curr_loss
        else:
                print("Overfit?")
                with open(save_fileenc,'wb') as f:
                        torch.save(modelEncoder.state_dict(),f)
                with open(save_filedec,'wb') as f:
                        torch.save(modelDecoder.state_dict(),f)
