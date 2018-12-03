#from model import *
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import operator
from LAS_v8 import *
from tqdm import tqdm
import time
from tensorboardX import SummaryWriter
import shutil
import matplotlib.pyplot as plt

label_map = {0: ' ', 1: "'", 2: '+', 3: '-', 4: '.', 5: 'A', 6: 'B', 7: 'C', 8: 'D', 9: 'E', 10: 'F', 11: 'G', 12: 'H', 13: 'I', 14: 'J', 15: 'K', 16: 'L', 17: 'M', 18: 'N', 19: 'O', 20: 'P', 21: 'Q', 22: 'R', 23: 'S', 24: 'T', 25: 'U', 26: 'V', 27: 'W', 28: 'X', 29: 'Y', 30: 'Z', 31: '_', 32: '(', 33: ')'}
cuda = torch.cuda.is_available()
batch_size = 32
grad_clip = 0.25
log_step = 50

load = False
save_fileenc = "encoder.pt"
save_filedec = "decoder.pt"
epoch = 100

LSTM_encoder = True

batch_times = []
train_losses = []
val_losses = []

val_strings = []
val_labels = []

train_steps = 0
val_steps = 0

if not os.path.isdir("logs"):
	os.mkdir("logs")
else:
	shutil.rmtree("logs")
writer = SummaryWriter(log_dir="logs")

def repackage_hidden(h):
    if isinstance(h, Variable):
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

def unpack_concatenate_predseq(predseq, lens):
	retseq = []
	for i,pred in enumerate(predseq):
		retseq.append(pred[:lens[i],:])
	ret = torch.cat(retseq,dim=0)
	return ret.view(-1,ret.size()[-1])

def translate_2(predseq,labelseq,lens):
	rstr = ""
	labelstr = ""
	for p,l,seqlen in zip(predseq.cpu().numpy(),labelseq.cpu().numpy(),lens):
		p = p[:seqlen]
		l = l[:seqlen]
		for charx,chary in zip(p,l):
			rstr += label_map[charx]
			labelstr += label_map[chary]
		rstr += "\n"
		labelstr += "\n"
	# else:
	# 	writer.add_text("validation_predictions",labelstr)
	return rstr,labelstr

def translate(predseq,labelseq):
	rstr = ""
	labelstr = ""
	_,index = torch.max(predseq,dim=-1)
	for p,l in zip(index,labelseq):
		rstr += label_map[p]
		labelstr += label_map[l]
	# print("PREDICTED SEQUENCE: \n" + rstr)
	# print("LABEL SEQUENCE: \n" + labelstr)
	return rstr,labelstr

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
if load:
	modelEncoder.load_state_dict(torch.load(save_fileenc))
	modelDecoder.load_state_dict(torch.load(save_filedec))

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
	global val_steps
	modelEncoder.eval()
	modelDecoder.eval()
	total_loss = 0
	val_attentions = []
	val_preds = []
	val_labels = []
	total_loss = 0
	for i in tqdm(range(0,validation_data.shape[0],batch_size)):
		start = time.time()
		acoustic_features,acoustic_lens,full_labels,_,label_lens = getbatch(validation_data,validation_transcripts,i,batch_size)
		if len(acoustic_lens) == 0:
			continue
		keys,values,enc_lens = modelEncoder(acoustic_features,acoustic_lens)
		logits,attentions,generated = modelDecoder(keys,values,enc_lens,full_labels[:,:-1])
		masks = createMasks(label_lens,max(label_lens)).float().unsqueeze(2)
		logits = (logits.transpose(1,0) * masks).contiguous()
		loss = criterion(logits.view(-1,logits.size(2)),full_labels[:,1:].contiguous().view(-1))
		loss = torch.sum(masks.view(-1) * loss)/logits.size(1)
		val_losses.append(loss.item())
		end = time.time()
		batch_times.append(end - start)
		total_loss += loss.item()
		val_attentions.append(attentions.cpu().detach().numpy())
		pred_string,true_string = translate_2(generated.detach(),full_labels,label_lens)
		fig = plt.figure()
		if i % log_step == 0:
			scaling = np.max(val_attentions[-1][0])
			writer.add_image("validation_attention",((val_attentions[-1][0])/scaling)*255, val_steps)
		writer.add_scalar("validation_loss",loss.item(),val_steps)
		writer.add_text("validation_predictions",pred_string, val_steps)
		val_preds.append(pred_string)
		val_labels.append(true_string)
		val_steps += 1
	np.save("validation_predictions.npy",val_preds)
	np.save("validation_attentions.npy",val_attentions)
	np.save("validation_losses.npy",val_losses)
	if os.path.isfile("validation_labels.npy"):
		np.save("validation_labels.npy",val_labels)
	total_loss = total_loss / validation_data.shape[0]
	print("Validation Loss: " + str(total_loss))
	return total_loss

def train():
	global train_steps
	train_strings = []
	train_labels = []
	train_attentions = []
	modelEncoder.train()
	modelDecoder.train()
	for i in tqdm(range(0,training_data.shape[0],batch_size)):
		start = time.time()
		optimizer.zero_grad()
		acoustic_features,acoustic_lens,full_labels,labels,label_lens = getbatch(training_data,training_transcripts,i,batch_size)
		if len(acoustic_lens) == 0:
			continue
		keys,values,enc_lens = modelEncoder(acoustic_features,acoustic_lens)
		logits,attentions,generated = modelDecoder(keys,values,enc_lens,full_labels[:,:-1])
		masks = createMasks(label_lens,max(label_lens)).float().unsqueeze(2)
		logits = (logits.transpose(1,0) * masks).contiguous()
		loss = criterion(logits.contiguous().view(-1,logits.size(2)),full_labels[:,1:].contiguous().view(-1))
		loss = torch.sum(masks.view(-1) * loss)/logits.size(1)
		loss.backward()
		torch.nn.utils.clip_grad_norm(list(modelEncoder.parameters())+list(modelDecoder.parameters()),grad_clip)
		optimizer.step()
		train_losses.append(loss.item())
		end = time.time()
		batch_times.append(end - start)
		pred_string,true_string = translate_2(generated.detach(),full_labels,label_lens)
		train_attentions.append(attentions.cpu().detach().numpy())
		if i % log_step == 0:
			scaling = np.max(train_attentions[-1][0])
			writer.add_image("train_attention",(train_attentions[-1][0]/scaling)*255, train_steps)
		writer.add_text("train_predictions",pred_string, train_steps)
		writer.add_scalar("train_loss",loss.item(),train_steps)
		writer.add_scalar("batch_time",batch_times[-1],train_steps)
		train_strings.append(pred_string)
		train_labels.append(true_string)
		train_steps += 1
	np.save("train_predictions.npy",np.array(train_strings))
	np.save("train_attentions.npy",np.array(train_attentions))
	np.save("train_losses.npy",np.array(train_losses))
	np.save('timers.npy', np.array(batch_times))
	if os.path.isfile("train_labels.npy"):
		np.save("train_labels.npy",np.array(train_labels))


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
                # with open(save_fileenc,'wb') as f:
                #         torch.save(modelEncoder.state_dict(),f)
                # with open(save_filedec,'wb') as f:
                #         torch.save(modelDecoder.state_dict(),f)
