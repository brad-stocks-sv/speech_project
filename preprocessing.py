import numpy as np
import os

train_transcripts = np.load(os.getcwd() + "/data/train_transcripts.npy")
dev_transcripts = np.load(os.getcwd() + "/data/dev_transcripts.npy")

def intefy(transcripts,label_map,prepend):
	nlist = []
	for tt in transcripts:
		tlist = [len(label_map) - 2]
		for c in tt:
			tlist.append(label_map[c])
		tlist.append(len(label_map)-1)
		nlist.append(np.array(tlist))
	nlist = np.array(nlist)
	np.save(os.getcwd() + "/data/" + prepend + "_labels.npy",nlist)

characters = {}
total_chars = 0
for tt in train_transcripts:
	for c in tt:
		total_chars += 1
		if c not in characters:
			characters[c] = 1
		else:
			characters[c] += 1
#print("(" in characters)
#print(")" in characters)
lb34 = []
for c in sorted(characters.keys()):
	lb34.append(characters[c]/total_chars)
lb34.append(1)
lb34.append(1)
lb34 = np.log(np.array(lb34,dtype="float32"))
#print(type(lb34))
np.save("data/lb34.npy",lb34)

label_map = {s:i for i,s in enumerate(sorted(characters.keys()))}
label_map["("] = len(label_map)
label_map[")"] = len(label_map)
np.save(os.getcwd() + "/data/label_map.npy",label_map)
intefy(train_transcripts,label_map,"train")
intefy(dev_transcripts,label_map,"dev")

