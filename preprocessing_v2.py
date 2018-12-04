import numpy as np
import os

train_transcripts = np.load(os.getcwd() + "/data/train_transcripts.npy", encoding='bytes')
dev_transcripts = np.load(os.getcwd() + "/data/dev_transcripts.npy", encoding='bytes')

train_labels = []
dev_labels = []

for tt in train_transcripts:
	temp = " ".join([str(j)[2:-1] for j in tt])
	train_labels.append(temp)

for tt in dev_transcripts:
	temp = " ".join([str(j)[2:-1] for j in tt])
	dev_labels.append(temp)

np.save('./data/train_labels.npy', np.array(train_labels))
np.save('./data/dev_labels.npy', np.array(dev_labels))
