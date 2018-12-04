import numpy as np
import os

train_transcripts = np.load(os.getcwd() + "/data/train_transcripts.npy", encoding='bytes')
dev_transcripts = np.load(os.getcwd() + "/data/dev_transcripts.npy", encoding='bytes')


label_map = {0: ' ', 1: "'", 2: '+', 3: '-', 4: '.', 5: 'A', 6: 'B', 7: 'C', 8: 'D', 9: 'E', 10: 'F', 11: 'G', 12: 'H', 13: 'I', 14: 'J', 15: 'K', 16: 'L', 17: 'M', 18: 'N', 19: 'O', 20: 'P', 21: 'Q', 22: 'R', 23: 'S', 24: 'T', 25: 'U', 26: 'V', 27: 'W', 28: 'X', 29: 'Y', 30: 'Z', 31: '_', 32: '(', 33: ')'}
char_map = {' ': 0, "'": 1, '+': 2, '-': 3, '.': 4, 'A': 5, 'B': 6, 'C': 7, 'D': 8, 'E': 9, 'F': 10, 'G': 11, 'H': 12, 'I': 13, 'J': 14, 'K': 15, 'L': 16, 'M': 17, 'N': 18, 'O': 19, 'P': 20, 'Q': 21, 'R': 22, 'S': 23, 'T': 24, 'U': 25, 'V': 26, 'W': 27, 'X': 28, 'Y': 29, 'Z': 30, '_': 31, '(': 32, ')': 33}
# train_labels = []
# dev_labels = []

# for tt in train_transcripts:
# 	temp = " ".join([str(j)[2:-1] for j in tt])
# 	train_labels.append(temp)

# for tt in dev_transcripts:
# 	temp = " ".join([str(j)[2:-1] for j in tt])
# 	dev_labels.append(temp)

# np.save('./data/train_labels.npy', np.array(train_labels))
# np.save('./data/dev_labels.npy', np.array(dev_labels))
def transcripts_unbyte(transcripts,prepend):
	sentences = []
	for t in transcripts:
		sentence = ""
		for word in t:
				sentence+= word.decode("utf-8") + " "
		sentence = "(" + sentence[:-1] + ")"
		sentences.append(sentence)
	np.save("/data/nb_" + prepend + "_transcripts.npy",np.array(sentences))
	return sentences

def transcripts_intefy(transcripts,prepend):
	labels = []
	for t in transcripts:
		l = []
		for ch in t:
			l.append(char_map[ch])
		print(l)
		labels.append(l)
	np.save("/data/" + prepend + "_labels.npy",np.array(labels))

ttnb = transcripts_unbyte(train_transcripts,"train")
dtnb =transcripts_unbyte(dev_transcripts,"dev")
transcripts_intefy(ttnb,"train")
transcripts_intefy(dtnb,"dev")


