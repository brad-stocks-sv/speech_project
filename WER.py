import csv
import numpy as np
# from jiwer import wer
import Levenshtein as lev

transcript_path = './nb_dev_transcripts.npy'
LSTM_path = './output_LSTM/LSTM_submission.csv'
Conv_path = './output_densenet/hw4-LAS-baseline/v1/submission.csv'

transcript = np.load(transcript_path)

conv_generated = []
lstm_generated = []
with open(Conv_path) as f:
	csv_reader = csv.reader(f, delimiter=',')
	for row in csv_reader:
		conv_generated.append(row)

with open(LSTM_path) as f:
	csv_reader = csv.reader(f, delimiter=',')
	for row in csv_reader:
		lstm_generated.append(row)

print(len(lstm_generated))
print(len(conv_generated))
lstm_generated = lstm_generated[1:]
conv_generated = conv_generated[1:]


l_cer = 0
c_cer = 0
for i in range(len(lstm_generated)):
	l_cer += float(lev.distance(lstm_generated[i][1],transcript[i]))/len(transcript[i])
	c_cer += float(lev.distance(conv_generated[i][1],transcript[i]))/len(transcript[i])
l_cer = l_cer/len(lstm_generated)
c_cer = c_cer/len(lstm_generated)
print(l_cer)
print(c_cer)
	# lstm_generated[i] = lstm_generated[i][1]
	# conv_generated[i] = conv_generated[i][1]

# print(conv_generated)
# print(lstm_generated[120])
# print(conv_generated[120])
# print(transcript[120])

# print(lev.distance("haha",))

# print(wer(lstm_generated[120], str(transcript[120]))) #= 0.43333 /161 = 0.00269
# print(wer(conv_generated[120], str(transcript[120]))) #= 0.4202 / 160 = 0.002626
# print(lev.distance('irs time', 'itt time'))