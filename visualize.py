import matplotlib.pyplot as plt
import numpy as np
import scipy.misc


val_attentions = np.load("validation_attentions.npy")

fig = plt.figure()

# idx = int(np.random.randn()*32)
idx = 0
for i in range(val_attentions[idx].shape[1]):
	val_img = scipy.misc.imresize(val_attentions[idx][:,i,:],(512,512))
	plt.imsave(str(i) + "_val_attentions.png",val_img,cmap='hot')
