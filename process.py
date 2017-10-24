import numpy as np

arr = range(1,81)

print(arr)

save=np.loadtxt(open("./hybrid.csv", "rb"), delimiter=",", skiprows=1, usecols = arr)

print(np.shape(save))
print(save[0])

nrow,ncol = np.shape(save)

for i in range(0,40):
	cut = np.mean(save[:,i])
	for j in range(0,nrow):
		if(save[j][i]>cut):
			save[j][i] = 1
		else:
			save[j][i] = 0

print(save[0])
np.savetxt('./clean.data',save,delimiter=",")
