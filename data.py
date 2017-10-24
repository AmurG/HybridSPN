#Some helper functions to make the entire thing run

import math
import numpy as np
import scipy
import scipy.cluster.hierarchy as hcluster
from scipy.cluster.vq import vq, kmeans, whiten

def split(arr,k,scope):
	pholder,clusters = scipy.cluster.vq.kmeans2(arr[:,sorted(list(scope))],k,minit='points')
	print ("clusters",clusters)
	big = []
	for i in range(0,len(set(clusters))):
		small = []
		for j in range(0,len(arr)):
			if (clusters[j]==i):
				small.append(arr[j,:])
		big.append(small)
		#print(big)
	return big

def submat(mat,subset):
	q = len(subset)
	print(q)
	ret = np.zeros(q*q)
	ret = np.reshape(ret,(q,q))
	w = 0
	for i in subset:
		z = 0
		for j in subset:
			ret[w,z] = mat[i,j] 
			z=z+1
		w=w+1
	return ret

def submean(mean,subset):
	q = len(subset)
	m = np.zeros(q)
	w = 0
	for i in subset:
		m[w] = mean[i]
		w=w+1
	return m





