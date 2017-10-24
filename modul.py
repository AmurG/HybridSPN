import math
import numpy as np
import scipy
import networkx as nx
from scipy.stats import multivariate_normal as mn
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hcluster
import nodes as nd
from nodes import *
from data import *
from sklearn import mixture
import matplotlib.pyplot as plt
from scipy.cluster.vq import vq, kmeans, whiten
from time import time

def returnarr(arr,scope):
	q = []
	te = list(scope)
	te = sorted(te)
	for i in arr:
		q.append(te[i])
	return set(q)

Leafcount = 0

globalcnt = 0
nodes = []

def induce(tempdat,maxsize,scope,indsize,flag):
	full = len(tempdat)
	if (flag==0):
		if (full>=30*len(scope)):
			tempdat2 = split(tempdat,8)
			s = sumNode()
			#global nodes
			#nodes.append(globalcnt,s.kind)
			#globalcnt = globalcnt + 1
			arr = []
			cnt = 0
			for i in range(0,len(tempdat2)):
				if(len(tempdat2[i])>=(2*len(scope))):
					arr.append(len(tempdat2[i]))
					s.children.append(induce(np.asarray(tempdat2[i]),maxsize,scope,indsize,1))
					cnt = cnt + 1
			
			for i in range(0,cnt):
				chosen = s.children[i]
				w = 0
				for j in chosen.children:
					s.children.append(j)
					arr.append(chosen.wts[w]*arr[i])
					w = w+1
			arr = arr[cnt:]
			s.children = s.children[cnt:]			
			s.setwts(arr)
			print("wts are",arr)
			return s
	effdat = np.zeros(len(tempdat)*len(scope))
	effdat = np.reshape(effdat,(len(tempdat),len(scope)))
	for i in range(0,len(tempdat)):
		temp = submean(tempdat[i],scope)
		for j in range(0,len(scope)):
			effdat[i][j] = temp[j]
	effcorr = np.corrcoef(np.transpose(effdat))
	effcov = np.cov(np.transpose(effdat))
	print(np.shape(effcorr))
	print(np.shape(effcov))
	empmean = np.mean(effdat,axis=0)
	print(np.shape(empmean))

	G = nx.from_numpy_matrix(-abs(effcorr))
	G = G.to_undirected()

	Dec = []

	T=nx.minimum_spanning_tree(G)
	Order = np.asarray(T.edges(data='weight'))
	k = len(Order)
	wts = np.zeros(k)
	Order = Order[Order[:,2].argsort()]
	Dec = []
	Gc = max(nx.connected_component_subgraphs(T), key=len)
	n = Gc.number_of_nodes()
	if(n<=maxsize):
		Dec.append(list(nx.connected_components(T)))
	
	#count = 0

	for i in range(0,k):
		sum = 0
		for j in range(0,len(Order)-i):
			sum = sum - Order[j,2]
		wts[i] = sum - np.log(k-i) + np.log(k)
		idx = int(Order[len(Order)-i-1,0])
		idx2 = int(Order[len(Order)-i-1,1])
		T.remove_edge(idx,idx2)
		Gc = max(nx.connected_component_subgraphs(T), key=len)
		n = Gc.number_of_nodes()
		if(n<=maxsize):
			Dec.append(list(nx.connected_components(T)))
			#count = count+1

	#wts[k-1]=0.1
	effwts = np.zeros(len(Dec))
	for i in range(0,len(Dec)):
		effwts[i] = wts[i+k-len(Dec)]

	s = sumNode()
	s.setwts(effwts)
	print(effwts)

	print(Dec)

	for i in range(0,len(Dec)):
		p = prodNode()
		s.children.append(p)
		for j in (Dec[i]):
			print(j)
			sub = returnarr(j,scope)
			print(sub)
			if (len(j)<=indsize):
				l = leafNode()
				global Leafcount
				Leafcount = Leafcount+1
				tempmean = submean(empmean,j)
				tempcov = submat(effcov,j)
				l.scope = sub
				l.create(tempmean,tempcov)
				p.children.append(l)
			else:
				p.children.append(induce(tempdat,maxsize-2,sub,indsize,0))
		

	return s

#test
'''
s = set(xrange(784))

ab=np.loadtxt(open("../train.csv", "rb"), delimiter=",", skiprows=1)

blank = []
for i in range(0,42000):
	if(ab[i][0]==2):
		blank.append(ab[i,1:])

blank2 = []

for i in range(0,42000):
	if(ab[i][0]==7):
		blank2.append(ab[i,1:])


print(np.shape(np.asarray(blank)))
ab = np.asarray(blank)

for i in range(0,len(ab)):
	for j in range(0,784):
		draw = np.random.uniform(0.2,0.6)
		ab[i][j] = ab[i][j] + draw

Tst = induce(ab[:,:],400,s,10,0,20)


for i in range(0,8000):
	t = time()
	idx = np.random.randint(0,len(ab))
	nd.globalarr = ab[idx]
	Tst.passon()
	placeholder = Tst.retval()
	Tst.update()
	print(time()-t)

sum = 0

plot1 = np.zeros(800)

for i in range(0,800):
	nd.globalarr = ab[i]
	Tst.passon()
	sum = sum + Tst.retval()
	plot1[i] = Tst.retval()


print(sum/800)
print(np.amax(plot1),np.amin(plot1))

ab = np.asarray(blank2)

sum = 0

plot1 = np.zeros(800)

for i in range(0,800):
	nd.globalarr = ab[i]
	Tst.passon()
	sum = sum + Tst.retval()
	plot1[i] = Tst.retval()

print(sum/800)
print(np.amax(plot1),np.amin(plot1))


values = []

s = set(xrange(8))

ab = np.genfromtxt('../AB.dat',delimiter=",")
ab = np.asarray(ab[:,1:])
ab = whiten(ab)
print(len(ab))





#for i in range(0,len(ab)):
#	for j in range(0,22):
#		ab[i][j] = ab[i][j] + 1e-6


for w in range(0,2):

	ab = np.random.permutation(ab)

	Tst = induce(ab[:3600,:],8,s,2,0)


	for i in range(0,7200):
		t = time()
		idx = np.random.randint(0,3600)
		nd.globalarr = ab[idx]
		Tst.passon()
		placeholder = Tst.retval()
		Tst.update()
		print(time()-t)

	sum = 0	

	plot1 = np.zeros(400)

	for i in range(3600,4000):
		nd.globalarr = ab[i]
		Tst.passon()
		sum = sum + Tst.retval()
		plot1[i-3600] = Tst.retval()

	values.append((sum/400))
	print(values)

print(values)
'''
