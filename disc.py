import numpy as np
import nodes as nd
import networkx as nx
from nodes import *
from modul import returnarr
from data import *
from sklearn.metrics import adjusted_mutual_info_score as ami
from time import time
from sklearn import metrics
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


Leafcount = 0

def infmat(mat,nvar):
	retmat = np.zeros(nvar*nvar)
	retmat = np.reshape(retmat,(nvar,nvar))
	for i in range(0,nvar):
		for j in range(0,nvar):
			if (i>j):
				retmat[i][j] = retmat[j][i]
			else:
				retmat[i][j] = (float(np.dot(mat[:,i],mat[:,j])*np.dot(mat[:,i],mat[:,j]) + 1e-4))/(float(np.dot(mat[:,i],mat[:,i])*np.dot(mat[:,j],mat[:,j]) + 1e-2))
				#temp = np.corrcoef(retmat[:,i],retmat[:,j])
				#retmat[i][j] = abs(temp[0][1])
	return retmat



def createpdf(mat,nsam,nvar):
	length = int(np.rint(np.power(2,nvar)))
	pdf = np.zeros(length)
	for i in range(0,nsam):
		#print(mat[i,:])
		idx = bintodec(mat[i,:])
		#print(idx)
		pdf[idx] = pdf[idx] + float((0.95)/nsam)
	for i in range(0,length):
		pdf[i] = pdf[i] + float(0.05/float(length))
	return pdf


def induce(tempdat,maxsize,scope,indsize,flag,maxcount,typearr):
	full = len(tempdat)
	
	if (flag==0):
		if (full>=3000):
			tempdat2 = split(tempdat,2,scope)
			s = sumNode()
			arr = []
			cnt = 0
			for i in range(0,len(tempdat2)):
				if(len(tempdat2[i])>=600):
					arr.append(len(tempdat2[i]))
					s.children.append(induce(np.asarray(tempdat2[i]),maxsize,scope,indsize,1,maxcount,typearr))
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
			#print("wts are",arr)
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

	fisher = infmat(effdat,len(scope))
	G = nx.from_numpy_matrix(-abs(fisher))
	G = G.to_undirected()

	Dec = []

	T=nx.minimum_spanning_tree(G)
	Order = np.asarray(T.edges(data='weight'))
	k = len(Order)
	#wts = np.zeros(k)
	Order = Order[Order[:,2].argsort()]
	Dec = []
	print(Order)	
	Gc = max(nx.connected_component_subgraphs(T), key=len)
	n = Gc.number_of_nodes()
	if(n<=maxsize):
		Dec.append(list(nx.connected_components(T)))

	count = 0
	
	for i in range(0,k):
		if(count>maxcount):
			break
		#sum = 0
		#for j in range(0,len(Order)-i):
		#	sum = sum - Order[j,2]
		#wts[i] = sum 
		idx = int(Order[len(Order)-i-1,0])
		idx2 = int(Order[len(Order)-i-1,1])
		T.remove_edge(idx,idx2)
		Gc = max(nx.connected_component_subgraphs(T), key=len)
		n = Gc.number_of_nodes()
		if((n<=maxsize)and(count<=maxcount)):
			Dec.append(list(nx.connected_components(T)))
			count = count + 1

	effwts = np.zeros(len(Dec))
	for i in range(0,len(Dec)):
		effwts[i] = 1./len(Dec)

	s = sumNode()
	s.setwts(effwts)
	print(effwts)

	print(Dec)

	for i in range(0,len(Dec)):
		if(len(Dec[i])>1):	
			p = prodNode()
			s.children.append(p)
			for j in (Dec[i]):
				sub = returnarr(j,scope)
				if (len(j)<=indsize):
					idx = int(np.rint(list(sub)))
					typ = typearr[idx]
					#print("zz",idx,typ)
					if(typ==0):
						print("disc")
						l = discNode()
						l.scope = sub
						pdf = createpdf(tempdat[:,sorted(list(sub))],len(tempdat),len(sub))
						l.create(pdf)
						p.children.append(l)
					else:
						l = leafNode()
						tempmean = submean(empmean,j)
						tempcov = submat(effcov,j)
						l.scope = sub
						l.create(tempmean,tempcov)
	 					p.children.append(l)
				else:
					p.children.append(induce(tempdat,decrease(maxsize),sub,indsize,0,maxcount,typearr))
	
	if(len(scope)<=indsize):
		idx = int(np.rint(list(sub)))
		typ = typearr[idx]
		if(typ==0):
			l = discNode()
			l.scope = scope
			pdf = createpdf(tempdat[:,sorted(list(sub))],len(tempdat),len(sub))
			l.create(pdf)
			p.children.append(l)
		else:
			l = leafNode()
			tempmean = submean(empmean,j)
			tempcov = submat(effcov,j)
			l.scope = scope
			l.create(tempmean,tempcov)
			p.children.append(l)

	return s

def decrease(value):
	if(value>4):
		return(max(4,value/4))
	else:
		return(value-3)
	#return(value-1)
		
extype = np.zeros(80)

for i in range(0,40):
	extype[i+40] = 1

nlt = np.genfromtxt('./clean.data',delimiter=",")

for i in range(0,8000):
	for j in range(0,40):
		nlt[i][j+40] += 200

s = set(xrange(80))

Tst = induce(nlt[:8000,:],20,s,1,0,3,extype)

Tst.normalize()

Tst.truncate()

for i in range(0,16000):
	t = time()
	idx = np.random.randint(0,8000)
	nd.globalarr = nlt[idx]
	Tst.passon()
	print("t1",time()-t)
	t = time()
	print(Tst.retval())
	print("t2",time()-t)
	t = time()
	Tst.update()
	print("t3",time()-t)

sum1 = 0
sum2 = 0

plot1 = np.zeros(900)

for i in range(0,900):
	arr = nlt[8000+i]
	nd.discreteconfig = arr[:40]
	nd.queryvar = -1
	nor = Tst.marginal()
	print(nor)
	rec = np.zeros(40)
	for j in range(0,40):
		nd.queryvar = j+40
		rec[j] = np.exp(Tst.marginal()-nor) - 200
		print("testing",rec[j])
	plot1[i] = np.linalg.norm(rec - arr[40:])
	sum2 += np.linalg.norm(arr[40:])
	sum1 += plot1[i]	

print(sum1/900)
print(sum2/900)
plt.plot(plot1)
plt.show()


