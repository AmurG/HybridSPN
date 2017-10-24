# HybridSPN
Hybrid SPN for continuous and discrete variables

Running and guide :

The disc.py file works exactly the same as the normal, discrete TreeSPN.

extype is a flag vector of size D. The i-th entry in extype is 0 if it is a discrete variable, else it is 1

Currently running disc.py will output ( in the end, and after training on the first 8K samples of clean.dat )

1) The average norm deviation of reconstruction vs true on next 900 samples
2) The average norm of the next 900 samples

The ratio of 1 and 2 is about 0.5, adding EM iters and tweaking hyperparams can improve this.

There is also a matplotlib graph as output that shows the norm deviation by index.
