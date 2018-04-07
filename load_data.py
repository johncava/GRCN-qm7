import scipy
import numpy
import torch
from torch.autograd import Variable
from scipy.io import loadmat
import torch.nn.functional as F

data = loadmat('qm7.mat')
#print data.keys#()
#print data['P']#, len(data['P'])
x = data['X']
p = data['P']
print x
print p

print data['X'][564]

batch = 5

# Preprocess
def preprocess(x):
	x = x.tolist()
	#print x
	#A = [[0]*23 for i in range(23)]
	A = numpy.zeros((23,23))
	# Create Adjacency Matrix
	for row in xrange(len(x)):
		for col in xrange(len(x[row])):
			if x[row][col] > 0.0:
				A[row][col] = 1
	print A
	# Create Degree Matrix
	degree = []
	for row in A:
		degree.append(row.sum() + 1)
	D = numpy.diag(degree)
	print D
	D = numpy.linalg.cholesky(D)
	D = numpy.linalg.inv(D)
	A_hat = A + numpy.eye(23)
	
	return A_hat.tolist(), D.tolist()
FloatTensor = torch.FloatTensor

# Create random tensor weights
W1 = Variable(torch.randn(batch, 23, 23).type(FloatTensor), requires_grad=True)
W2 = Variable(torch.randn(batch, 23, 23).type(FloatTensor), requires_grad=True)
W3 = Variable(torch.randn(batch, 23, 5).type(FloatTensor), requires_grad=True)
W4 = Variable(torch.randn(23*5, 1).type(FloatTensor), requires_grad=True)

p = p[0]

for iteration in xrange(len(p)/batch):
	indices = p[iteration*batch: iteration*batch + batch]
	a_list = []
	d_list = []
	x_list = []
	for index in indices:
		a_hat, d = preprocess(x[index])
		a_list.append(a_hat)
		d_list.append(d)
		x_list.append(x[index])
	a_list = numpy.array(a_list)
	a_list = torch.from_numpy(a_list)
	a_list = Variable(a_list, requires_grad = False)
	a_list = a_list.float()

        d_list = numpy.array(d_list)
        d_list = torch.from_numpy(d_list)
        d_list = Variable(d_list, requires_grad = False)
        d_list = d_list.float()

        x_list = numpy.array(x_list)
        x_list = torch.from_numpy(x_list)
        x_list = Variable(x_list, requires_grad = False)
        x_list = x_list.float()

	A = a_list.view(batch, 23, 23)
	D = d_list.view(batch, 23, 23)
	x = x_list.view(batch, 23, 23)

	hidden_layer_1 = F.relu(D.bmm(A).bmm(D).bmm(x).bmm(W1))
	hidden_layer_2 = F.relu(D.bmm(A).bmm(D).bmm(hidden_layer_1).bmm(W2))
	y_pred = F.relu(D.bmm(A).bmm(D).bmm(hidden_layer_2).bmm(W3))
	y_pred = y_pred.view(batch, 23*5)
	y_pred = y_pred.mm(W4)
	print y_pred.sum()
	break
print "done"
