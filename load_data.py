import scipy
import numpy
import torch
from torch.autograd import Variable
from scipy.io import loadmat
import torch.nn.functional as F

data = loadmat('qm7.mat')
X = data['X']
p = data['P']
label = data['T']

batch = 100
# Preprocess
def preprocess(x):
	x = x.tolist()
	A = numpy.zeros((23,23))
	# Create Adjacency Matrix
	for row in xrange(len(x)):
		for col in xrange(len(x[row])):
			if x[row][col] > 0.0:
				A[row][col] = 1
	# Create Degree Matrix
	A_hat = A + numpy.eye(23)
	degree = []
	for row in A_hat:
		degree.append(row.sum())
        D = numpy.diag(degree)
	D = numpy.linalg.cholesky(D)
	D = numpy.linalg.inv(D) 	
	return A_hat.tolist(), D.tolist()

#print x[2010]
FloatTensor = torch.FloatTensor
learning_rate = 1e-2

# Create random tensor weights
W1 = Variable(torch.randn(batch, 23, 23).type(FloatTensor), requires_grad=True)
W2 = Variable(torch.randn(batch, 23, 23).type(FloatTensor), requires_grad=True)
W3 = Variable(torch.randn(batch, 23, 5).type(FloatTensor), requires_grad=True)
W4 = Variable(torch.randn(23*5, 1).type(FloatTensor), requires_grad=True)

p = p[0]
epochs = 50

optimizer = torch.optim.Adam([W1,W2,W3,W4], lr=learning_rate)
loss_fn = torch.nn.MSELoss(size_average=False)
for epoch in xrange(epochs):
	for iteration in xrange(len(p)/batch):
		indices = p[iteration*batch: iteration*batch + batch]
		a_list = []
		d_list = []
		x_list = []
		y_list = []
		for index in indices:
			a_hat, d = preprocess(X[index])
			a_list.append(a_hat)
			d_list.append(d)
			x_list.append(X[index])
			y_list.append(label[0][index])
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

		y_list = numpy.array(y_list)
		y_list = torch.from_numpy(y_list)
		y_list = Variable(y_list, requires_grad = False)
		y = y_list.float()

		A = a_list.view(batch, 23, 23)
		D = d_list.view(batch, 23, 23)
		x = x_list.view(batch, 23, 23)

		hidden_layer_1 = F.leaky_relu(D.bmm(A).bmm(D).bmm(x).bmm(W1))
		hidden_layer_2 = F.leaky_relu(D.bmm(A).bmm(D).bmm(hidden_layer_1).bmm(W2))
		y_pred = F.leaky_relu(D.bmm(A).bmm(D).bmm(hidden_layer_2).bmm(W3))
		y_pred = y_pred.view(batch, 23*5)
		#y_pred = 2000*F.sigmoid(y_pred.mm(W4))
		y_pred = F.leaky_relu(y_pred.mm(W4))
		loss = loss_fn(y_pred,y.view(batch,1)) + W1.norm(2) + W2.norm(2) + W3.norm(2)
		print torch.log(loss)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
print "done"
