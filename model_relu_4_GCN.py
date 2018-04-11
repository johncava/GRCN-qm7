import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import scipy
import numpy
import torch
from torch.autograd import Variable
from scipy.io import loadmat
import torch.nn.functional as F

data = loadmat('qm7.mat')
dataset = data['X']
folds = data['P']
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
learning_rate = 1e-5

# Create random tensor weights
#W1 = Variable(torch.randn(batch, 23, 23).type(FloatTensor), requires_grad=True)
#W2 = Variable(torch.randn(batch, 23, 23).type(FloatTensor), requires_grad=True)
#W3 = Variable(torch.randn(batch, 23, 5).type(FloatTensor), requires_grad=True)
#W4 = Variable(torch.randn(23*5, 1).type(FloatTensor), requires_grad=True)

#p = p[0]
epochs = 100

#optimizer = torch.optim.Adam([W1,W2,W3,W4], lr=learning_rate)
#loss_fn = torch.nn.MSELoss(size_average=False)
master_loss_array = []
master_iteration_array = []
fold_iteration = 0
#master_iteration = 0

# Preprocess the entire dataset beforehand
X = {}
for index, item in enumerate(dataset):
	A, D = preprocess(item)
	X[index] = (A,D)

print 'Preprocessing Finished'

for fold in folds:
    #break
    fold_iteration += 1
    master_iteration = 0
    loss_array = []
    iteration_array = []

    W1 = Variable(torch.randn(batch, 23, 23).type(FloatTensor), requires_grad=True)
    W2 = Variable(torch.randn(batch, 23, 23).type(FloatTensor), requires_grad=True)
    W3 = Variable(torch.randn(batch, 23, 23).type(FloatTensor), requires_grad=True)
    W4 = Variable(torch.randn(batch, 23, 23).type(FloatTensor), requires_grad=True)
    W5 = Variable(torch.randn(batch, 23, 5).type(FloatTensor), requires_grad=True)
    W6 = Variable(torch.randn(23*5, 1).type(FloatTensor), requires_grad=True)

    optimizer = torch.optim.Adam([W1,W2,W3,W4], lr=learning_rate)
    loss_fn = torch.nn.MSELoss(size_average=False)

    for epoch in xrange(1,epochs):
        for iteration in xrange(1,len(fold)/batch):
            indices = fold[iteration*batch: iteration*batch + batch]
            a_list = []
            d_list = []
            x_list = []
            y_list = []
            for index in indices:
                a_hat, d = X[index]
                a_list.append(a_hat)
                d_list.append(d)
                x_list.append(dataset[index])
                y_list.append(label[0][index])
            #print x_list
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
            hidden_layer_3 = F.leaky_relu(D.bmm(A).bmm(D).bmm(hidden_layer_2).bmm(W3))
            hidden_layer_4 = F.leaky_relu(D.bmm(A).bmm(D).bmm(hidden_layer_3).bmm(W4))
            y_pred = F.leaky_relu(D.bmm(A).bmm(D).bmm(hidden_layer_4).bmm(W5))
            y_pred = y_pred.view(batch, 23*5)
            #y_pred = 2000*F.sigmoid(y_pred.mm(W4))
            y_pred = F.leaky_relu(y_pred.mm(W6))
            loss = loss_fn(y_pred,y.view(batch,1))# + W1.norm(2) + W2.norm(2) + W3.norm(2)
            #print torch.log(loss)
            if master_iteration % (len(fold)/batch) == 0:
                loss_array.append(torch.log(loss).data.numpy().tolist())
                iteration_array.append(master_iteration)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            master_iteration += 1
    master_loss_array.append(loss_array)
    master_iteration_array.append(iteration_array)
    #if fold_iteration >= 4:
    #	break
    print 'Fold ', fold_iteration, ' Done'
print "done"
print loss_array

a = master_loss_array
b = master_iteration_array
plt.plot(b[0],a[0], b[1],a[1], b[2], a[2], b[3], a[3], b[4], a[4])
plt.legend(['Fold1', 'Fold2', 'Fold3', 'Fold4', 'Fold5'], loc='upper left')
plt.xlabel('Iterations')
plt.ylabel('Log Error')
plt.title('GCN 4 Layer with leaky_relu (lr=1e-5)')
plt.show()
plt.savefig('results_relu_4layer_GCN_lr=1e-5.png')
