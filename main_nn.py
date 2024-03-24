from neuralNet import NeuralNetwork

import numpy as np
import matplotlib.pyplot as plt

# generate data
var = 0.2
n = 800
class_0_a = var * np.random.randn(n//4,2)
class_0_b =var * np.random.randn(n//4,2) + (2,2)

class_1_a = var* np.random.randn(n//4,2) + (0,2)
class_1_b = var * np.random.randn(n//4,2) +  (2,0)

X = np.concatenate([class_0_a, class_0_b,class_1_a,class_1_b], axis =0)
Y = np.concatenate([np.zeros((n//2,1)), np.ones((n//2,1))])
X.shape, Y.shape

# shuffle the data
rand_perm = np.random.permutation(n)

X = X[rand_perm, :]
Y = Y[rand_perm, :]

# train test split
ratio = 0.8
X_train = X [:, :int (n*ratio)]
Y_train = Y [:, :int (n*ratio)]

X_test = X [:, int (n*ratio):]
Y_test = Y [:, int (n*ratio):]

plt.scatter(X_train[0,:], X_train[1,:], c=Y_train[0,:])
plt.show()

def plot_decision_boundary(W1, W2, b1, b2):
  x = np.linspace(-0.5, 2.5,100 )
  y = np.linspace(-0.5, 2.5,100 )
  xv , yv = np.meshgrid(x,y)
  xv.shape , yv.shape
  X_ = np.stack([xv,yv],axis = 0)
  X_ = X_.reshape(2,-1)
  A2, Z2, A1, Z1 = forward_pass(X_, W1, W2, b1, b2)
  plt.figure()
  plt.scatter(X_[0,:], X_[1,:], c= A2)
  plt.show()

  alpha = 0.001
W1, W2, b1, b2 = init_params()
n_epochs = 10000
train_loss = []
test_loss = []
for i in range(n_epochs):
  ## forward pass
  A2, Z2, A1, Z1 = forward_pass(X_train,W1, W2, b1, b2)
  ## backward pass
  dW1, dW2, db1, db2 = backward_pass(X_train,Y_train, A2, Z2, A1, Z1, W1, W2, b1, b2)
  ## update parameters
  W1, W2, b1, b2 = update(W1, W2, b1, b2,dW1, dW2, db1, db2, alpha )

  ## save the train loss
  train_loss.append(loss(A2, Y_train))
  ## compute test loss
  A2, Z2, A1, Z1 = forward_pass(X_test, W1, W2, b1, b2)
  test_loss.append(loss(A2, Y_test))

  ## plot boundary
  if i %2000 == 0:
    plot_decision_boundary(W1, W2, b1, b2)

## plot train et test losses
plt.plot(train_loss)
plt.plot(test_loss)

y_pred = predict(X_train, W1, W2, b1, b2)
train_accuracy = accuracy(y_pred, Y_train)
print ("train accuracy :", train_accuracy)

y_pred = predict(X_test, W1, W2, b1, b2)
test_accuracy = accuracy(y_pred, Y_test)
print ("test accuracy :", test_accuracy)

neuralNet = NeuralNetwork()

if __name__ == "__main__":
  neuralNet
 
   