class NeuralNetwork:
    
    def __init__(self,X,Y,h0, h1, h2,W1,W2, b1, b2):
        self.X = X
        self.Y = Y
        self.h0 = 2
        self.h1 = 10
        self.h2 = 1
        self.W1 = W1
        self.W2 = W2
        self.b1 = b1
        self.b2 = b2
    
    
    
    def sigmoid(self,z):
        self.z = None

        # Your code here
        sigma_z = 1 / (1 + np.exp(-z))

        return sigma_z

    def d_sigmoid(self):
        d_sigma_z = self.sigmoid(z)*(1 - self.sigmoid(z))

        return d_sigma_z

    def loss(self,y_pred, Y):
        y_pred = self.sigmoid(z)
        return  np.divide(-(np.sum(Y * np.log(y_pred) + (1- Y) * np.log((1-y_pred)))),Y.shape[1])
    
        

    def init_params(self):
      h0, h1, h2 = 2, 10, 1

      # Your code here
      #init_range = 0.1
      #Weights
      self.W1 = np.random.randn(h1,h0)
      self.W2 = np.random.randn(h2,h1)

        #Bias
      self.b1 = np.zeros((h1,1))
      self.b2 = np.zeros((h2,1))
      return self.W1, self.W2, self.b1, self.b2
    
    def forward_pass(self,Z1,A1,Z2,A2):
        self.Z1 = Z1
        self.A1 = A1
        self.Z2 = Z2
        self.A2 = A2
        
        self.Z1 = self.W1.dot(X) + self.b1
        self.A1 = self.sigmoid(Z1)
        self.Z2 = self.W2.dot(A1) + self.b2
        self.A2 = self.sigmoid(Z2)
        return A2, Z2, A1, Z1 
    
    def backward_pass(self):
        
        
        #L = loss(A2,Y)
        # Your code here
        dL_dA2 = (A2 - Y)/(A2 * (1-A2))
        dA2_dZ2 = self.d_sigmoid(Z2)
        dZ2_dW2 = A1.T

        dW2 = (dL_dA2 * dA2_dZ2) @ dZ2_dW2
        db2 = dL_dA2 @ dA2_dZ2.T

        dZ2_dA1 = W2
        dA1_dZ1 = self.d_sigmoid(Z1)
        dZ1_dW1 = X.T

        dW1 = (dZ2_dA1.T * (dL_dA2 * dA2_dZ2)* dA1_dZ1) @ dZ1_dW1
        db1 = ((dL_dA2 * dA2_dZ2)@(dZ2_dA1.T *dA1_dZ1).T).T

        return dW1, dW2, db1, db2


    def accuracy(self,y_pred, y):
        
        
        # Your code here
        pred = (y_pred >=0.5).astype(int)
        acc = np.sum(y==pred)/y.shape[1]
        return acc

    def predict(self):
      

        # Your code here
        #X = X_train
        A2, Z2, A1, Z1 = self.forward_pass(X, W1,W2, b1, b2)

        return A2


    def update(W1, W2, b1, b2,dW1, dW2, db1, db2, alpha ):

        # Your code here
        W1 = W1 - alpha * dW1
        W2 = W2 - alpha * dW2

        b1 = b1 - alpha * db1
        b2 = b2 - alpha * db2


        return W1, W2, b1, b2
    
        #plot decision boundary
    def plot_decision_boundary(self,W1, W2, b1, b2):
      x = np.linspace(-0.5, 2.5,100 )
      y = np.linspace(-0.5, 2.5,100 )
      xv , yv = np.meshgrid(x,y)
      xv.shape , yv.shape
      X_ = np.stack([xv,yv],axis = 0)
      X_ = X_.reshape(2,-1)
      A2, Z2, A1, Z1 = self.forward_pass(X_, W1, W2, b1, b2)
      plt.figure()
      plt.scatter(X_[0,:], X_[1,:], c= A2)
      plt.show()



   #Training loop
    def fit(self, X, y):
      
      alpha = 0.001
      W1, W2, b1, b2 = self.init_params()
      n_epochs = 10000
      train_loss = []
      test_loss = []
      for i in range(n_epochs):
        
        ## forward pass
        A2, Z2, A1, Z1 = self.forward_pass(X_train,W1, W2, b1, b2)
        ## backward pass
        dW1, dW2, db1, db2 =self.backward_pass(X_train,Y_train, A2, Z2, A1, Z1, W1, W2, b1, b2)
        ## update parameters
        W1, W2, b1, b2 = self.update(W1, W2, b1, b2,dW1, dW2, db1, db2, alpha )

        ## save the train loss
        train_loss.append(loss(A2, Y_train))
        ## compute test loss
        A2, Z2, A1, Z1 = self.forward_pass(X_test, W1, W2, b1, b2)
        test_loss.append(loss(A2, Y_test))

        ## plot boundary
        if i %2000 == 0:
          self.plot_decision_boundary(W1, W2, b1, b2)

      ## plot train et test losses
      plt.plot(train_loss)
      plt.plot(test_loss)

      y_pred = self.predict(X_train, W1, W2, b1, b2)
      train_accuracy = self.accuracy(y_pred, Y_train)
      print ("train accuracy :", train_accuracy)

      y_pred = self.predict(X_test, W1, W2, b1, b2)
      test_accuracy = self.accuracy(y_pred, Y_test)
      print ("test accuracy :", test_accuracy)
