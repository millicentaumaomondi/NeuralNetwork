class NeuralNetwork:

    def __init__(self,h0, h1, h2,alpha,epochs):
        self.h0 = h0
        self.h1 = h1
        self.h2 = h2
        self.alpha = alpha
        self.epochs = epochs
        self.W1,self.W2, self.b1, self.b2 = self.init_params()



    def sigmoid(self,z):
        #self.z = None

        # Your code here
        sigma_z = 1 / (1 + np.exp(-z))

        return sigma_z

    def d_sigmoid(self,z):
        d_sigma_z = self.sigmoid(z)*(1 - self.sigmoid(z))

        return d_sigma_z

    def loss(self,y_pred, Y):

        return  np.divide(-(np.sum(Y * np.log(y_pred) + (1- Y) * np.log((1-y_pred)))),Y.shape[1])



    def init_params(self):
      
      self.W1 = np.random.randn(self.h1,self.h0)
      self.W2 = np.random.randn(self.h2,self.h1)

        #Bias
      self.b1 = np.zeros((self.h1,1))
      self.b2 = np.zeros((self.h2,1))

      return W1, W2, b1, b2

    def forward_pass(self,X):
        W1, W2, b1, b2 = self.init_params()


        Z1 = self.W1.dot(X) + self.b1
        A1 = self.sigmoid(Z1)
        Z2 = self.W2.dot(A1) + self.b2
        A2 = self.sigmoid(Z2)
        return A2, Z2, A1, Z1

    def backward_pass(self,X,Y):
        A2, Z2, A1, Z1 = self.forward_pass(X)

        #L = loss(A2,Y)
        # Your code here
        dL_dA2 = (A2 - Y)/(A2 * (1-A2))
        dA2_dZ2 = self.d_sigmoid(Z2)
        dZ2_dW2 = A1.T

        dW2 = (dL_dA2 * dA2_dZ2) @ dZ2_dW2
        db2 = dL_dA2 @ dA2_dZ2.T

        dZ2_dA1 = self.W2
        dA1_dZ1 = self.d_sigmoid(Z1)
        dZ1_dW1 = X.T

        dW1 = (dZ2_dA1.T * (dL_dA2 * dA2_dZ2)* dA1_dZ1) @ dZ1_dW1
        db1 = ((dL_dA2 * dA2_dZ2)@(dZ2_dA1.T *dA1_dZ1).T).T

        return dW1, dW2, db1, db2


    def accuracy(self,X, Y):

        y_pred=self.predict(X)

        
        # Your code here
        pred = (y_pred >=0.5).astype(int)
        acc = np.sum(Y==pred)/Y.shape[1]
        return acc

    def predict(self,X):


        # Your code here
        #X = X_train
        A2, Z2, A1, Z1 = self.forward_pass(X)

        return A2


    def update(self,dW1, dW2, db1, db2):

        # Your code here
        self.W1  -= self.alpha * dW1
        self.W2  -= self.alpha * dW2

        self.b1 -= self.alpha * db1
        self.b2 -= self.alpha * db2


        return W1, W2, b1, b2

    #plot decision boundary
    def plot_decision_boundary(self):
        x = np.linspace(-0.5, 2.5,100 )
        y = np.linspace(-0.5, 2.5,100 )
        xv , yv = np.meshgrid(x,y)
        xv.shape , yv.shape
        X_ = np.stack([xv,yv],axis = 0)
        X_ = X_.reshape(2,-1)
        A2, Z2, A1, Z1 = self.forward_pass(X_)
        plt.figure()
        plt.scatter(X_[0,:], X_[1,:], c= A2)
        plt.show()



   #Training loop
    def fit(self, X_train, y_train,X_test,y_test):
     
        train_loss = []
        test_loss = []
        for i in range(n_epochs):

          ## forward pass
          A2, Z2, A1, Z1 = self.forward_pass(X_train)
          ## backward pass
          dW1, dW2, db1, db2 =self.backward_pass(X_train,Y_train)
          ## update parameters
          self.update(dW1, dW2, db1, db2)

          ## save the train loss
          train_loss.append(loss(A2, Y_train))
          ## compute test loss
          A2, Z2, A1, Z1 = self.forward_pass(X_test)
          test_loss.append(loss(A2, Y_test))

          ## plot boundary
          if i %2000 == 0:
            self.plot_decision_boundary()

      ## plot train et test losses
    plt.plot(train_loss)
    plt.plot(test_loss)

    # train_accuracy = self.accuracy(X_train, Y_train)
    # print ("train accuracy :", train_accuracy)

    # y_pred = self.predict(X_test, W1, W2, b1, b2)
    # test_accuracy = self.accuracy(y_pred, Y_test)
    # print ("test accuracy :", test_accuracy)




