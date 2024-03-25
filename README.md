# $${\color{orange}Neural \space \color{orange}Network  \space \color{orange}from \space \color{orange}Scratch.}$$

## ${\color{lightblue}Introduction: \space \color{lightblue}Neural  \space \color{lightblue}Network}$

A neural network is a machine learning model that operates in a manner similar to the human brain, by mimicking the way biological neurons work together to process information.

A neural network consists of:
- An input layer,
- one or more hidden layers, and
- an output.
  
The nodes connect to each other and have their own weights and threshold. If the output of any individual node is above th ee stated threshold value, that nodewill be activated, sending to the next layer of the network. Otherwise, no data is passed to the next layer.

In this project, our neural network has two input values, one hidden layer with ten neurons and one output layer.

We are workin on a binary classification problem and  therefore we will use the *sigmoid* activation function. An *activation function* introduces non-linearity to the neural network thereby helping the neural network to handle complex data and generalize easily.

## ${\color{lightblue}Building \space \color{lightblue}the  \space \color{lightblue}neural \space \color{lightblue}network \space \color{lightblue}from  \space \color{lightblue}scratch.}$

We start by defining a class NeuralNetwork, after that we create some functions within this class.
- The sigmoid function.
- derivative of the sigmoid function.
- the cross entropy loss function
- parameter initialization function, where we initialize our parameters, that is, the weight and bias.
- forward pass, that is, where the input data is fed through the network to generate predictions.
- backward pass, here we calculate the gradient of neural parameters. We move in reverse order, from the output to the input to the input layer, following the chain rule in differential calculus.
- accuracy function to determine the accuracy rate of our model.
- predict function
- update function to update the weights as we perform backpropagation.
- plot function
- fit function to fit our model to the our data set.

## ${\color{lightblue}User\space \color{lightblue}Instructions}$

We can find two .py files in this repository: neuralNet.py and main_nn.py

### 1. neuralNet.py

It has all the fuctions stated above in  NeuralNetwork class.

### 2. main_nn.py

This is where we call the NeuralNetwok class that is found in the neuralNet.py. 
- We import the necessary python libary.
- We generate the data set we would like to train our model on or you may use your own data set / modify it.
- Finally, we call the NeuralNetwork function from the neuralNet.py and we will obtain our results.
