# Practical techniques to tune parameters of neural networks


## Architecture

### Number of Layers
Start with 1.

### Number of Hidden neurons in a layer
16, 32, 128, ...
Seldomly exeeding 1000.

### Word Embedding
128，256, ...

## Data and the Loss

### Get the Data
High quality data.

### Preprocess
1. it is essential to center the data so that its mean is zero and so that the variance of each of its dimensions is one. 
2. Sometimes, it is better to take the log(1 + x) of that dimension, when the input dimension varies by orders of magnitude. 
Basically, it’s important to find a faithful encoding of the input with zero mean and sensibly bounded dimensions. Doing so makes learning work much better. This is the case because the weights are updated by the formula: change in $$w_{ij} \propto x_idL/dy_j$$ (w denotes the weights from layer x to layer y, and L is the loss function). If the average value of the x’s is large (say, 100), then the weight updates will be very large and correlated, which makes learning bad and slow. Keeping things ZERO-MEAN and with SMALL VARIANCE simply makes everything work much better.
3. Maybe PCA and whitening

### Data augmentation
Be creative, and find ways to algorithmically increase the number of training cases that are in your disposal. If you have images, then you should translate and rotate them; if you have speech, you should combine clean speech with all types of random noise; etc. Data augmentation is an art (unless you’re dealing with images). Use common sense.

### Weight initialization
1. It is usually enough to do something like 0.02 * randn(num_params). A value at this scale tends to work surprisingly well over many different problems. Of course, smaller (or larger) values are also worth trying.
2. If it doesn’t work well (say your neural network architecture is unusual and/or very deep), then you should initialize each weight matrix with the init_scale / sqrt(layer_width) * randn. In this case init_scale should be set to 0.1 or 1, or something like that.
3. Try many different kinds of initialization. This effort will pay off. If the net doesn’t work at all (i.e., never “gets off the ground”), keep applying pressure to the random initialization. It’s the right thing to do.

### Batch Normalization
Insert BatchNorm layer immediately after fully connected layers (or convolutional layers, as we’ll soon see), and before non-linearities. Batch normalization can be interpreted as doing preprocessing at every layer of the network, but integrated into the network itself in a differentiably manner.

### Biases Initialization
1. It is possible and common to initialize the biases to be zero
2. If you are using LSTMs with very long range dependencies, you should initialize the biases of the forget gates of the LSTMs to large values. By default, the forget gates are the sigmoids of their total input, and when the weights are small, the forget gate is set to 0.5, which is adequate for some but not all problems. This is the one non-obvious caveat about the initialization of the LSTM.

### Parameter Update
The two recommended updates to use are either SGD+Nesterov Momentum or Adam


### Regularization

#### L2 regularization
1.0, seldomly exceeding 10.

#### L1 regularization

#### Max norm constraints
If you are training RNNs or LSTMs, use a HARD CONSTRAINT OVER THE NORM OF THE GRADIENT. Something like 15 or 5 works well in practice. Take your gradient, divide it by the size of the minibatch, and check if its norm exceeds 15 (or 5). If it does, then shrink it until it is 15 (or 5). Otherwise the exploding gradient can cause learning to fail and force you to use a puny learning rate like 1e-6 which is too small to be useful.

#### Dropout
The value of $ p = 0.5 $ is reasonable default, but can be tuned on validation data.
Dropout provides an easy way to improve performance. Remember to tune the dropout probability, and to not forget to turn off Dropout and to multiply the weights by (namely by 1-dropout probability) at test time. Also, be sure to train the network for longer. Unlike normal training, where the validation error often starts increasing after prolonged training, dropout nets keep getting better and better the longer you train them. So be patient.

## Learning and Evalution

### Sanity Check
Make sure the initial loss is reasonable

### Numerical Gradient Checking
If you implement your own gradients. It is easy to make a mistake when we implement a gradient, so it is absolutely critical to use numerical gradient checking. 

### Minibatches
It is vastly more efficient to train the network on minibatches of 128 examples, because doing so will result in massively greater throughput. 
It would actually be nice to use minibatches of size 1, and they would probably result in improved performance and lower overfitting; but the benefit of doing so is outweighed the massive computational gains provided by minibatches. But don’t use very large minibatches because they tend to work less well and overfit more. So the practical recommendation is: use the smaller minibatch that runs efficiently on your machine

### Gradient normalization
Divide the gradient by minibatch size.

### Learning rate schedule
1. Start with a normal-sized learning rate (LR) and reduce it towards the end: 1, 0.1, 0.01, 0.001...
2. A typical value of the LR is 0.1. Learning rates frequently tend to be smaller but rarely much larger.
3. Use a validation set to decide when to lower the learning rate and WHEN TO STOP TRAINING (e.g., when error on the validation set starts to increase). A practical suggestion for a learning rate schedule: if you see that you stopped making progress on the validation set, divide the LR by 2 (or by 5), and keep going. Eventually, the LR will become very small, at which point you will stop your training. 
4. One useful idea used by some researchers (e.g., Alex Krizhevsky) is to monitor the RATIO BETWEEN THE UPDATE NORM AND THE WEIGHT NORM. This RATIO should be at around 10-3. If it is much smaller then learning will probably be too slow, and if it is much larger then learning will be unstable and will probably fail.

### Hyperparameter Optimization

#### Random Search
#### Grid Search
#### Bayesian

### Cross Validation

### Ensembling
Train 10 neural networks and average their predictions. 
It’s a fairly trivial technique that results in easy, sizeable performance improvements. One may be mystified as to why averaging helps so much, but there is a simple reason for the effectiveness of averaging. Suppose that two classifiers have an error rate of 70%. Then, when they agree they are right. But when they disagree, one of them is often right, so now the average prediction will place much more weight on the correct answer. The effect will be especially strong whenever the network is confident when it’s right and unconfident when it’s wrong.

# References
http://yyue.blogspot.jp/2015/01/a-brief-overview-of-deep-learning.html
