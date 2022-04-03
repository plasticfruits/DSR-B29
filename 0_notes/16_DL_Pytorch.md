# DeepLearning with Pytorch
Reference: https://github.com/fastai/fastbook

* Deep Learning is about doing feature extraction to minimize the loss function and convert this to vectors in a n-dimensional Euclidean space so that we can group input (i.e. images, words, sounds...) together in that space.

Deep learning is good at learning interpretations, performs good at solving a classification or regression task.


## Libraries
- Optuna: A hyperparameter optimization framework
- 



## Jargon

* **Neuron (AKA node):** the processing unit in a deep neural network
* ***Parameter*** = ***weight*** = ***connection strength***
* **Connection strength:**
    - parameter: the weights in the network (everything adjusted via back-propagation)
    - Hyperparameter: everything else that is a decision on the design/architecture of the network.
* **Entropy:** is a measure of surprise (VERY IMP!!!)
    - The more balanced the dataset the higher the entropy
    - the negative sum of probability of an observation times its log (base 2 in information theory)
        - Entropy og AAAAAB is  `-(math.log(5/6)) * 5/6 + math.log(1/6) * 1/6)`
    - We are minimizing the Kullback-Leibler divergence
    - https://github.com/fastai/fastbook/blob/master/04_mnist_basics.ipynb

* **Transfer learning:** when using an already-trained model in a new context (take the architecture and the weights of the model)
* **Capacity:** how wiggely the line of your decision line is (the more non-linearity the more sophisticated your decision boundaries)
* **Model decay:** models perform worse over time de to unpredictable / unanticipated changes in environments
* **Learning rate decay:** decreasing the learning rate progressively (the more epochs the less learning rate)
* 
* **Weight decay:** L1 (Lasso) or L2 (Ridge) regularization
* **A Dense layer:** a layer in which every neuron is fully connected (all input neurons to all output neurons)
* **A batch:** the size of the sample we take to update the weights
* **Stochastic gradient descent:** we compute the gradient of a batch, and batches are shuffled (are stochastic).
* **The vanishing gradient problem:** when you perform power operation of numbers that are too small they tend to disappear, hence we lose information. This is due to the large amounts of bits needed to represent value.
    - If you can bypass the problem of vanishing gradient, you can have as many layers as you want.
* **Latency:** the time changes in an operation to be completed
* **Dropout:** penalises strong conections so that other neurons are forced to learn. Dropout only used for trining, not for testing!
* **Regularization:** any trick that reduces overfitting :)
* **Active learning:** using a model to identify miss-labelled data
    - Miss-labelled data will score high cross-entropy
* **Patience:** the number of epochs we wait to see if model improves.


<br>

## Evaluation
* **Loss function:** a measure of the prediction error.
    - has to be differentiable with respect to the weights (of back propagation)
    - Most common:
        - mean_squared_error
        - mean_average_error
* **When to stop training?** When model overfits, that is, when training loss is going down while validation loss starts to increase. 
* A well train model will always perform better on training set than validation set* If validation loss is smaller than training loss we are under-fitting the model --> keep training!
* The `Negative log-likelihood` should assign low value if model has high confidence in correct class
* **Cross-entropy:**
    - **Low** if if you are very confident on a correct prediction
    - **High** if very confident on wrong prediction
* **Cyclical learning rate:** while training calculates between high-learning rate and low-learning rate and adjusts.
    - It saves a lot of time on experimenting with learning rate value.
* **Performance metrics:** a measure to know if we are solving the problem
    - ***Accuracy:*** hits/trials --> (TP+TN)/(all predictions)
    - ***Recall*** = TP / (TP+FN)
    - ***Precision*** = TP / (TP + FP)
    - ***F1*** = 2P*R / (P+R) --> harmonious mean between Recall and Precision
* Call`model.eval()` before calling the validation sample to drop all transformations


## Convolutional Networks
* **Convolution kernels:** they must but uneven dimension (3x3, 5x5, ...) so that there is always a center
* **A convolution layer:** a way to extract data to minimise loss function using a matrix of weights applied
    - Also referred as "cross-correlation"
    - Smart because you abstract input before applying a fully-connected layer which is much lighter and better performing.
* Pooling: a down sampling technique to reduce the size of the images. Generally not good but in some cases increases performance.
    - Max Pooling: takes maximum activation
    - Average Pooling: takes the average of the window matrix
* Freeze layer: when the weights are not changed during training of a model
* Transfer learning: you freeze the feature learning layers (with kernels) and only update the dense layers with backprop.
    - The deeper we go into the network te more sophisticated the features become, hence the more dependent on the dataset/specific use-case, hence this weights are more likely to need updating in comparison to the earlier layers (more generic)


## Other
* **Active learning:** using your network for cleaning mislabelled data.
* **batch size:** if you run into `cuda out of memory` error then try reducing batch size by 1/2.
* Why Relu AF became so popular: because its easier to compute and you don't loose information on the gradient (***the vanishing gradient problem***)
* **Model Ensemble:** we train model in different GPU's and at the end we coalesce all models into one by taking the mean of the weights (% by num of models)
* **The problem of induction:** science is based on a fallacy that the feature will resemble the past (David Hume)
* **The no-free-lunch theorem:** there is no guarantee that what you have learned will apply on new datasets
* **Exponential moving average:** A way to smooth curves by assigning an order to the point you are valuating, giving a lower order to those further away.
    - Used by Adam optimizer (Stochastic GD with momentum)
* K-means and many other ML models are limited to convex boundaries.
* You need at least a **2-layer network** to learn non-linear correlations

## Interview Questions
- What can one do with one layer?
- Fast Fourier transform