# NLP

Tensorflow and Pytorch are the same (as of today), but Pytorch is slightly faster
Both perform stochastic gradient descend on your data
Keras: a module inside TF that allows easier expression (higher level approach to deep learning)
10.000 samples per class for training Neural Network

### **Quality of model depends on quantity and quality of data**
**The No free lunch theorem**: if you dont train your model on enough data you will never improve it
* Loss: if too big, very likely that you need better and/or more quality data

- Avoid under-fitting: make neural network bigger
- Avoid over-fitting: make neural network smaller?
    - Data augmentation:
        * images: cloning with edits (eg. flipping cat image)
        * text: translate to a diff language and back to english - diff. sentences, same semantics



TF uses floating points number with 32bits, we have to map the numbers to those!

`model.add(layers.Flatten(input_shape=(28, 28)))` flattens matrices into a vector
 

* Regression problem: you predict a continuos value
* Classification problem: finite amount of discrete output
    - Binary: two classes
    - Categorical: more than 2 classes
        * Always use `softmax` for activation function
        * Always use `categorical_crossentropy` for loss function and categorical classifier
    - sample size recommendation: to be 10.000 samples per class

* split rate: 7:1:2 (70% train, 10% validate, 20% test)


 * Loss function: distance (i.e. error) between prediction and actual result of model
 Examples of loss functions:
 - MSE (mean squared error) - 
 - MAE (mean absolute error) - 

Notes:
    - More layer are better for dealing with complex data, but if sample is small it can lead to over-fitting.
    - How to find number of layers / params? --> **experiment!** --> trial / error
    - Finding best Neural NEtwork architechture is a matter of "creative" experimentation!
    - .˙. Keep track of experiments!

Deep Neural Network: 
- each hidden layer should extract information
- we multiply each layer by an activation function to make it non-linear
- Tip implement a logistic regression by hand 1 time to get a good grasp :)


## What is NLP?
Only written text 
Dataset goal: **High quality high quantity**

Data Prep:
* observations should have the same length
    - We can pad empty characters until it matches the mean and cut those longer
    - Try out different models with different cutting points
* data should be balance, for this we can use:
    - Over / under-sampling
    - Adding different weights
    - slightly modifying some observations (more common with images)
    - ... [other ways exist]

Batch size limits:
- gradient descent too smooth
- run out of memory
Usually start with small/big size and adjust until you run out of memory, then reduce. 

* Techniques:
    - lemmatizing:
        - not relevant any more with transformers architecture (since 2017)
        - Paper: Attention is all you need
    - all small caps
* Corpus: the amount of data that you have
* word embedding:
* lstm: only looks at recent words
* transformer: long context .˙. does not lose data
* vocabulary size: list of unique words starting with to most common to less common
    - its a hyper-parameter
* shuffling: you always have to shuffle batches to avoid overfitting on order of dataset
    - for big datasets is important to use a shuffle buffer to reduce computation
* Caching: always good practice in TensorFlow to save memory


NL with mulitple inputs:
- model subclasing
- functional API



## Types of encoding

### Bag of words encoding
This is outdated, we use as quick baseline do to easy implementation.  
Use-case: small data and for baseline
(outdated)

Problems:  
- Leads to highly sparse matrix
- high loss of information 
  - no word count
  - no order of words
  - not using values between 0:1

### Word Embeddings Encoding

* Mapping semantic similarities to geometric distances
* Words that are close to each other in meaning are close to each other in vector space
* Used by all state-of-the-art language models
* Note: anything that has a finite amount can be tokenized into an embedding layer (not just words)
* Encoder hyperparameters:
    - vocabulary_size
    - sequence_length
* Embedding layer is a look-up table (initially randomized)
* **Correlation in vectors correspond to correlation in context**


## Activation Functions
* hidden layers: `relu` most common activation function used
* output layer: 
    - Binary classifier: always `sigmoid` activation function in last layer
    - Categorical classifier: always `softmax` activation function in last layer
* tanh: replace by relu as it was too computationally expensive


## optimizer
* stochasticGD is the one with most control, but you need to add the `learning_rate`
* adam also good, learning rate adjusted automatically
* we can add a Dropout layer
    - It randomly sets parts of the output of a layer to 0
    - max seen 0.7 before output layer

## Loss functions
* `binary_crossentropy` for binary classifier
* `categorical_crossentropy` for multi classifier 

## GENSIM
  

## SimpleRNN
**DO NOT USE !!** --> Very outdated and not accurate

## LSTM
Deep Neural network architecture 
* Embedding layer only needed for NLP data
(used in word suggestion for phone messages)
- good if dataset small but not state-of-the-art
- also for time-series
- High computation and long-term training (due to time memory)
- Architecture:
    - 1 layer usually good enough
    - 2 layers could help, double the nodes and set `return_sequence=True`
    - 3 layers: "absolute maximum"
- Bi-directional LSTM: concatenate two LSTM's in one layer, one reading from left-right the otheronw starting from the end. 
    - using `layers.Bidirectional(layers.LSTM(64))`

## GRU (Gated Rectified Unit)
Faster (less trainable params) and almost as good as LSTM



## TRANSFORMERS --> Way to go!
- See "Attention is all you need" paper
- dominates NLP and computer vision
(will probably dominate everything)
- state-of-the-art have 96 identical layers stack on top of eachother
- Hyperparams:
    * number of encoders
    * embedding size
    * number of heads (the most important parameter!)
- Encoder layer by itself very good as tokenizer
    * Can be trained without labels first then add fully connected layer for classification
- Decoder layer by itself is a great language generator (like LSTM)
- Both combined (encoder + decoder): sequence-to-sequence models for translation, text summarization, etc.
- TransformerEncoder, duplicated up to 96 times (layers) in high-end models
- `num_heads` the higher the better

Problems:
    - Multihead Attention is quadratic, so slow to compute!


## SELF-SUPERVISED LEARNING
When you train your model to your data itself.
You do this by changing the output to a signmoid activation with 1 node
--> if i understand correctly, you train on your unlabbeled data and classify it with sigmoid 0/1 to rate the quality of it



## layers
* Convolution: stay the same or increase
* Dense: stay the same or decrease