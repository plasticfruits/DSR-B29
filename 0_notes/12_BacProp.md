# Back Propagation

## 
* Learning: the acquisition of knowledge or kills through ...
* Supervised learning: learning from labelled data
* Perceptron: the first design of a neural network (1958)

* learning rathe (alpha): 
* termination of training:
    - when all samples correctly labeled
    - after a pre-set number of epochs where the & of misclassified labels remains stable
* epoch: a full cycle over the sample data
* Activation function: 
    - Sigmoid function: mainly used for teaching
    - ReLU: more common
    - Soft: more common
* hidden layers: layers between input and output nodes
    - How to decide how many hidden layers?
    - HOw to decide how many nodes per layer?
* Arg & Soft Max: extra layer from output of NL that helps us understand results better.
    - Arg MAx: sets largest value to 1 and all other to 0
        - problem: cannot be used for back propagation, that is, to optimise W and beta's in NL
        - commonly used after training
    - Soft MAx:
        often used for training and back propagation

## Optimization problem
Solved with gradient descent
* the good enough principle: finding a low enough value to be useful
* loss funciton: 
    - classification: categorical cross entropy or sigmoid for binary class
    - Regression: MSE
* activation function