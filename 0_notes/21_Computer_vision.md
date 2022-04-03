# Computer Vision with Pytorch

## Tools:
* Labelme
* Fastai2 (pytorch)
* RetineNet: one of the best networks for object detection
* https://pypi.org/project/google_images_download/ 
* TF implementation of Mask R-CNN: https://github.com/tensorflow/models/tree/master/research/object_detection
* Pytorch implementation of R-CNN: https://github.com/facebookresearch/detectron2
* DL with Keras: 

## Definitions:
* Embedding: any way to convert an image into a representation in a n-dim space.
    - embedding for audio: mel spectrogram then img2vec
    - embedding for video: concatenate n frames / use mean of all frames
* **Instance segmentation:** all entities in image are considered different (different masks)
* **semantic segmentation:** entities are grouped into **1 single mask ** (i.e. all ships in image are ships, all persons in an image are 1 single instance / mask)
* **panoptic segmentation:** a combination of both instance and semantic segmentation where only selected objects are segmented as instances. (e.g. sky is semantic while people and cars are instances)
* Cosine similarity: measures the size of the angle between two points. If small angle (closely related) then the bigger the cosine similarity.
    - bounded between -1 and 1
* perceptual hashing: overlaying similarities when hashing an image
    - useful for forensic matching, as we can see how closely related elements are
* Hamming distance: the number of positions at which the corresponding symbols are different. Bounded by length of the input.
    * **Xor function:** "there can only be 1"
        - 0 1 OR 1 0  = 1
        - 0 0 OR 1 1 = 0
* The Cold Start Problem: a system for recommended systems, can be solved by using embeddings of images and people's preferences.
* K-medoids: using a real dataset point as the centroid
    - Medoid: is a most centrally located point in the cluster

## Vision transformer
Very good for image segmentation when little labelled data.
For classification better to use other tools. 


## Transfer learning
* **Fine-tunning:** we use a pre-trained neural network, freeze convolution, randomise last layer and train for it to specifically tune to our dataset.
* **Cyclical learning rate:** a faster way of getting the best learning rate by running one epoch with different learning rates across batches to find the minimum and maximum learning rates needed, then transitioning from max to min over the epochs.
    - "one-step policy": running the first epoch to 
    - To use with transfer learning only(?)
    - for convolutional networks only 
* **Discriminative learning rate:** a transfer learning technique where you change the weights more in the last layers (more specific features from your data) of the network than the earlier layers (more generalised information).
* 


## "Fully Convolutional" Network
* Used for image segmentation or object detecction
* Bounding box: you need them for any image detection problem. The point in the e¡image where different 
* **1x1 convolution:** is a fully connected layer used to label the pixels of the image. Usually its at the end of the network.
    - "one-cycle policy": this can also be applied to batch size and other hyperparams.
    - The hypercolumns fromt he 1x1 convolution are the output of the network, we use this to label the pixels and then use it to do segmentation.
    - Identifying the layer of the bounding box is a regression problem to identify the coordinates.
* Banding-box prediction: 
    - use MSE / L1 as loss function
* In the end the network adds the L1 and coress-entropy losses together for updating weights: 
* RetinaNet: 
* U-Net
* You can add a score to each pixel in an image that allows you to maximise something about it.
* **Batch size:** is a lot smaller when using big images, as we cannot fit so much in it (you'll run out of memory)


## K-medoids
* We choose representative points (i.e. median) as centroid while on k-means we use median (an imaginary point) and KNN we check the majority of a class by neighbors to assign a new point a class.

* Siluete method for finding "optimal" number of clusters: 
https://towardsdatascience.com/silhouette-coefficient-validating-clustering-techniques-e976bb81d10c



# Obejct Detection
* single shot detectors: 
    - evaluate all possible regions
    - faster as no squeezing
    - Frameworks:
        - YOLO
        - RetinaNet
* 2-stage detectors (faster R-CNN):
    - discards many potential bounding boxes by the extra region-proposal network 
    - Frameworks:
        - Tensor Flow object detection
        - Detectron2 (https://github.com/facebookresearch/detectron2)
            - withoutbg
                * 1. Use detectrom to filter the best candidate
                * 2. You ca use a unit to fine-tune the result

* IoU: intersection over union


## Region Proposal Networks - Faster R-CNN 
Used for object detection
Process: image >> convolution layers >> features map >> proposals >> classifier (cross_entropy)
We keep the proposal that scores best in the classifier

* **Anchor box:** your aspect ratios (crop sizes) to test across image (similar to kernel in convolutions) for object detection performance.
    - **Very Important Hyperparam !!!** You need to checK dataset and make the anchor boxes as close as possible es the objects you want to recognise in the ground truth (the original labelled images)
    - You should do some data summary statistics of the shape of the anchor boxes in the labelled data (aka ground truth)
* **RoI Pulling:** Helps us to take regions of image the help us reduce prediction error before squeezing an image into a square. This is the "smart" way as we keep the most relevant information.
* 

## Precision and Recall metrics
https://en.wikipedia.org/wiki/F-score
* Precision: TP / (TP + FP)
* Recall: TP / (TP + FN)
* F-1 score for perfect precision and recall
* Fß score:
    - F-05 when ß=0.5 --> when we value precision 2x as recall
    - F-2 when ß = 2 --> we value recall 2x than precision
    - F-1: ß = 1
* Specificity: TN / (TN + FN)

* Threshold: the activation threshed you want to use for classifying as positive. Depends on use case, e.g. brain tumor treatment (maximise F1) vs brain tumor diagnostic (minimise FN, prioritize Recall) vs trial (minimise FP, innocent people going to jail)
    - The higher the threshold the more PRecision and lower recall
* ROC Curve: was develope to minimise friendly fire in war (max precision and low TN)
    - Example: for covid result we aim to maximise ROC curve

## MaP (Mean Average Precision)
For evaluation of Object Detection Models
https://blog.paperspace.com/mean-average-precision
- The mAP is the average prediction across classes. It can be misleading as it might perform really good on some classes that are not relevant for your particular problem.


## Focal Loss:
Penalises hard missclasifications by adding a modulation term in the cross entropy loss function.

## Image pyramid
Mos standa
Evaluate the image at different scale and do the down-sampling path and up-sampling path