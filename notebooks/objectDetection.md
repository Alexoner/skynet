# Object detection idea

Detection as regression.

## Traditional methods
Sliding window to region proposal.

The general idea is to make the computation graph more and more end to end: reusing neural network features

### problems & algorithms

#### Intersection over union(IoU)
A metric to describe the overlapping of regions with area.

#### non-maximum suppression
- Question: redundant, overlapping bounding boxes of same object.
Different regions proposed may overlap, all containing the object and giving a bounding box, how to merge the results?
Tune the intersection over union hyperparameter(grid search used in R-CNN). 

The algorithm is NON-MAXIMUM SUPPRESSION, a post-processing algorithm responsible for merging all detections that belong to the same object.
For predicted boxes of the same class, only take the one with largest probability and discard those with high IOU with the maximum one.

Reference:
[soft nms](https://arxiv.org/pdf/1704.04503.pdf)
[learning non-maximum suppression](https://arxiv.org/pdf/1705.02950.pdf)

#### Anchor boxes

## R-CNN
- [preprocess]supervised pre-training on large data sets for image classifier
- [preprocess]REGION PROPOSAL: selective search for ROI
- COMPUTE CNN features: warp the region to a standard square size in bounding boxes through CNN network
- CLASSIFY REGION: On the final layer of the CNN, R-CNN adds a Support Vector Machine (SVM) that simply classifies whether this is an object, and if so what object
- REGRESS BOUNDING BOXES offsets: linear regression for bounding box offsets
- [postprocess] greedy non-maximum suppression

- Dataset
	- resizing to fixed width
	- hard negative mining

- Inputs: sub-regions of the image corresponding to objects.
- Outputs: New bounding box coordinates for the object in the sub-region.


Problem:
- passes all regions through the neural network
- train three models separately, post-hoc


## Fast R-CNN
- Region of Interest Pooling: forward regions in one pass
- joint training of CNN feature extractor, object classifier, bounding box regressor in a single model with multiple heads, multi-task loss: log loss(classification) + smooth l1 loss(regression)

- Inputs: Images with region proposals obtained by selective search.
- Outputs: Object classifications of each region along with tighter bounding boxes.

## Faster R-CNN
- Insert region proposal network to predict proposals from features

Jointly train with 4 losses:
1. RPN classify object / not object
2. RPN regress box coordinates
3. Final classification score (object
classes)
4. Final box coordinates

## Detection without proposals
YOLO/SSD

## Mask R-CNN
