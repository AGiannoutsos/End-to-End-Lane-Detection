# End-to-End-Lane-Detection
End to End Lane Detection for autonomous driving by using various Autoencoder models with classic segmentation filters and transforms

# Abstract
  Auto lane keeping is one of the many driver assistance technologies that are becoming more common in modern vehicles.
The latter enables the car to correctly place itself inside the road lanes, which is also critical for any subsequent lane deviation or trajectory planning decision in fully autonomous vehicles. 
Classic lane detection methods rely on a variety of highly advanced, hand-crafted features and heuristics, which may be efficient computationally, however they are vulnerable to scalability due to the vast changes and differences in the road scenes.
More recently, with the advances in machine learning and especially with the Convolutional NNs,  the hand-crafted feature detectors have been replaced with deep networks to learn predictions on pixel-wise lane segmentations.

  In this repository, we are going to solve the lane detection problem with different methods. To be more specific, we feed a dataset of highway lane images to both methods in order to compare them.
First, we try the traditional edge-detection method with hand-crafted features, and then we look at different Deep Convolutional Network architectures to address this problem.
Finally, we compare these methods by using images from their output.
Following an analysis of the model outputs, we evaluate the ability of the algorithms to detect the lanes on the road effectively. 

# Experiments
