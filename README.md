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

Results of the **Classic lane detection methods** can be recreated here [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AGiannoutsos/End-to-End-Lane-Detection/blob/main/classic_detector.ipynb)

Results of the **Convolutional Autoencoder methods** can be recreated here [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AGiannoutsos/End-to-End-Lane-Detection/blob/main/autoencoder_detectors.ipynb)


In both colabs it is required to clone this repository. All of the code needed is located in `./scripts`

The results of the testing in both methods, with images and videos, can be found in `./results`

The detailed report with the experiments, the methods and the references can be found here [Report](https://github.com/AGiannoutsos/End-to-End-Lane-Detection/blob/main/report.pdf)

## Dataset
The dataset used in the experiments is the [TuSimple dataset](https://github.com/TuSimple/tusimple-benchmark). TuSimple contains 6,408 road photos from US highways. The picture has a resolution of 1280x720. 
The dataset sample set is made up of 3,626 images for training, 358 for validation, and 2,782 for testing, all of which are taken in various weather conditions. 

## Training
For the training, the resolution of the images was reduced to 128x128 and for the deep learning methods, the number of channels was reduced from 3 to 1 due to hardware limitations. The training of the models was done in  Google Colab using GPU runtime. The model parameters were the same, with the learning rate set at 0.0001. The batches were of size 32 and the models were trained for 15 epochs, taking about 7 hours each. In all of the cases, Adam optimizer was applied and the Binary Cross Entropy loss was used. The code can be found at `./scripts` and the experiments together with version of the models can be found at this [WandB Project](https://wandb.ai/andreas_giannoutsos/lane_detection). For the implementation of the models, the segmentation library [Segmentation Models Pytorch](https://github.com/qubvel/segmentation_models.pytorch) was used.


# Results
## Classic lane detection methods

### No average slopes
![Classic lane detection methods no averaging](https://github.com/AGiannoutsos/End-to-End-Lane-Detection/blob/main/results/canny_edge_detector/small2_canny_grid_gk5_no_average_gif.gif)

As we can see from the gif, the results of the classic line detector without receiving the average of the line calls are not very satisfactory. We can see the lines that are formed. The many different lines end up forming 2 imaginary lines that make up the 2 lines on the road. This effect, however, has been processed as the area in which lines are detected is only the triangle in front of the image. This method does not appear to be fully accurate in recognizing different dynamic environments. The 9 images in the results show different values of the hysteresis thresholding sampling. The low? and high? thresholds are indicated by the labels beneath the picture.


### Average slopes
![Classic lane detection methods averaging](https://github.com/AGiannoutsos/End-to-End-Lane-Detection/blob/main/results/canny_edge_detector/small2_canny_grid_gk7.gif)

One way to partially reduce this problem, however, is to take the average of the slopes of the detected lines so as to create only 2 lines on the road in front of the driver. Here we can see more satisfactory results as lines are normally formed on the road at points where we would drive. However, the peculiarity of the asphalt in some places makes the forecast problematic. Also it would be very wrong to exclude some thresholds as in some that the detector is doing well in this image it is not doing well in the previous example with the increased brightness. Moreover, there is an additional problem here as in the turns at many thresholds we do not have a good result at all and that is another reason why it is very difficult to choose a price. For this reason we should move away from the logic of finding the best parameters and perhaps we should let the parameters automatically adjust the street images based on. 



## Convolutional Autoencoder methods
![Convolutional Autoencoder methods](https://github.com/AGiannoutsos/End-to-End-Lane-Detection/blob/main/results/autoencoder_models/large1_video_AEmodels_grid.gif)

As we move to the machine learning methods we have immediately much better results. The results of all 4 different models of machine learning are presented there at the same time. These are the 4 Deep Convolutional Network architectures that we have tested. The number of parameters that the model has is indicated at the end of their name. The superiority of this method from the classical one is immediately apparent as the parameters are adjusted based on all the images in the dataset for training that we have available. 

From the training curve, we can see, with the exception of the simplest model with the fewest parameters, that the other models reach a very high degree of optimization. 
![training curve](https://github.com/AGiannoutsos/End-to-End-Lane-Detection/blob/main/results/autoencoder_models/validation_BCEloss_all.png)


# Further Discussion
When we compare the results of the two approaches, we can easily infer that the techniques of Convolutional Neural Networks are superior since they reliably predict the lines in ambient dynamics with many variations and are more noise resistant. 
However, we could have made better predictions with the classical learning models if we had used a statistical approach with particle or Kalman filters.
Furthermore, these predictions are more consistent since we know exactly what happens at every point of the algorithm, as opposed to the Convolutional Neural models, which operates like a black box.
Convolutional Neural Networks continue to require a large amount of data and time to train, but as we have shown, there have been many discoveries to reduce this computational cost with smarter and more powerful architectures.
As a result, the distinction between these two approaches can vary depending on the application and its particulars. 




