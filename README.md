# Hurricane-Harvey-Challenge--Satellite-image-segmentation-by-Deep-Learning

Hurricane Harvey Challenge 

Final Project for Foundations of Deep Learning 

M.Sc. in Data Sciences and Business Analytics 

CentraleSupélec 

## PROBLEM DEFINITION 

Hurricane Harvey was a destructive Category 4 storm that hit Texas and Louisiana in August 2017, resulting in severe flooding and over 100 deaths. As the world faces growing threats from natural disasters caused by climate change, our ability to prepare for and recover from them depends on our ability to improve our monitoring and assessment of ground conditions before, during, and after such events.  

Aerial post-flood maps can provide localized insight into the extent of flood-related damage and the degree to which communities’ access to shelter, clean water, and communication channels have been compromised. So far, such insights typically only emerge hours or days after a flooding event has occurred. So we’re going to develop a segmentation model for serial post-flood aerial images which can generate a post-flood map.  

## PREPROCESSING 

As we first approached the dataset, we wanted to have a general idea of its characteristics. Our initial findings were: 

We have 299 training datasets and 75 testing datasets to predict. 

We have two sizes of the image in our training set and testing set: 3000*4000 and 3072*4592. We cannot train a model with different sizes of images, so we need to either resize them or crop them  

Each target value “mask” is a png, which we need to transform into the label we want to predict. There was an unequal distribution of labels, with most observations being labeled Trees / Shrub. 

From these results, we decided that:  

Split the training dataset into 75% training and 25% validation datasets to avoid over-fitting.  

We resize images into 256*256 in the beginning, after we have a baseline model, we’ll explore different methods to extract more key features from the dataset.  We transform the mask file into a matrix that contains the label we predict by using mask.astype(np.float). For the issue of unequal distribution of data, we will tune the model with hyperparameter class_weight. 

We start to process the data by the following steps. 

### a. Splitting data into train set and validation set:  

To avoid an uneven distribution of traits of images, we randomly shuffle the images and split them into training and validation datasets, which is 224 images for training and 75 images for validation.  

To accomplish this, we imported the random module and set the seed to 42 or a random number by preference, resulting in the same random sequence of numbers being generated each time the code is executed. The image list is then shuffled using the shuffle() function in the random module to ensure that the images are not in any particular order. 

We divided the image list into two lists after shuffling the images: train images filenames and val images filenames. The image list's first 224 images are assigned to the train images filenames list, while the remaining 75 images are assigned to the val images filenames list. By doing so, we can make sure that the training and validation sets are balanced.  

### b. Transformer:  

A common deep learning technique to prevent overfitting and improve the model’s generalization performance is to apply several image transformation methods to the training and validation set. Overfitting occurs when a model is performing badly on unseen data after being trained too well on the training dataset. In our case, since we have a smaller set of drone image data, it is very likely that overfitting may occur during the training process. Therefore, implementing image transformation methods would allow the model to be exposed to a broader range of variations in the training images and aid the model in acquiring more robust features.  

We defined transformers for the training set and validation set. Different transformation methods have been implemented during the tuning and training process. The methods we used are included in the “Albumentation” library which was imported as “A”.  

For the training set, we resized the image to 256x256 by using the Resize() method and then normalized it with the mean and standard deviation pre-trained by the Imagenet dataset. The mean we used is mean=[0.485, 0.456, 0.406] and the standard deviation we used is std=[0.229, 0.224, 0.225].  

We’ve also tried a few other transformation methods to see if they improved the performance of our models. Those include:  

ShiftScaleRotate(): this method is used to randomly shift, scale and rotate the images by a small amount that is fixed manually.  

RGBShift(): this method is used to shift the color channels of the images in a random manner. 

RandomBrightnessContrast(): this method is used to change the brightness and contrast of the images.  

RandomGamma(): this method is used to alter the gamma value of the image. Changing the gamma value leads to a larger variation in exposure.  

RandomRotate90(): this method is used to rotate images randomly by 90, 180, 270 degrees in hope of handling orientation of images better.  

HueSaturationValue(): this method is used to change the hue and saturation of the images. Changing this value allows the model to handle color balance better.  

The aforementioned image transformation methods that we implemented can assist the model in becoming more resistant to changes in lighting, color balance, and image orientation. This is really useful especially in real-world scenarios as well as in the validation and test sets where the images to which the model will be applied may not be of the same quality, color, lighting, orientation or have the same properties, for instance, ratio of houses and trees, as the training images.  

For the validation set, we resized and normalized it so we can apply the model to it precisely. The reason we did not apply any data augmentation methods to the validation set is because we want to preserve the validation set as it is of the true representation of the unseen real data. The transformation for both training and validation set is finished by converting the images to tensors.  

### c. Patching images:  

We have also tried cropping and patching images. For semantic segmentation, patching an image is a smart concept since it allows the model to concentrate on a smaller area of the image at a time. There are several existing methods for patching images that divide images into smaller overlapping or non-overlapping regions, for instance, window sliding and patchify. We have tried both methods as each have its own advantages.  

The sliding window method allows us to adjust the size of the window regardless of the scale of the object and can easily detect multiple objects at the same time, whereas patchify is easy to implement overlapped patches in a computationally efficient manner.  

Ideally, they can be helpful when the objects we want to identify in the images are small or when there is a lot of background noise. However, despite numerous tries with various window and patch sizes, we were unable to significantly enhance the performance of the model. Nevertheless, cropping images into smaller regions is theoretically helpful to gather more information about objects from the training sets.  

### d. Dataloader:  

We define dataloaders to split our dataset into batches for training our models. We start with batch_size = 16 and then adjust it to evaluate the corresponding performance. We have also set the shuffle argument to true for train dataloader and validation dataloader, so that the data in each dataset will be randomly shuffled at the beginning of each epoch.  

e. Further image processing:  

To retain more features from images, we decided to try to apply different methods after we create our baseline model. Instead of resizing all the images to 256*256 straightforwardly, we slice 3000*4000, and 3072*4592 images into 512*512 and save them. Then we resize the 512*512 image into a 256*256 image. Thus, the size of the training dataset became 8581 and the validation dataset became 2936.  

## LOSS FUNCTION 

There are several loss functions that are suitable for multi-class semantic segmentation. Those that are common in practice include cross-entropy loss, focal loss, dice loss, and tversky loss. When selecting from multiple loss functions, it is important to understand the trade-offs between the computation time and performance, as well as how specific our tasks are and what characteristics our images have. From the pixel count of the mask images of the training set we gathered, we can see that the pixel count of class 5, which is the label for grass, is 41 thousand times larger than the least pixel count in the training mask set, which is class 25, the label for boats.  For the same pixel count of class 5, it has almost 1.5 times more pixel count than class 24, representing flooded, that ranks second. It is clear that we are dealing with a class imbalanced dataset where certain classes are accounting for more data within the training set.  

In the case of class imbalance, cross-entropy loss may be sensitive to the imbalance, and the model may be biased towards class 5 and class 24. In this case, focal loss or Dice loss can be a better choice because they are less sensitive to class imbalance.  

By reducing the weight of the loss for pixels that are correctly classified, focal loss is intended to alleviate the problem of class imbalance. We will only need to tune the value for gamma, which is a modifying element that lowers the loss for cases with clear classifications. On the other hand, dice loss, which is calculated by the dice coefficient, assesses the degree of similarity between the predicted and actual segmentation masks. Dice loss is calculated as 1 - dice coefficient.  

## MODEL TUNING AND COMPARISON 

After feature engineering, we explored different models to evaluate the performance.  

U-net: We started with U-Net as our baseline model. The U-Net is a type of convolutional neural network (CNN) which is built on the architecture of fully convolutional neural networks but has been modified to work well with fewer training images and to produce more accurate segmentation results. The basic idea behind U-Net is to use a "contracting" network that is followed by an "expanding" network, where up sampling operations are used in the expanding network in place of pooling operations. This increases the resolution of the output. Additionally, the expanding network is able to learn to create a detailed output using the encoded information. We practiced the U-Net model with batch_size = 8, 16, 32 and learning rate = 0.001, 0.01, 0.05. We got our best-performed baseline model with a score = 68.54, batch_size = 16, epoch = 15, and learning rate = 0.01.  

The U-Net model that we built consists of two parts, the encoder structure and the decoder structure. The former is a series of down-sampling convolutional layers that extract features of the input images, and the latter is also a sequence of up-sampling layers that reconstruct the image to the output shape. What connects the two structures is a set of skip connections that concatenates the feature maps from the encoder side to the decoder side.  

After we got a baseline model, we tried to train the U-Net model with the dataset with further image processing (training set: 8581, validation set: 2936). We found that the accuracy rate hasn’t increased significantly after 10 epochs, and the final score is not as high as the original dataset.  

Also, we tried to train U-Net model with images that are resized to 300*400 to retain more features from the original dataset which contains images with sizes 3000*4000 and 3072*4592. However, it took too much time to compute, and the performance hasn’t increased significantly within 10 epochs. To reduce the time consumption and retain more computational power, we decided not to proceed with it. 

Additionally, in our earlier experiments, we have noticed that the training loss and accuracy are decreasing while the validation loss and accuracy are increasing, with the validation accuracy plateauing around a fixed value. We suspect that the model may be overfitting since the model is performing well on the training data but not so well on the validation data. Therefore, we tried a few techniques to mitigate this problem.  

First, we tried to introduce more dropouts after each convolutional layer and adjust the dropout rate between 0.2 to 0.5. When the rate is set to 0.2, this means that during each training iteration, 20% of the layer's neurons will be dropped out or ignored. This method is particularly useful since it forces the remaining neurons to learn and acquire more robust features, as well as reducing the number of training parameters so that when unseen data is being introduced, the model can generalize it better. 

Then, we looked into the possibility of stopping the training process earlier before the model starts to “memorize” the training data. This is the main issue when it comes to overfitting which results in a model that cannot generalize unseen data. Luckily, there are existing functions like the one called “EarlyStopping” from keras that allows us to monitor certain metrics, for instance, the validation loss and stop the model from training if the validation loss does not improve in a few iterations.  

As noted in the previous paragraph on “Transformer”, we also experimented with data augmentation and normalization as well as principal component analysis, which allows us to reduce the dimension of the data by projecting it into a space of a lower dimensionality. However, the latter is not really helpful in our case as we would discard some important features of the images that are useful for segmentation. It is yet an accomplished task to find the right balance between reducing the dimensionality of the data and preserving the useful information.  

We attempted to compare a ResNet101 deep convolutional neural network architecture to the U-Net architecture we had previously tried. Both of these architectures are intended for image segmentation, but ResNet101 is notable for having 101 layers and being specifically designed to address the issue of vanishing gradients. It was also trained on a large dataset, such as ImageNet, so it has a lot of image segmentation features. With this model, we eventually achieved the highest Kaggle score.  

We used accuracy to measure performance.  

| Model | Train Set Accuracy rate after tuning  | Validation Set Accuracy rate after tuning   | Kaggle score    | Comments |
| ------- | --- | --- | --- | --- |
| U-Net  | 0.50  | 0.47  | 66.7 | Baseline model |
| Resnet  | 0.52   | 0.62   | 73.05  | Best model |

## CONCLUSION  

Our goal was to create a post-flood map by developing a segmentation model for serial post-flood aerial images. During the preprocessing stage, we encountered several difficulties, including uneven image sizes and an unequal distribution of labels. We solved these problems by resizing the images to 256*256 and incorporating a class weight into the model.  

In training phase, we tested two deep learning architectures for image segmentation, U-Net and ResNet101. We discovered that the ResNet101 model performed better, earning the highest Kaggle score. Unfortunately, we were unable to improve our performance through patching. This could be because our dataset was not large enough to support patching, or patching was not the best approach for this specific dataset and task. 

If this project is continued, we could look into other models like DeepLabV3+ or FCN. We could also experiment with different image sizes and transformation methods to extract more key features from the dataset. Some other techniques can even be tested out, such as ensemble methods, to improve the performance of the models. 

## REFERENCES  

1. ‘DeepLabV3+ (ResNet101) for Segmentation (PyTorch) | Kaggle’. Accessed 25 January 2023. https://www.kaggle.com/code/balraj98/deeplabv3-resnet101-for-segmentation-pytorch/notebook.  

2. Heydarian, Amirhossein. ‘U-Net for Semantic Segmentation on Unbalanced Aerial Imagery’. Medium, 5 October 2021. https://towardsdatascience.com/u-net-for-semantic-segmentation-on-unbalanced-aerial-imagery-3474fa1d3e56.  

3. 206 - The Right Way to Segment Large Images by Applying a Trained U-Net Model on Smaller Patches, 2021. https://www.youtube.com/watch?v=LM9yisNYfyw.  

4. 208 - Multiclass Semantic Segmentation Using U-Net, 2021. https://www.youtube.com/watch?v=XyX5HNuv-xE.
