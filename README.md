# Semantic-Segmentation-using-Deep-Learning
Project Goal:

This project aims to solve a semantic segmentation task using deep learning models. 

Dataset: Cambridge Labeled Objects in Video
It consists of 101 images (960x720 pixel) in which each pixel was manually assigned to one of the following 32 object classes that are relevant in a driving environment:
<img width="422" alt="image" src="https://github.com/kiransparakkal/Semantic-Segmentation-using-Deep-Learning/assets/70934344/977dc334-b16e-43eb-b10a-e9e7bc43c616">


The "void" label indicates an area which ambiguous or irrelevant in this context. The colour/class association is given in the file label_colors.txt. Each line has the R G B values (between 0 and 255) and then the class name.

Example of original and labelled frames:
<img width="455" alt="image" src="https://github.com/kiransparakkal/Semantic-Segmentation-using-Deep-Learning/assets/70934344/1f5a7d68-1fdd-4ea6-be0b-14258447537d">


Image format and naming: All images (original and ground-truth) are in uncompressed 24-bit colour PNG format. For each frame from the original sequence, its corresponding labelled frame bears the same name, with an extra "_L" before the ".png" extension. For example, the first frame is "0016E5_07959.png" and the corresponding labelled frame is "0016E5_07959_L.png".

The dataset (including training dataset, testing dataset, label_colors.text) can be downloaded via: Cam101.zipDownload Cam101.zip
Introduction of the task:
Different from the traditional image segmentation methods (which are based on colour spaces, clustering methods or Watershed algorithm), we leverage a deep neural network to segment each pixel of an image into a category, which is called semantic segmentation. You will be required to write Python code to build various deep-learning models for semantic segmentation on the given dataset and compare the performance of different segmentation models. 
In this project, we will provide the dataset, which includes the training dataset and testing dataset. You need to explore at least three different semantic segmentation model structures. For example,
Fully Convolutional Networks (FCN) (e.g., FCN-16s, FCN-8s)
UNet
Pyramid scene parsing network (PSPNet)
DeepLab series (e.g., DeepLabV2, DeepLabV3, DeepLabV3+)
For other model structures, please see: https://paperswithcode.com/sota/semantic-segmentation-on-pascal-voc-2012Links to an external site.
Links to an external site.You also need to adopt data augmentation techniques (pls see: https://albumentations.ai/docs/introduction/image_augmentation/Links to an external site. ) to expand the dataset. You can use this library (https://albumentations.ai/docs/getting_started/mask_augmentation/Links to an external site.) to do the data augmentation; or you can use other libraries or implement the data augmentation by yourself from scratch.
When training your models, you can have two options:
First option: you can build a model and train the model from scratch
Second option: you can download a pre-trained model with various backbones, like resnet50, resnet101, mobilenet_v2 and so on (For more details, please refer to https://pytorch.org/vision/stable/models.html)Links to an external site.. As the pre-trained model was previously trained on a large dataset, typically on a large-scale image classification task, the model will effectively serve as a generic model of the visual world. You can just take advantage of these learned feature maps without having to start from scratch by training a large model on a large dataset. Hence, you can unfreeze a few of the top layers of a frozen base model and jointly train both the newly added classifier layers and the last layers of the base model. This allows us to "fine-tune" the higher-order feature representations in the base model to make them more relevant for the specific task.
You can freely use any Python library to complete this project. However, your code must include the following steps
Indicate the imported packages/libraries
Load the dataset, understand the dataset, and visualize the dataset:
Print out the number of training and testing samples in the dataset.
Plot some figures to visualize some samples.
Pls explore data augmentation techniques to expand the data in the training dataset.
Split the augmented training dataset into the training set and validation set (90% vs 10%). Build a training, validation, and testing dataset pipeline (like DataLoader in PyTorch)
For each model structure, you can choose either (i) build the model from scratch or (ii) load in the pre-trained backbone models with pre-trained weights and add the classification layers for your specific task. To train the model, you need to find suitable hyperparameters for your model, such as learning rate, optimizer, and loss function (such as Pixel-wise Cross Entropy, Weighted Pixel Cross Entropy, Dice Coefficient), etc.: 
Doing the validation on the validation dataset and finding your best version of the model (i.e., the model with the best evaluation results on the validation dataset), you can use cross-validation to find the best hyperparameters.
You can use the mean Intersection over Union (IoU) and pixel accuracy as the evaluation metrics (Pixel-wise accuracy indicates the ratio of pixels which are correctly predicted, while mean IoU indicates the Intersection of the Union of pixels averaged over all the semantic categories) please:
Plot the loss function graph with respect to the epoch on the training and validation set.
Plot the evaluation metrics graph with respect to the iteration/epoch.
If you can use either CPU or GPU to train your model and please print the training time.
After you obtain the best version of each model structure, please test the models on the testing dataset:
Load your model and print some results in the testing dataset, and compare them with the original input, which will be similar to the below figure.
Summarize the performance of each model using evaluation metrics (pixel accuracy and mean IoU).
