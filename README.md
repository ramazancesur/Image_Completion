###### Disclaimer:
This is an unofficial TensorFlow (version 1.1.0) implementation of a publicly available research paper on InPainting i.e. completing images based on the surrounding pixel values.

The entire code has been written by me with the sole purpose of exploration. I don't have any intention of gaining any monetary benefits, whatsoever, from it.

The research paper can be found at the below location.

<http://iizuka.cs.tsukuba.ac.jp/projects/completion/en/>


______________________________________


## Demo

Please go the link mentioned below for demo web service.


<http://inpaint.online>

######
Source Code for this web service can be found at below mentioned link.

<https://github.com/karantrehan/Image_Completion_Web_Service>



## Summary

In this research paper authors have tried to solve the challenging problem of image completion using surrounding pixel values, also known as InPainting.

Here, they have taken a learning based approach where the neural network can be trained once on a large no. of real images and then be used for completing patches in the new images.

The Neural Network Architecture used for training is a combination of Convolutional Neural Networks(CNNs) and Generative Adversarial Networks(GANs) and is depicted in the picture right below.


*The picture below has been taken from the research paper and gives a pictorial representation of the neural network architecture used for training the model*

![network.png](metadata/images/network.png)

image_courtsey - Research Paper


## Model Architecture

Model architecture is divided in two parts, Generator and Discriminator.


Generator is used for completing the patched image. It consists of several convolutional layers and is used to re-generate the entire image when a patched image is fed to the model.
The regenerated image is then completed by using patch area from it and the surrounding area from the original image.

Discriminator, on the other hand, is used for identifying if the image fed to the discriminator is original or completed and is required only during the model training process.
Once training is over, we discard the Discriminator and use only Generator for completing images. It consists of two separate networks, Global Discriminator and Local Discriminator.
Global Discriminator takes the entire image as input whereas Local Discriminator takes only surrounding area of the patch and the patch itself. Outputs of both Global and Local Discriminators are
concatenated and fed to a logistic function which gives probability of image being real.


## Model Training

Model Training part is divided in three parts:

1. Training only Generator - In this part only generator network is trained using L1 loss. In this a patched image is fed to the generator network which completes the image(fills the patch).
The objective of training in this part is to minimize L1 distance between original image(without patch) and completed image.

2. Training only Discriminator - In this part only discriminator is trained using Cross Entropy loss. Here completed and original images are fed to the discriminator alternatively and
the Cross Entropy loss is minimized over the training process.

3. Training both Generator and Discriminator - In this part both Generator and Discriminator are trained. Here Generator is trained using both L1 and GAN loss, GAN loss is nothing but reverse of Cross Entropy
where the objective is to train generator in such a way that the generated images are very close to the original ones and discriminator is forced to classify them as original(real).
Discriminator is trained the same way as in part 2.
______________________________________

*The following images are from my model training process and depict network graph, loss values and completed images during training*


#### Tensorflow Graph


![network_tf_graph.png](metadata/images/network_tf_graph.png)


Discriminator           |  Generator
:-------------------------:|:-------------------------:
![discriminator.png](metadata/images/discriminator.png) |  ![generator.png](metadata/images/generator.png)
______________________________________


#### Loss Value Visualization


Generator Loss (L1)        |Generator Loss (L1 and GAN)|  Discriminator Loss (Cross Entropy)
:-------------------------:|:-------------------------:|:------------------------
![generator_loss_L1.png](metadata/images/generator_loss_L1.png)|![generator_loss_L1_GAN.png](metadata/images/generator_loss_L1_GAN.png)|![discriminator_loss.png](metadata/images/discriminator_loss.png)

______________________________________


####  Completed Images on Tensorboard


![train_end.png](metadata/images/train_end.png)
______________________________________


#### Images Comparison


Patched | Completed | Patched | Completed | Patched | Completed
:-------------------------:|:-------------------------:|:------------------------:|:------------------------|:------------------------:|:------------------------
![image_0_original_to_be_patched.jpg](results/train/main_session/image_0_original_to_be_patched.jpg)|![image_0_both_gen_dis.jpg](results/train/main_session/image_0_both_gen_dis.jpg)|![image_1_original_to_be_patched.jpg](results/train/main_session/image_1_original_to_be_patched.jpg)|![image_1_both_gen_dis.jpg](results/train/main_session/image_1_both_gen_dis.jpg)|![image_2_original_to_be_patched.jpg](results/train/main_session/image_2_original_to_be_patched.jpg)|![image_2_both_gen_dis.jpg](results/train/main_session/image_2_both_gen_dis.jpg)
![image_3_original_to_be_patched.jpg](results/train/main_session/image_3_original_to_be_patched.jpg)|![image_3_both_gen_dis.jpg](results/train/main_session/image_3_both_gen_dis.jpg)|![image_4_original_to_be_patched.jpg](results/train/main_session/image_4_original_to_be_patched.jpg)|![image_4_both_gen_dis.jpg](results/train/main_session/image_4_both_gen_dis.jpg)|![image_5_original_to_be_patched.jpg](results/train/main_session/image_5_original_to_be_patched.jpg)|![image_5_both_gen_dis.jpg](results/train/main_session/image_5_both_gen_dis.jpg)
![image_6_original_to_be_patched.jpg](results/train/main_session/image_6_original_to_be_patched.jpg)|![image_6_both_gen_dis.jpg](results/train/main_session/image_6_both_gen_dis.jpg)|![image_7_original_to_be_patched.jpg](results/train/main_session/image_7_original_to_be_patched.jpg)|![image_7_both_gen_dis.jpg](results/train/main_session/image_7_both_gen_dis.jpg)|![image_8_original_to_be_patched.jpg](results/train/main_session/image_8_original_to_be_patched.jpg)|![image_8_both_gen_dis.jpg](results/train/main_session/image_8_both_gen_dis.jpg)
![image_9_original_to_be_patched.jpg](results/train/main_session/image_9_original_to_be_patched.jpg)|![image_9_both_gen_dis.jpg](results/train/main_session/image_9_both_gen_dis.jpg)|![image_10_original_to_be_patched.jpg](results/train/main_session/image_10_original_to_be_patched.jpg)|![image_10_both_gen_dis.jpg](results/train/main_session/image_10_both_gen_dis.jpg)|![image_11_original_to_be_patched](results/train/main_session/image_11_original_to_be_patched.jpg)|![image_11_both_gen_dis](results/train/main_session/image_11_both_gen_dis.jpg)


**Note:** The results above are on a miniscule train dataset and these are training images only. Training on actual train data and then testing on test data is not possible on a personal laptop due to hardware inefficiency. Moreover, i just wanted to see if i was going in the right direction and if the code was working as intended or not. As a result, the model available here, on Github, doesn't work well on test images.
      As per the authors, to get results like those in the published paper the entire training process takes around 2 months on a single machine with 4 K80 GPUs.

______________________________________

