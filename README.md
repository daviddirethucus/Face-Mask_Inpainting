# :mask: Face-Mask Inpainting :kissing:

| | |
|:-------------------------:|:-------------------------:|
|<img src="info_imgs/test1.jpg" alt="screen" width="300px" > | <img src="info_imgs/test1.gif" alt="screen" width="300px" > |
|<img src="info_imgs/test2.jpg" alt="screen" width="300px" > | <img src="info_imgs/test2.gif" alt="screen" width="300px" > |


This project attempted to achieve the paper **[A novel GAN-based network for unmasking of 
masked face](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9019697)**. The model 
is designed to remove the face-mask from facial image and inpaint the left-behind region based 
on a novel GAN-network approach.

## Training Environment
- Google Cloud Platform 
- GPU (Nvidia Tesla T4) 
- Python 3.8

## Models Architecture
<img src="info_imgs/md_archi.png" alt="screen" width="500px" >

Rather than using the traditional pix2pix U-Net method, in this work the model consists of two main modules, 
map module and editing module.In the first module, we detect the face-mask object and generate a 
binary segmentation map for data augmentation. In the second module, we train the modified U-Net 
with two discriminators using masked image and binary segmentation map.

### Data preparation
- For collecting the ground truth, we use **[Flickr-Faces-HQ Dataset (FFHQ)](https://github.com/NVlabs/ffhq-dataset)**.
- For creating the masked images, we use **[MaskTheFace](https://github.com/aqeelanwar/MaskTheFace)** to masking the ground truth. 
