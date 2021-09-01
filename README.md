# :mask: Face-Mask Inpainting :kissing:

| | |
|:-------------------------:|:-------------------------:|
|<img src="info_imgs/test1.jpg" alt="screen" width="250px" > | <img src="info_imgs/test1.gif" alt="screen" width="250px" > |
|<img src="info_imgs/test2.jpg" alt="screen" width="250px" > | <img src="info_imgs/test2.gif" alt="screen" width="250px" > |


This project attempted to achieve the paper **[A novel GAN-based network for unmasking of 
masked face](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9019697)**. The model 
is designed to remove the face-mask from facial image and inpaint the left-behind region based 
on a novel GAN-network approach.

## Training Environment
- Google Cloud Platform 
- GPU (Nvidia Tesla T4) 
- Python 3.8

## Models Architecture
<img src="info_imgs/md_archi.png" alt="screen" width="700px" >

Rather than using the traditional pix2pix U-Net method, in this work the model consists of two main modules, 
**map module** and **editing module**.In the first module, we detect the face-mask object and generate a 
binary segmentation map for data augmentation. In the second module, we train the modified U-Net 
with two discriminators using masked image and binary segmentation map.
### Data preparation
- For collecting the ground truth, we use **[Flickr-Faces-HQ Dataset (FFHQ)](https://github.com/NVlabs/ffhq-dataset)**.
- For creating the masked images, we use **[MaskTheFace](https://github.com/aqeelanwar/MaskTheFace)** to masking the ground truth. 

## Get Started
It is recommended to make a **[new virtual environment](https://towardsdatascience.com/manage-your-python-virtual-environment-with-conda-a0d2934d5195)** with **Python 3.8** and install the dependencies. Following steps
can be taken to download and run the Face-mask inpainting streamlit webapp on local host
### Clone the repository
```
git clone https://github.com/daviddirethucus/Face-Mask_Inpainting.git
```
### Download the trained models
Since it is not permissable to push the model which is larger than 100MB on Github, so we provide a link to download our trained Facemask Inpainting models: **[Here](https://drive.google.com/drive/folders/1l-5ntQyPi4hy1oc_3BHHTNCY4w4nzfEk?usp=sharing)**

The path of the trained models should be located at: 
```
/Face-Mask_Inpainting/models
```
### Install required packages
The provided requirements.txt file consists the essential packages to install. Use the following command
```
cd Face-Mask_Inpainting
pip install -r requirements.txt
```
### Run the stremalit webapp
```
cd Face-Mask_Inpainting
streamlit run main.py
```
Copy the **Local URL** / **Network URL** and view it in your browser.
<img src="info_imgs/terminal_s.png" alt="screen" width="550px" >

### Demo
|<img src="info_imgs/demo1.png" alt="screen" width="400px" > | <img src="info_imgs/demo2.png" alt="screen" width="400px" > |

## Related Project

## Paper References
- [A novel GAN-based network for unmasking of masked face](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9019697)
- [Generation of Realistic Facemasked Faces With GANs](http://cs230.stanford.edu/projects_winter_2021/reports/70681837.pdf)
---
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf)
- [Squeeze-and-Excitation Networks](https://arxiv.org/pdf/1709.01507.pdf)
- [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/pdf/1706.05587.pdf)
- [Language Modeling with Gated Convolutional Networks](https://arxiv.org/pdf/1612.08083.pdf)
---
- [Image Quality Assessment: From Error Visibility to Structural Similarity](https://www.cns.nyu.edu/pub/lcv/wang03-preprint.pdf)
- [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/pdf/1603.08155.pdf)

## Code References
- https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4
- https://github.com/VainF/pytorch-msssim
- https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49