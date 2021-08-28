import streamlit as st
from PIL import Image
import cv2
import numpy as np
import os
import imghdr
import base64
import pandas as pd
from Unet_I import binary_unet
from Unet_II import inpaint_unet

st.title(":mask: Face Mask Inpainting :kissing:")
st.write("---")

sidebar_options = ["Project Info","Demo Image","Upload your Image","Training Analysis"]
st.sidebar.success("HAVE FUN ~ ~ :smiley_cat: :dog:")
st.sidebar.write('---')
st.sidebar.title("Explore the following")
box = st.sidebar.selectbox("Please select from the following:",sidebar_options)

def run_demo(name):
    st.image("demo_imgs/gt_imgs/"+name+".jpg",width=450,caption="Ground Truth")
    col1,col2 = st.columns(2)
    with col1:
        st.image("demo_imgs/masked_imgs/mask_"+name+".jpg",caption="Masked Photo")
    with col2:
        st.image("demo_imgs/binary_imgs/binary_"+name+".jpg",caption="Binary Segmentation Map")
    st.image("demo_imgs/pred_imgs/fake_"+name+".jpg",width=450,caption="Face-Mask Inpainted Photo")

if box == "Project Info":

    col1,col2,col3 = st.columns(3)
    with col1:
        st.image("info_imgs/test1.jpg")
        st.image("info_imgs/test2.jpg")
    with col2:
        st.markdown(" $~$\n\n$~$\n\n $~~~~~~~~~~~~~~$ ---------->")
        st.markdown(" $~$\n\n$~$\n\n$~$\n\n$~$\n\n$~$\n\n$~~~~~~~~~~~~~~$ ----------->")
    with col3:
        st.image("info_imgs/test1.gif")
        st.image("info_imgs/test2.gif")
    
    st.markdown("This project attempted to achieve the paper **[A novel GAN-based network for unmasking of "
                "masked face](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9019697)**. The model "
                "is designed to remove the face-mask from facial image and inpaint the left-behind region based "
                "on a novel GAN-network approach. ")
    st.image("info_imgs/md_archi.png")
    st.markdown("Rather than using the traditional pix2pix U-Net method, in this work the model consists of two main modules, "
                "map module and editing module. In the first module, we detect the face-mask object and generate a "
                "binary segmentation map for data augmentation. In the second module, we train the modified U-Net "
                "with two discriminators using masked image and binary segmentation map.")
    st.markdown("***Feel free to play it around:***")
    st.markdown(":point_left: To get started, you can choose ***Demo Image*** to see the performance of the model.")
    st.markdown(":camera: Feel free to ***upload*** any masked image you want and see the performance.")
    st.markdown(":chart_with_upwards_trend: Also, press ***Training Analysis*** to see the training insight.")


elif box == "Demo Image":
    st.sidebar.write("---")

    demoimg_dir = "demo_imgs/gt_imgs"
    photos=[]
    for file in os.listdir(demoimg_dir):
        filepath = os.path.join(demoimg_dir,file)
        if imghdr.what(filepath) is not None:
            photos.append(file[:-4])
    photos.sort()

    inpaint_option = st.sidebar.selectbox("Please select a sample image, then click the 'Inpaint!' button.",photos)
    inpaint = st.sidebar.button("Inpaint !")

    if inpaint:
        st.empty()
        run_demo(inpaint_option)


elif box == "Upload your Image":
    st.sidebar.info('Please upload ***single masked person*** image. For best result, please also ***center the face*** in the image, and the face mask should be in ***light green/blue color***.')
    image = st.file_uploader("Upload your masked image here",type=['jpg','png','jpeg'])
    if image is not None:
        col1,col2 = st.columns(2)
        masked = Image.open(image).convert('RGB')
        masked = np.array(masked)
        masked = cv2.resize(masked,(224,224))
        with col1:
            st.image(masked,width=300,caption="masked photo")
        binary = binary_unet(masked)
        with col2:
            st.image(binary,width=300,caption="binary segmentation map")

        fake = inpaint_unet(masked,binary)
        st.image(fake,width=600,caption="Inpainted photo")

elif box=="Training Analysis":
    fid_frames = []
    for i in range (1,17):
        f = pd.read_csv("metrics_20k/FID_epoch"+str(i),header=None)
        fid_frames.append(f)
    df = pd.concat(fid_frames)
    dffid = pd.DataFrame(columns=['iters','FID'])
    for i in range(len(df)):
        if i%2==0:
            new_row = {"iters":df.iloc[i].values[0],"FID":df.iloc[i+1].values[0]}
            dffid = dffid.append(new_row,ignore_index=True)
    dffid.set_index("iters",inplace=True)
    st.write("$~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~$***Frechet Inception Distance (FID)***")
    st.line_chart(dffid)


    losses_frames = []
    df_header = pd.read_csv("metrics_20k/loss_epoch1",header=None)[0:1]
    df_header = df_header.to_string()[66:]
    df_loss = pd.DataFrame(columns=(df_header.split('   ')))
    df_loss.insert(0,'iters','') 
    for i in range (1,17):
        f = pd.read_csv("metrics_20k/loss_epoch"+str(i),header=None)[1:]
        losses_frames.append(f)
    df_l = pd.concat(losses_frames)

    for i in range(len(df_l)):
        if i%2==0:
            new_row = {"iters":float(df.iloc[i].values[0])}
        if i%2!=0:
            loss_terms = df_l.iloc[i].values[0].split('    ')
            new_row["gen"]=float(loss_terms[0])
            new_row["disc_whole"]=float(loss_terms[1])
            new_row["disc_mask"]=float(loss_terms[2])
            new_row["l1_loss"]=float(loss_terms[3])
            new_row["ssim_loss"]=float(loss_terms[4])
            new_row["percep"]=float(loss_terms[5])
            df_loss = df_loss.append(new_row,ignore_index=True)
    df_loss.set_index("iters",inplace=True)

    dfgen = df_loss[["gen"]]
    st.write("$~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~$***Generator Loss***")
    st.line_chart(dfgen)

    dfdisc = df_loss[["disc_whole","disc_mask"]]
    st.write("$~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~$***Discriminators Loss***")
    st.line_chart(dfdisc)

    dflosses = df_loss[["l1_loss","ssim_loss","percep"]]
    st.write("$~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~$***L1,SSIM,Perceptual Loss***")
    st.line_chart(dflosses)

    


