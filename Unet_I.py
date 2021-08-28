import cv2
import torch
import numpy as np
from torch import nn
from torchvision import transforms
from torchvision.utils import make_grid


def crop(image,new_shape):
    middle_height = image.shape[2]//2
    middle_width = image.shape[3]//2
    starting_height = middle_height-round(new_shape[2]/2)
    final_height = starting_height+new_shape[2]
    starting_width = middle_width-round(new_shape[3]/2)
    final_width = starting_width+new_shape[3]
    cropped_image = image[:,:,starting_height:final_height,starting_width:final_width]
    return cropped_image


class ContractingBlock(nn.Module):
    def __init__(self,input_channels,use_dropout=False,use_in=True):
        super(ContractingBlock,self).__init__()
        self.conv = nn.Conv2d(input_channels,input_channels*2,kernel_size=3,padding=1)
        self.activation = nn.LeakyReLU(0.2)
        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        if use_in:
            self.insnorm = nn.InstanceNorm2d(input_channels*2)
        self.use_in = use_in
        if use_dropout:
            self.dropout = nn.Dropout()
        self.use_dropout = use_dropout
    
    def forward(self,x):
        x = self.conv(x)
        if self.use_in:
            x = self.insnorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        x = self.maxpool(x)
        return x

class ExpandingBlock(nn.Module):
    def __init__(self,input_channels,use_dropout=False,use_in=True):
        super(ExpandingBlock,self).__init__()
        self.tconv = nn.ConvTranspose2d(input_channels,input_channels//2,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.conv2 = nn.Conv2d(input_channels, input_channels // 2, kernel_size=3, padding=1)
        if use_in:
            self.insnorm = nn.InstanceNorm2d(input_channels//2)
        self.use_in = use_in
        if use_dropout:
            self.dropout = nn.Dropout()
        self.use_dropout = use_dropout
        self.activation = nn.LeakyReLU(0.2)
    
    def forward(self,x,skip_x):
        x = self.tconv(x)
        skip_x = crop(skip_x,x.shape)
        x = torch.cat([x,skip_x],axis=1)
        x = self.conv2(x)
        if self.use_in:
            x = self.insnorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        return x

class FeatureMapBlock(nn.Module):
    def __init__(self,input_channels,output_channels):
        super(FeatureMapBlock,self).__init__()
        self.conv = nn.Conv2d(input_channels,output_channels,kernel_size=1)
    
    def forward(self,x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self,input_channels,output_channels,hidden_channels=32):
        super(UNet,self).__init__()
        self.upfeature = FeatureMapBlock(input_channels,hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels,use_in=False)
        self.contract2 = ContractingBlock(hidden_channels*2)
        self.contract3 = ContractingBlock(hidden_channels*4)
        self.contract4 = ContractingBlock(hidden_channels*8)
        self.contract5 = ContractingBlock(hidden_channels*16)
        self.expand0 = ExpandingBlock(hidden_channels*32)
        self.expand1 = ExpandingBlock(hidden_channels*16)
        self.expand2 = ExpandingBlock(hidden_channels*8)
        self.expand3 = ExpandingBlock(hidden_channels*4)
        self.expand4 = ExpandingBlock(hidden_channels*2)
        self.downfeature = FeatureMapBlock(hidden_channels,output_channels)
        self.tanh = torch.nn.Tanh()
    
    def forward(self,x):
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        x4 = self.contract4(x3)    #x4:512
        x5 = self.contract5(x4)    #x5:1024
        x6 = self.expand0(x5,x4)
        x7 = self.expand1(x6,x3)
        x8 = self.expand2(x7,x2)
        x9 = self.expand3(x8,x1)
        x10 = self.expand4(x9,x0)
        xn = self.downfeature(x10)
        return self.tanh(xn)


class Discriminator(nn.Module):
    def __init__(self,input_channels,hidden_channels=8):
        super(Discriminator,self).__init__()
        self.upfeature = FeatureMapBlock(input_channels,hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels,use_in=False)
        self.contract2 = ContractingBlock(hidden_channels*2)
        self.contract3 = ContractingBlock(hidden_channels*4)
        self.contract4 = ContractingBlock(hidden_channels*8)
        self.final = nn.Conv2d(hidden_channels*16,1,kernel_size=1)   #should change?
        
    def forward(self,x): 
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        x4 = self.contract4(x3)
        xn = self.final(x4)
        return xn


def binary_unet(img):
    transform = transforms.Compose([
            transforms.ToTensor()
            ])

    input_dim = 3
    binary_dim = 1         # (-1,1,224,224)
    lr = 0.0002
    device = 'cpu'
    gen = UNet(input_dim,binary_dim).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(),lr=lr)
    disc = Discriminator(binary_dim).to(device)
    disc_opt = torch.optim.Adam(disc.parameters(),lr=lr)

    pretrained = True
    if pretrained:
        loaded_state = torch.load("models/Bi_UNet.pth",map_location=torch.device('cpu'))
        gen.load_state_dict(loaded_state["gen"])
        gen_opt.load_state_dict(loaded_state["gen_opt"])
        disc.load_state_dict(loaded_state["disc"])
        disc_opt.load_state_dict(loaded_state["disc_opt"])

    img = transform(img)
    img = img.detach().cpu().view(-1,*(3,224,224))
    image_tensor = gen(img)
    image_tensor = (image_tensor + 1) / 2
    #image_tensor = image_tensor.detach().cpu().view(-1, *(1,224,224))
    image_grid = make_grid(image_tensor[:1], nrow=5)
    image_tensor = image_grid.permute(1, 2, 0).squeeze()
    image = np.array(image_tensor)

    kernel = np.ones((4,4),np.uint8)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret, imgg = cv2.threshold(image,0.7,1,cv2.THRESH_BINARY)
    opening = cv2.morphologyEx(imgg, cv2.MORPH_OPEN, kernel)

    return opening
