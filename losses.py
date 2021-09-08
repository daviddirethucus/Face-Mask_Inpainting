import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
from torchvision.models import vgg19
from torchvision.models import inception_v3
import scipy.linalg
import numpy as np
device = 'cuda'

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        ###
        img1 = (img1+1)/2
        img2 = (img2+1)/2
        ###
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)



def recon_loss(gt,fake,recon_criterion):
    ssim = SSIM()
    ssim_loss = ssim(gt,fake)
    l1_loss = recon_criterion(gt,fake)
    return l1_loss,ssim_loss



class PerceptualNet(nn.Module):
    def __init__(self, name = "vgg19", resize=True):
        super(PerceptualNet, self).__init__()
        blocks = []
        blocks.append(vgg19(pretrained=True).features[:4].eval())
        blocks.append(vgg19(pretrained=True).features[4:9].eval())
        blocks.append(vgg19(pretrained=True).features[9:16].eval())
        blocks.append(vgg19(pretrained=True).features[16:23].eval())
        
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks).to(device)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)).to(device)
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)).to(device)
        self.resize = resize
    
    def forward(self, inputs, targets):
        if inputs.shape[1] != 3:
            inputs = inputs.repeat(1, 3, 1, 1)
            targets = targets.repeat(1, 3, 1, 1)
        inputs = (inputs+1)/2
        targets = (targets+1)/2
        if self.resize:
            inputs = self.transform(inputs, mode='bilinear', size=(224, 224), align_corners=False)
            targets = self.transform(targets, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = inputs
        y = targets
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.l1_loss(x, y)
        return loss

def percep_loss(gt,fake):
    percep_net = PerceptualNet()
    return percep_net(gt,fake)



def normalize(img):
    return (img-(-1))/(1-(-1))
def anti_normalize(img):
    return img*(1-(-1))+(-1)

def discwhole_loss_func(disc_whole,gt,mask,binary,fake,adv_criterion,lambda_Dwhole):
    input_imgs = torch.cat((mask,binary),1)
    fake_pred = disc_whole(fake.detach(),input_imgs)
    gt_pred = disc_whole(gt,input_imgs)
    fake_loss = adv_criterion(fake_pred,torch.zeros_like(fake_pred))
    gt_loss = adv_criterion(gt_pred,torch.ones_like(gt_pred))
    return lambda_Dwhole * (fake_loss+gt_loss)/2


def discmask_loss_func(disc_mask, gt,fake,mask,binary, adv_criterion, lambda_Dmask): 
    nor_mask = normalize(mask)
    nor_binary = normalize(binary)
    nor_fake = normalize(fake)
    
    oofs = torch.mul(nor_mask,1-nor_binary)
    oops = torch.mul(nor_fake,nor_binary)
    ooo = anti_normalize(oofs+oops)
    input_imgs = torch.cat((mask,binary),1)
    fake_pred = disc_mask(ooo.detach(),input_imgs)
    gt_pred = disc_mask(gt,input_imgs)
    
    fake_loss = adv_criterion(fake_pred,torch.zeros_like(fake_pred))
    gt_loss = adv_criterion(gt_pred,torch.ones_like(gt_pred))
    
    return lambda_Dmask * (fake_loss+gt_loss)/2



def gen_adv_loss(gen,disc, gt,mask,binary, adv_criterion):
    input_imgs = torch.cat((mask,binary),1)
    fake = gen(input_imgs)
    fake_pred = disc(fake,input_imgs)
    adv_loss = adv_criterion(fake_pred,torch.ones_like(fake_pred))
    return adv_loss,fake

def generator_loss(cur_step,gen,disc_whole,disc_mask, gt,mask,binary,
                  adv_criterion,recon_criterion,
                  lambda_recon,lambda_adv_whole,lambda_adv_mask):
    if cur_step<3516*6:
        adver_loss_whole,fake = gen_adv_loss(gen,disc_whole,gt,mask,binary,adv_criterion)
        l1_loss,ssim_loss = recon_loss(gt,fake,recon_criterion)
        reconstruction_loss = l1_loss*0.5 + (1-ssim_loss)*0.5
        perceptual_loss = percep_loss(gt,fake)
        gen_loss = lambda_recon*(reconstruction_loss+perceptual_loss)+lambda_adv_whole*adver_loss_whole
    else:
        adver_loss_whole,fake = gen_adv_loss(gen,disc_whole,gt,mask,binary,adv_criterion)
        adver_loss_mask,fake = gen_adv_loss(gen,disc_mask,gt,mask,binary,adv_criterion)
        l1_loss,ssim_loss = recon_loss(gt,fake,recon_criterion)
        reconstruction_loss = l1_loss*0.5 + (1-ssim_loss)*0.5
        perceptual_loss = percep_loss(gt,fake)
        gen_loss = lambda_recon*(reconstruction_loss+perceptual_loss)+lambda_adv_whole*adver_loss_whole+lambda_adv_mask*adver_loss_mask
    
    
    return gen_loss,fake,l1_loss,ssim_loss,perceptual_loss




inception_model = inception_v3(pretrained=True)
inception_model.to(device)
inception_model = inception_model.eval() # Evaluation mode
inception_model.fc = torch.nn.Identity()

def matrix_sqrt(x):
    y = x.cpu().detach().numpy()
    y = scipy.linalg.sqrtm(y)
    return torch.Tensor(y.real,device=x.device)

def frechet_distance(mu_x,mu_y,sigma_x,sigma_y):
    return torch.norm(mu_x-mu_y)**2 + torch.trace(sigma_x+sigma_y-2*matrix_sqrt(sigma_x@sigma_y))

def get_covariance(features):
    return torch.Tensor(np.cov(features.detach().numpy(),rowvar=False))

