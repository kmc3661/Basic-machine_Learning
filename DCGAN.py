from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn as parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from torch.utils.tensorboard import SummaryWriter

#Setting hyper parameters

manualSeed=999
#manualSeed = random.randint(1,10000), use this when I want new results
print("RandomSeed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

dataroot = "/home/minchan/anconda3/Data/coco/images"
workers = 8
batch_size = 128
image_size = 64
nc = 3
nz = 100
ngf = 128 # size of feature maps in generator
ndf = 64 # size of feature maps in discriminator
num_epochs = 50
lr = 0.0002
beta1 = 0.5 # Beta1 hyperparameter for Adam
ngpu = 4
mean=0.5
std=0.5
#Data

dataset = dset.ImageFolder(root=dataroot,
                           transform = transforms.Compose([
                                                           transforms.Resize(image_size),
                                                           transforms.CenterCrop(image_size),
                                                           transforms.ToTensor(),
                                                           transforms.Normalize((mean, mean, mean), (std, std, std)),
                           ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size,
                                         shuffle = True, num_workers=workers)
device = torch.device("cuda:0" if(torch.cuda.is_available() and ngpu >0) else "cpu")

# real_batch = next(iter(dataloader)) # for i in enumulate(dataloader)
# plt.figure(figsize=(8,8))
# plt.axis("off")
# plt.title("Training Images")
# plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64],padding=2,
#                                          normalize=True).cpu(),(1,2,0)))

def weights_init(m):
    classname = m.__class__.__name__ # class name을 string으로 받을 수 있음
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02) # 평균 0,표준편차 0.02
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Generator & discriminator
# Generator code

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz,ngf*8,kernel_size=4,stride=1,padding=0, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True), #ReLU안의 inplace=True는 추가적인 output 할당없이 바로 input을 수정 -> memory usage를 약간 줄여줌
            #state size : ngf*8 x 4 x 4
            nn.ConvTranspose2d(ngf*8, ngf*4, 4,2,1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            #state size : ngf*4 x 8 x 8
            nn.ConvTranspose2d(ngf*4, ngf*2, 4,2,1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            #state size : ngf*2 x 16 x 16
            nn.ConvTranspose2d(ngf*2, ngf, 4,2,1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            #state size : ngf x 32 x 32
            nn.ConvTranspose2d(ngf,nc,4,2,1, bias=False),
            nn.Tanh()
            #state size : nc x 64 x 64

        )
    
    def forward(self, input):
        return self.main(input)
#create the generator

netG = Generator(ngpu).to(device)

if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

netG.apply(weights_init)
    
#print(netG)
   
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator,self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            #state: nc x 64 x 64
            nn.Conv2d(nc, ndf,4,2,1, bias = False),#3->64
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            #state: ndf x 32 x 32
            nn.Conv2d(ndf, ndf*2,4,2,1, bias=False),#64->128
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            #state: ndf*2 x 16 x 16
            nn.Conv2d(ndf*2, ndf*4,4,2,1, bias=False),#128->256
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            #state: ndf*4 x 8 x 8
            nn.Conv2d(ndf*4, ndf*8,4,2,1, bias=False),#256->512
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            #state: ndf*8 x 4 x 4
            nn.Conv2d(ndf*8, 1,kernel_size=4, stride=1, padding=0, bias=False),#512->1, 4 x 4 -> size 4의 kernel filter에 의해 1 x 1으로 바뀜
            nn.Sigmoid()

        )

    def forward(self, input):
        return self.main(input)

netD = Discriminator(ngpu).to(device)

if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))
netD.apply(weights_init)
#print(netD)

#Loss Function

criterion = nn.BCELoss() #Binary Cross Entropy, 두개의 class중 하나를 예측하는 task에 대한 cross entropy의 special case
fixed_noise = torch.randn(64, nz, 1, 1, device=device) 

real_label = 1
fake_label = 0

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999)) #beta: momentum
optimizerG = optim.Adam(netG.parameters(), lr = lr, betas=(beta1, 0.999))

img_list = []
G_losses = []
D_losses = []
iters = 0

log_dir = '/home/minchan/anaconda3/tensorboard/DCGAN/log'
summary = SummaryWriter(log_dir,'DCGAN')

num_batch = np.ceil(len(dataset) / batch_size)
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0,2,3,1)
fn_denorm = lambda x, mean, std: (x*std)+ mean

#D(x): real batch들의 discriminator의 평균 값, 1에 가까운 값으로 시작해서 0.5로 수렴할때 좋은 generator를 얻을 수 있음
#D(G(z)): fake batch들의 discriminator 평균 값, 0 근처 값에서 시작해서 0.5로 수렴할때 좋은 generator를 얻은 것
for epoch in range(1, num_epochs+1):
    for i, data in enumerate(dataloader,0): # start = 0, 몇번째 배치부터 뽑아올건지
        # Update D network: maximize log(D(x))+log(1-D(G(z)))
        
        # train with all-real batch
        # data[0]: (batch_size,channel,image_size,image_size), data[1]: mini-batch내의 각 image의 class들: 전부 0으로 세팅
        netD.zero_grad()
        # format batch
        real_gpu = data[0].to(device) #배치 이미지들을 gpu에 넣음(128,3,64,64)
        b_size = real_gpu.size(0) # 0차원 사이즈, 즉 batch_size
        label = torch.full((b_size,), real_label, dtype=torch.float, device = device)
        # batch_size만큼의 real_label(=1)로 full value를 가지는 tensor 생성, 1차원 벡터 [128]
        
        output = netD(real_gpu).view(-1) #128,100,1,1 -> 12800으로 flatten
        # calculate loss on all-real batch
        errD_real = criterion(output,label) #output: [12800], label:[100], output들은 전부 real image들이고 얼마나 많이 real로 판정했는지 계산
        # calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item() #real image들에 대해서 얼마나 잘 예측했는지 평균(1에 가까울 수록 real로 확신)
        summary.add_scalar('D(x)', D_x, epoch)
        
        # Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device = device)
        # Generate fake image batch with G
        fake = netG(noise)# fake: [128][3][64][64]
        
        label.fill_(fake_label)# 1로 꽉 차있던 label에 (fake_label=0)으로 채워줌 size=[128]
        output = netD(fake.detach()).view(-1) # detach: 기존 tensor를 gradient가 전파가 안되도록 복사
        
        errD_fake = criterion(output, label)
        # calculate the gradient, accumulated with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()# fake image들에 대한 예측 값 평균(0에 가까울수록 fake라고 인식을 잘 하는것, 1에 가까우면 real이라고 착각하는것)
        summary.add_scalar('D(G(x))', D_G_z1, epoch)
        errD = errD_real + errD_fake # log(D(x)) -> real image와 fake image에 대한 판별 loss
        summary.add_scalar('D_loss', errD.item(), epoch)
        
        optimizerD.step()
        
        # Update G network: minimize log(1-(D(G(z))) -> maximize log(D(G(z)))
        
        netG.zero_grad()
        label.fill_(real_label)# fake label 대신 real label -> gradient ascent(1-D(G(z)))
        
        output = netD(fake).view(-1)# fake= netG(noise), 1-D(G(x))
        errG = criterion(output, label)# log(1-D(G(x)))
        summary.add_scalar('G_loss', errG.item(), epoch)
        errG.backward()# backpropagation -> save gradients of each parameters for model 
        D_G_z2 = output.mean().item()# fake image를 real이라고 예측한 확률의 평균
        optimizerG.step()# gradient decent as saved gradients
        real = fn_tonumpy(fn_denorm(data[0], mean=0.5, std=0.5))
        fake = netG(fixed_noise).detach().cpu()
        fake = fn_tonumpy(fn_denorm(fake, mean=0.5, std=0.5))
        
        
        
        if i % 300 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  %(epoch, num_epochs, i, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            summary.add_image('real_image',real,num_batch*(epoch-1)+i, dataformats = 'NHWC')
            summary.add_image('fake_image',fake,num_batch*(epoch-1)+i, dataformats = 'NHWC')
        