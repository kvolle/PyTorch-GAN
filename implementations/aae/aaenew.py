import argparse
import os
import numpy as np
from matplotlib import pyplot as plt
import math
import itertools

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from dataexploration import NPZLoader
from matplotlib import pyplot as plt
#from data import load_dataset

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00005, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=32, help="dimensionality of the latent code")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False
TRAIN = True

def reparameterization(mu, logvar):
    std = torch.exp(logvar / 2)
    sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), opt.latent_dim))))
    z = sampled_z * std + mu
    return z


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.mu = nn.Linear(512, opt.latent_dim)
        self.logvar = nn.Linear(512, opt.latent_dim)

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        x = self.model(img_flat)
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = reparameterization(mu, logvar)
        return z


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, int(np.prod(img_shape))),
            nn.Tanh(),
        )

    def forward(self, z):
        img_flat = self.model(z)
        img = img_flat.view(img_flat.shape[0], *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        validity = self.model(z)
        return validity


# Use binary cross-entropy loss
adversarial_loss = torch.nn.BCELoss()
pixelwise_loss = torch.nn.L1Loss()

training_data = NPZLoader("./data/train/")
train_dataloader=DataLoader(training_data, batch_size=opt.batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_data = NPZLoader("./data/test/")
test_dataloader=DataLoader(test_data, batch_size=opt.batch_size, shuffle=True, num_workers=4, pin_memory=True)
#img = training_data.__getitem__(990)
#print(img[0].shape)

#img_np = img[0].numpy()
#plt.figure()
#plt.imshow(img_np)
#plt.show()

# Initialize generator and discriminator
encoder = Encoder()
decoder = Decoder()
discriminator = Discriminator()

if cuda:
    print("Using cuda")
    encoder.cuda()
    decoder.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    pixelwise_loss.cuda()

# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(encoder.parameters(), decoder.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def sample_image(n_row, batches_done):
    #load images from test dataset 
    """Saves a grid of generated digits"""
    # Sample noise
    z = Variable(Tensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    gen_imgs = decoder(z)
    #test = test_dataloader
    real_imgs = Variable(imgs.type(Tensor))
    #print(real_imgs.type)
    latent = encoder(real_imgs)
    #dec = decoder(latent) 
    #latent_matrix = np.concatenate((latent_matrix,dec.cpu().numpy().squeeze()))
    #latent_matrix = np.concatenate((latent_matrix,latent.cpu().numpy().squeeze()))
    decoded = decoder(latent)
    latent = Variable(decoded.type(Tensor))
    save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)
    #latent captures dyanmics of the generatvie process: periodicity

def grid_image(n_row, batches_done):
    #load images from test dataset 
    # Sample noise

    #tensor by 24,3,64,64 (Two of them) => start with 0 by 3 by 64 by 64
    decoded_collection = torch.empty([0,3,64,64]).cuda()
    real_collection = torch.empty([0,3,64,64]).cuda()
    #print(decoded_collection.shape)
    print(real_collection)
    with torch.no_grad():
        for i, (imgs, _) in enumerate(test_dataloader):
            #encoder.to('cuda')
            if imgs.shape[0] == 8:
                real_imgs = Variable(imgs.type(Tensor))
                #print(real_imgs.shape)
                real_collection = torch.cat((real_collection,real_imgs),0) 
                #print("real collection loop: ",real_collection.shape)
                #decode = torch.cat((real_collection,real_imgs),0)
                #print("real collection loop: ",real_collection.shape)


                
                encoded_imgs = encoder(real_imgs)
                #imgs_1 = Variable(Tensor(encoded_imgs)
                decoded_imgs = Variable(Tensor(decoder(encoded_imgs)))
                #print("Decoded Images: ", decoded_imgs.shape)
                decoded_collection = torch.cat((decoded_collection,decoded_imgs),0)
                #print("Decoded Collection: ",decoded_collection.shape)
                #could be collection
                #concatenate the two tensors; 
                      



         #imgs = Tensor(decoded_imgs)
         #if decoded_imgs.shape[0] == 8:
             
         #print(decoded_imgs.shape)
    #z = Variable(Tensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    #encoded_imgs = encoder(imgs)
    #gen_imgs = decoder(z)
    #torch.
    #print(dist[i][mask[i] == 0])
    #figure out shape of images matrix by 24 by 3 by 64 by 64
    
    
    #print("real collection : ", real_collection.shape)
    #print("decoded collection : ", decoded_collection.shape)
    save_image(real_collection, "images/real.png", nrow=n_row, normalize=True)
    save_image(decoded_collection, "images/decoded.png", nrow=n_row, normalize=True)


#test data concatenate latent space 
# ----------
#  Training
# ----------
if TRAIN:
    plt.figure("Gen Loss")
    gen = plt.gca()
    plt.figure("Dis Loss")
    dis = plt.gca()
    for epoch in range(opt.n_epochs):
        for i, (imgs, _) in enumerate(train_dataloader):
            # net = Net()
            # Adversarial ground truths
            valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(Tensor))
            # -----------------
            #  Train Generator
            # -----------------
            


            

            encoded_imgs = encoder(real_imgs)
            decoded_imgs = decoder(encoded_imgs)
            # Loss measures generator's ability to fool the discriminator
            g_loss = 0.001 * adversarial_loss(discriminator(encoded_imgs), valid) + 0.999 * pixelwise_loss(
                decoded_imgs, real_imgs
            )

            g_loss.backward()

            optimizer_G.step()
            #print(real_imgs.data.max())
            #print(decoded_imgs.data.max())
            #for i in range(3):
            #    ax_1[i].hist(real_imgs.data.cpu().numpy()[0,i,:,:].reshape(-1))
            #    ax_2[i].hist(decoded_imgs.data.cpu().numpy()[0,i,:,:].reshape(-1))
            #plt.show()
            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Sample noise as discriminator ground truth
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(z), valid)
            fake_loss = adversarial_loss(discriminator(encoded_imgs.detach()), fake)
            d_loss = 0.5 * (real_loss + fake_loss)

            d_loss.backward()
            optimizer_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(train_dataloader), d_loss.item(), g_loss.item())
            )
            if i == 59:
                gen.plot(epoch, g_loss.item(),'bo')
                dis.plot(epoch, d_loss.item(),'bo')
            
            batches_done = epoch * len(train_dataloader) + i
            if batches_done % opt.sample_interval == 0:
                sample_image(n_row=10, batches_done=batches_done)
                grid_image(n_row=10, batches_done=batches_done)

    torch.save(encoder, 'encoder.pth')
    torch.save(decoder,'decoder.pth')
    torch.save(discriminator,'discriminator.pth')
else: 
    encoder = torch.load('encoder.pth')
    decoder = torch.load('decoder.pth')
    discriminator = torch.load('discriminator.pth')
latent_matrix = np.empty((0,opt.latent_dim))
with torch.no_grad():
    for i, (imgs, _) in enumerate(test_dataloader):
        #test = test_dataloader
        real_imgs = Variable(imgs.type(Tensor))
        print(real_imgs.type)
        latent = encoder(real_imgs)
        latent_matrix = np.concatenate((latent_matrix,latent.cpu().numpy().squeeze()))
        decoded = decoder(latent)
        latent = Variable(decoded.type(Tensor))
        
    #latent_matrix = np.empty((0,opt.latent_dim))
#with torch.no_grad():
fig_1, ax_1 = plt.subplots(3,3)
fig_2, ax_2 = plt.subplots(3,3)
ax_1 = ax_1.ravel()
ax_2 = ax_2.ravel()
for i in range(9):
        #ax_1[i].hist(latent_matrix[i])
        #ax_2[i].plot(latent_matrix[i])
    ax_1[i].hist(latent_matrix[:,i])
    ax_2[i].plot(latent_matrix[:,i])
plt.show()
#save them to grid and the 6 * 4 grid of images with real and reconstruced by decoder images
# loop over test dataset => feed everything throguh the encoder and decoder
#concatentate torhc tensors in torch 
#if imgs.shape[0] == 8:

"""
def grid_image(n_row, batches_done):
    #load images from test dataset 
    # Sample noise
    for i, (imgs, _) in enumerate(test_dataloader):
         encoded_imgs = encoder(imgs)
         decoded_imgs = decoder(encoded_imgs)
         print(decoded_imgs.shape)
    #z = Variable(Tensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    #encoded_imgs = encoder(imgs)
    #gen_imgs = decoder(z)
    save_image(decoded_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)
    """