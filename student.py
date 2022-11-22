#!/usr/bin/env python3
## This is homework 2 done by group members:
## Jiaxuan Li


"""
Several models are attempted in this project, which are VGG19, AlexNet and ResNet18. By experimenting, it is discovered that the accuracy of AlexNet is around 65%, and VGG19 is around 70%. 
Amazingly, ResNet18 can reach around 81% accuracy because of its deeper network and down-sampling. Thus, ResNet18 is decided to use as the final model, which has four basic blocks, and each 
block contains two convolutional layers and one shortcut connection. Each convolution layer has applied a 2-dimensional  batch normalization and a Relu activation function, which could prevent 
the problem of exploding and vanishing gradient problems. Furthermore, each block has increased the channel by enhancing the number of filters. The fully connected layer has applied an adaptive 
average pooling, which converts the channel to 512. This approach reduces the size of the model. The project also applies a Dropout function to reduce the situation of overfitting. 

-----------------Structure of ResNet---------------------
            convolutional layer 7*7 s = 2 p = 3         -
            BatchNorm, Relu, Maxpool 3*3 s=2 p =1      Downsampling 1*1 s =2
                                                        -
   ----------------------Layer 1 ----------------------
            convolutional layer 3*3 s = 1 p = 1         -
                      BatchNorm, Relu,                  Downsampling 1*1 s =2
            convolutional layer 3*3 s = 1 p = 1         -
                      BatchNorm, Relu,                  -
   ----------------------Layer 2 ----------------------
            convolutional layer 3*3 s = 1 p = 1         -
                      BatchNorm                         Downsampling 1*1 s =2
            convolutional layer 3*3 s = 1 p = 1         - 
                      BatchNorm,                        -
   ----------------------Layer 3 ----------------------
            convolutional layer 3*3 s = 1 p = 1         -
                      BatchNorm                         Downsampling 1*1 s =2
            convolutional layer 3*3 s = 1 p = 1         -
                      BatchNorm                         -
   ----------------------Layer 4 ----------------------
            convolutional layer 3*3 s = 1 p = 1         -
                      BatchNorm                         Downsampling 1*1 s =2
            convolutional layer 3*3 s = 1 p = 1         -
                      BatchNorm                         -
    ---------------------------------------------------
                Relu, adpavgpool, Dropout
                      Linear


However, the overfitting problem is still serious because the model is over-complex and deep. Thus, pre-processing is quite important to enhance the increasing rate of test accuracy. 
By researching, data augmentation could efficiently reduce this problem through using transformations during training. Firstly, on both training data and validation data, the Resize function 
is used to fit the ResNet model, and ToTenser is applied to convert the images into Torch tensors and scale pixel values to numbers between 0.0 and 1.0. Additionally, since the images will be
 more blurred after being resized, RandomAutocontrast and RandomAdjustSharpness are used to increase the image qualities. Finally, after observing 8 different classes of images, RandomRotation 
 and RandomHorizontalFlip are chosen to increase the image diversity and quantities. These applications improve the accuracy of ResNet18 from 72% to 81%.

The output of the layer has 8 lables, so CrossEntopy() is very suitable for this model whose output is a probability between 0 and 1 The formula is :
    loss = 1/n * sum(-y(log(p)) + (1-y) log(1-p))
Optimizer is using adam optimizer. Adam optimizer is an extension to SGD which improve the performance on the problem with sparse gradients. The optimizer uses the 'L2' norm as weight decay and
set the initial learning rate to 0.001. A larger learning rate will reduce the loss very quick but it is not always accurate. Since Schedule is needed to reduce the learning rate, which uses StepLR 
and set the step size as 30 and gamma = 0.5. The formula is shown below:
    Lr = initial Lr * gamma ^ (epoch//step size)
Thus, the learning rate will reduce along the epoch increased. This approach will improve the efficiency of training because the beginning loss is not stable. Therefore, a constant learning 
rate will take a very long time for the first 50 epochs. Weight initialization is used to define the initial values for the parameters in the ResNet model. 
For the fully connected layer, xavier_normal_  is applied to maintain a constant variance for input and output. Additionally, kaiming_normal_ is used for the convolutional layer and constant_ for 
the batchnorm2d. The accuracy increases from 81% to 83%.

By using full data, the accuracy has imporve to 90% in the final test.

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.conv import Conv2d
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

"""
   Answer to Question:

Briefly describe how your program works, and explain any design and training
decisions you made along the way.
"""

############################################################################
######     Specify transform(s) to be applied to the input images     ######
############################################################################
def transform(mode):
    """
    Called when loading the data. Visit this URL for more information:
    https://pytorch.org/vision/stable/transforms.html
    You may specify different transforms for training and testing
    """
    # image preprocessing
    if mode == 'train':
        return transforms.Compose(
            [
            transforms.Resize(224),
            transforms.RandomRotation(degrees=(-30, 30)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomPerspective(p=0.2),
            transforms.RandomAutocontrast(),
            transforms.RandomAdjustSharpness(sharpness_factor=2),
            transforms.ToTensor(),
            ]
        )
    elif mode == 'test':
        return transforms.Compose(
            [
            transforms.Resize(224),
            transforms.ToTensor(),
            ]
        )

############################################################################
######   Define the Module to process the images and produce labels   ######
######      We are using ResNet34 to do the image classification      ######
############################################################################


### ResNet18 ####

class Network(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer_input = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride= 2,padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )


        self.layer_1 = nn.Sequential(
             
            nn.Conv2d(64,64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        #Down Sample
        self.downsample2 = nn.Sequential(
            nn.Conv2d(64,128,kernel_size=1, stride= 2),
            nn.BatchNorm2d(128),
        )

        self.layer_2 = nn.Sequential(
            #Down Sample 
            nn.Conv2d(64,128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
        )
        #Down Sample
        self.downsample3 = nn.Sequential(
            nn.Conv2d(128,256,kernel_size=1, stride= 2),
            nn.BatchNorm2d(256),
        )

        self.layer_3 = nn.Sequential(
            #Down Sample 
            nn.Conv2d(128,256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
        )

        #Down Sample
        self.downsample4 = nn.Sequential(
            nn.Conv2d(256,512,kernel_size=1, stride= 2),
            nn.BatchNorm2d(512),
        )

        self.layer_4 = nn.Sequential(
            #Down Sample 
            nn.Conv2d(256,512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
        )
        self.relu = nn.ReLU(inplace=True)
        self.adpavgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p = 0.5)
        self.linear = nn.Linear(512, 8)

    def forward(self, input):
        layer_input = self.layer_input(input)
        layer1 = self.layer_1(layer_input)

        ds2 = self.downsample2(layer1)
        layer2 = self.layer_2(layer1)
        layer2 = layer2+ds2
        layer2 = self.relu(layer2)

        ds3 = self.downsample3(layer2)
        layer3 = self.layer_3(layer2)
        layer3 = layer3+ds3
        layer3 = self.relu(layer3)

        ds4 = self.downsample4(layer3)
        layer4 = self.layer_4(layer3)
        layer4 = layer4+ds4
        layer4 = self.relu(layer4)


        adpavgpool = self.adpavgpool(layer4)
        adpavgpool = torch.flatten(adpavgpool, 1)
        adpavgpool = self.dropout(adpavgpool)
        ret = self.linear(adpavgpool)


        return ret

net = Network()
    
############################################################################
######      Specify the optimizer and loss function                   ######
############################################################################
lr = 0.001

optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
#optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum=0.9,weight_decay=1e-5)
loss_func = nn.CrossEntropyLoss()


############################################################################
######  Custom weight initialization and lr scheduling are optional   ######
############################################################################

# Normally, the default weight initialization and fixed learing rate
# should work fine. But, we have made it possible for you to define
# your own custom weight initialization and lr scheduler, if you wish.
def weights_init(m):
    if isinstance(m, nn.Linear):
        ## Reference from TORCH.NN.INIT official document
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        ## Reference from TORCH.NN.INIT official document
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        ## Reference from TORCH.NN.INIT official document
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

scheduler =  torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
############################################################################
#######              Metaparameters and training options              ######
############################################################################
dataset = "./data"
train_val_split = 1 # 1 for last training, you can change this to 0.8 to default.
batch_size = 25
epochs = 210
