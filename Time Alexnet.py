#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import  torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy


# In[2]:


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# In[3]:


imagenet_data = torchvision.datasets.CIFAR10('D:\\datasets\\', download = True, transform = transform)


# In[4]:


dataloader = torch.utils.data.DataLoader(imagenet_data, batch_size=256, shuffle = False, num_workers=6)


# In[5]:


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# In[6]:


class modifiedAlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(modifiedAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        time_arr = []
        deb_arr = []
        global device
        
        for layer in self.features:
            if isinstance(layer, nn.Conv2d):
                
                if device == torch.device('cpu'):
                    start = time.time()
                    x = layer(x)
                    end = time.time()
                    time_arr.append((end - start) * 1000)
                else:
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    x = layer(x)
                    end.record()
                    torch.cuda.synchronize()
                    time_arr.append(start.elapsed_time(end))
                    
            else:
                #start = time.time()
                x = layer(x)
                #end = time.time()
                #deb_arr.append(end - start)
        
        #start = time.time()
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        #end = time.time()
        #end_time = end - start
        
        #return x, np.array(time_arr), np.array(deb_arr), end_time
        return x, np.array(time_arr)


# In[7]:


alexnet = models.alexnet(pretrained=True)


# In[8]:


best_model_wts = copy.deepcopy(alexnet.state_dict())
model = modifiedAlexNet()
device = torch.device('cuda')
model.to(device)
model.load_state_dict(best_model_wts)


# In[9]:


#from torch.hub import load_state_dict_from_url
#model_urls = {'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'}
#state_dict = load_state_dict_from_url(model_urls['alexnet'])
#model.load_state_dict(state_dict)


# In[13]:


since = time.time()
model.eval()

with torch.no_grad():

    if isinstance(model, modifiedAlexNet):
        tot_times = np.zeros(5)
        #deb_times = np.zeros(8)
        #end_times = 0
    corrects = 0
    done = 0

    end_to_end_time = 0.0

    with torch.autograd.profiler.profile(use_cuda = (device == torch.device('cuda'))) as prof:
        for inputs, labels in dataloader:

            start_main = torch.cuda.Event(enable_timing=True)
            end_main = torch.cuda.Event(enable_timing=True)
            start_main.record()

            inputs = inputs.to(device)
            labels = labels.to(device)
            done += 1
            if done * len(inputs) >= 10000:
                break
            print(done * len(inputs), end = '\r')

            if isinstance(model, modifiedAlexNet):
                #outputs, exec_times, hid_times, end_time = model(inputs)
                outputs, exec_times = model(inputs)
                tot_times += exec_times
                #deb_times += hid_times
                #end_times += end_time
            else:
                outputs = model(inputs)

            end_main.record()
            torch.cuda.synchronize()
            end_to_end_time += start_main.elapsed_time(end_main)

            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)

    acc = corrects.double() / (done * len(inputs))
    print('Acc: {:.4f}'.format(acc))

    if isinstance(model, modifiedAlexNet):
        print(tot_times)
        #print(deb_times)
        #print(end_times)
        tot_times = tot_times / done
        print(tot_times)

    time_elapsed = time.time() - since
    print('Total time taken = {} seconds'.format(time_elapsed))

    end_to_end_time = end_to_end_time / (done)
    print('Avg End to End = {} ms'.format(end_to_end_time))


# In[14]:


print(prof.key_averages().table())


# In[ ]:


alexnet, vgg16, vgg19, lenet5, zfnet, resnet34

