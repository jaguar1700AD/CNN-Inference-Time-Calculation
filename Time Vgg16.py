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


dataloader = torch.utils.data.DataLoader(imagenet_data, batch_size=32, shuffle = False, num_workers=4)


# In[5]:


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class VGG(nn.Module):

    def __init__(self, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = make_layers(cfgs['D'], batch_norm = False)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# In[6]:


class modifiedVGG(nn.Module):

    def __init__(self, num_classes=1000, init_weights=True):
        super(modifiedVGG, self).__init__()
        self.features = make_layers(cfgs['D'], batch_norm = False)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

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
                
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x, np.array(time_arr)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# In[7]:


vgg = models.vgg16(pretrained=True)


# In[8]:


best_model_wts = copy.deepcopy(vgg.state_dict())
model = modifiedVGG()
device = torch.device('cuda')
model.to(device)
model.load_state_dict(best_model_wts)


# In[9]:


num_conv = 13

since = time.time()
model.eval()

with torch.no_grad():

    if isinstance(model, modifiedVGG):
        tot_times = np.zeros(num_conv)
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
            if done * len(inputs) >= 1000:
                break
            print(done * len(inputs), end = '\r')

            if isinstance(model, modifiedVGG):
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

    if isinstance(model, modifiedVGG):
        print(tot_times)
        #print(deb_times)
        #print(end_times)
        tot_times = tot_times / done
        print(tot_times)

    time_elapsed = time.time() - since
    print('Total time taken = {} seconds'.format(time_elapsed))

    end_to_end_time = end_to_end_time / (done)
    print('Avg End to End = {} ms'.format(end_to_end_time))


# In[10]:


print(prof.key_averages().table())


# In[ ]:


alexnet, vgg16, vgg19, lenet5, zfnet, resnet34

