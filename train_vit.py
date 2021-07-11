import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transforms
import random
import os
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from model import ViT
from utils.XrayDataset import ChestXRayDataSet

seed = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
    ])

transform_val = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor()
    ])

transform_test = transforms.Compose([
    transforms.RandomResizedCrop(224), 
    transforms.ToTensor(),
    ])

def read_data(path = ".\chest_xray", mode = 'train'):
    data_path = os.path.join(path,mode)
    if mode == 'train':
        dataset_obj = ChestXRayDataSet(data_path, transform_train)
        dataset = dataset_obj()
    elif mode == 'val':
        dataset_obj = ChestXRayDataSet(data_path, transform_val)
        dataset = dataset_obj()
    else:
        dataset_obj = ChestXRayDataSet(data_path, transform_test)
        dataset = dataset_obj()
    return dataset

def train(model, train_loader, valid_loader, criterion, optimizer, scheduler, epochs: int = 100):
    model.to(device)
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        for data, label in tqdm(train_loader):
            model.train()
            optimizer.zero_grad()
            #Load data into cuda
            data = data.to(device)
            label = label.to(device)
            #Pass data to model
            output = model(data)
            loss = criterion(output, label)
            #Optimizing
            loss.backward()
            optimizer.step()
            #Calculate Accuracy
            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(train_loader)
            if valid_loader is not None:
                epoch_val_accuracy = 0
                epoch_val_loss = 0
                for data, label in valid_loader:
                    model.eval()
                    #Load val_data into cuda
                    data = data.to(device)
                    label = label.to(device)
                    #Pass val_data to model
                    val_output = model(data)
                    val_loss = criterion(val_output, label)
                    #Calculate Validation Accuracy
                    acc = (val_output.argmax(dim=1) == label).float().mean()
                    epoch_val_accuracy += acc / len(valid_loader)
                    epoch_val_loss += val_loss / len(valid_loader)
        if valid_loader is not None:
            print(
                f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
                )
        else:
            print(
                f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f}\n"
                )

if __name__ == "__main__":
    #read in data
    default_data_path = ".\chest_xray"
    train_data = read_data(default_data_path, 'train')
    val_data = read_data(default_data_path, 'val')
    test_data = read_data(default_data_path, 'test')
    #data loaders
    train_loader = DataLoader(dataset=train_data, batch_size = 64, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size = 64, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size = 16)
    #define model
    vision_transformer = ViT(img_size=224, patch_size=16, num_class=2, d_model=576,n_head=6,n_layers=6,d_mlp=1024)
    #configs
    epochs = 100
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(vision_transformer.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.0001)
    #train
    train(model=vision_transformer, train_loader=train_loader, valid_loader=val_loader, 
        criterion=criterion, optimizer=optimizer, scheduler=scheduler, epochs=epochs)
    