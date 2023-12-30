import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse

from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image

from config import *
import ic_model as ic

def train(args):
    print("Loading the data...")
    train_dataset, valid_dataset, test_dataset, trainloader, validloader, testloader = load_data(args)
    print("Loading the data - DONE")

    print("Preparing the model...")
    model = ic.initialize_model('alexnet', args.hidden_units, True)
    model.to(device)
    model.class_to_idx = train_dataset.class_to_idx
    print("Preparing the model - DONE")

    print("Starting training...")
    params_to_update = ic.get_params_to_update(model, True)
    optimizer = optim.SGD(params_to_update, args.lr) 
    model = ic.train_model(model, optimizer, args.epochs, trainloader, validloader)
    print("End Training - DONE")
    
    ic.save_model(model, args.save_dir)
    print("Model is saved in path \ "+args.save_dir + MODEL_FILE);

def load_data(args):
    train_dir = args.data_dir + '/train'
    valid_dir = args.data_dir + '/valid'
    test_dir = args.data_dir + '/test'

    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])

    # TODO: Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_dataset =datasets.ImageFolder(valid_dir, transform = valid_transforms) 
    test_dataset =datasets.ImageFolder(test_dir, transform = test_transforms) 


    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_dataset, valid_dataset, test_dataset, trainloader, validloader, testloader

def test(args):
    train_dataset, valid_dataset, test_dataset, trainloader, validloader, testloader = load_data(args)
    print("Loading the checkpoint model")
    model = ic.load_model(args.checkpoint_path, 'alexnet', args.hidden_units)

    print("Testing model")
    ic.evaluate(model, testloader)

    print("Testing Finish")
    
def main():
    parser = argparse.ArgumentParser(description='Flowers Classifcation Trainer')
    parser.add_argument('--gpu', type=bool, default=False, help='Enable/Disable GPU')
    parser.add_argument('--arch', type=str, default='alexnet', help='architecture [available: alexnet, googlenet, vgg16]')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='hidden units for  fc layer')
    parser.add_argument('--epochs', type=int, default=15, help='number of epochs')
    parser.add_argument('--data_dir', type=str, default='flower_data', help='dataset directory')
    parser.add_argument('--save_dir' , type=str, default='./assets/', help='checkpoint directory path')
    parser.add_argument('--test' , type=bool, default= False, help='test loader')
    parser.add_argument('--checkpoint_path', type=str, default='assets/' + MODEL_FILE, help='path to en existance  checkpoint')

    args = parser.parse_args()

    print(args)
    if args.test:
        test(args)
    else:
        train(args)
        print("training Finished\n")
if __name__ == "__main__":
    main()
