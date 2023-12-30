import torch
from torchvision import models
from torch import nn
from collections import OrderedDict

from config import *

def initialize_model(model_name, hidden_units, feature_extract=True):

    model = None
    
    if model_name == "alexnet":
        model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.classifier[1].in_features
        #model.classifier[6] = nn.Linear(num_ftrs,num_classes)
        model.classifier = build_network(num_ftrs, hidden_units)
    
    elif model_name == "googlenet":
        model = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, hidden_units)

    elif model_name == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.classifier[6].in_features
        model.classifier = build_network(num_ftrs, hidden_units)
    
    else:
        print("Invalid model name")
        exit()
    
    return model

def build_network(num_ftrs, hidden_units):
    return nn.Sequential(OrderedDict([
        ('dropout1', nn.Dropout(p=0.5)),
        ('fc1', nn.Linear(num_ftrs, hidden_units)),
        ('relu', nn.ReLU()),
        ('dropout2', nn.Dropout(p=0.5)),
        ('fc2', nn.Linear(hidden_units, hidden_units)),  
        ('relu2', nn.ReLU()),
        ('dropout3', nn.Dropout(p=0.5)),
        ('fc3', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1)),
    ]))

def train_model(model, optimizer, epochs, trainloader, validloader):
    print("Training has started... \n")
    model.batch_size=trainloader.batch_size
    model.optim_state_dict=optimizer.state_dict()

    criterion = nn.NLLLoss()
    best_accuracy = 0

    for epoch in range(epochs):
        running_loss = 0
        for inputs, labels in trainloader:
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        else:
            valid_loss = 0
            accuracy = 0
            model.eval()
            
            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    loss = criterion(logps, labels)
                    
                    valid_loss += loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                    if best_accuracy<accuracy:
                        best_accuracy=accuracy
                        model.accuracy=best_accuracy
                        model.epoch = epoch
                    
                print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/len(trainloader):.3f}.. "
                    f"Valid loss: {valid_loss/len(validloader):.3f}.. "
                    f"Valid accuracy: {accuracy/len(validloader):.3f}")
        model.train()
    print("\nTraining is completed!") 

      
    return model

def evaluate(model, loader):
    valid_loss = 0;
    accuracy = 0;
    model.eval()
    criterion = nn.NLLLoss()
    
    with torch.no_grad(): 
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss = criterion(output,labels)
            valid_loss += loss.item()
            
            output_Exp = torch.exp(output)
            top_p, top_c = output_Exp.topk(1,dim=1)
            equals= top_c ==labels.view(*top_c.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        print(f"test  loss: {valid_loss/len(loader):.3f}.. "
              f"test  accuracy: {accuracy/len(loader):.3f}")
    model.train()

def save_model(model, dir):
    checkpoint = {
        'state_dict': model.state_dict(),
        'epoch': model.epoch,
        'batch_size': model.batch_size,
        'optimizer_state' : model.optim_state_dict,
        'class_to_idx': model.class_to_idx,
        'output_size' : 102,
        'input_size' : (224,224,3),
        'accuracy':model.accuracy
    }

    torch.save(checkpoint, dir + MODEL_FILE)

def load_model(path, model_name, hidden_units):
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)

    model = initialize_model(model_name, hidden_units, True)
    model.class_to_idx=checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    model.epoch=checkpoint['epoch']
    model.optimizer_state=checkpoint['optimizer_state']
    
    return model


def set_parameter_requires_grad(model, feature_extract):
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False

def get_params_to_update(model, feature_extract):
    params_to_update = []
    if feature_extract:
        for param in model.parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
    else:
        params_to_update = model.parameters()
    return params_to_update