import argparse
import json
import os
import random
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from collections import OrderedDict

from PIL import Image



#-----------------------------------------------------------------------------------------------------------------------------------------------
def arg_parse():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_dir", type=str, default = "flowers/")
    parser.add_argument('--device',default = 'cuda',help = 'choose the device GPU',type = str)
    parser.add_argument('--save_dir',default = 'checkpoint3.pth',help = 'location to save',type =str)
    parser.add_argument('--arch',default = 'vgg16',type = str )
    parser.add_argument('--learning_rate',default = '0.001',help = 'learning rate with default value 0.001',type = float)
    parser.add_argument('--epochs',default = 4,help = 'list the number of epochs',type = int)
    parser.add_argument('--path_to_image', type = str, default = '47/image_04966.jpg',help = 'image path to test')
    parser.add_argument('--checkpoint_dir', type = str, default = 'image_checkpoint.pth', help = 'path to save checkpoints')
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json',help = 'json file with category names')
    parser.add_argument("--hidden_units", type=int, default=2048, help="Number of hidden units")
    parser.add_argument('--top_k', type = int, default = 5, help = 'Top K classes')
    
    return parser.parse_args()
    
#-----------------------------------------------------------------------------------------------------------------------------------------------
def main():
    args = arg_parse()
    
    train_dir = args.data_dir + '/train'
    valid_dir = args.data_dir + '/valid'
    
    # Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)

    # Label mapping
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
     # Device CPU or gpu
    device = args.device
    
    # architectures list
    if (args.arch == 'vgg16' ): 
        model = models.vgg16(pretrained=True)
    else:
        print("please enter vgg16 ")
    
    for param in model.parameters():
        param.requires_grad = False    
            
    # define classifier
    classifier = nn.Sequential(OrderedDict([
        ('dense1',nn.Linear(25088,4096)),
        ('relu1',nn.ReLU()),
        ('dense2',nn.Linear(4096,args.hidden_units)),
        ('relu2',nn.ReLU()),
        ('dense3',nn.Linear(args.hidden_units,102)),  
        ('output',nn.LogSoftmax(dim=1))
        ]))
    
        
    model.classifier = classifier
    criterion = nn.NLLLoss()
    learning_rate = args.learning_rate
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    model.to(device)
    
    print("Training started.")
    
    # Train model
    epochs = args.epochs
    steps = 0
    running_loss = 0
    print_every = 10

    for epoch in range(epochs):
            
        # Training loop
        for images, labels in trainloader:
            steps += 1
        
            # Move images & labels to GPU if available
            images, labels = images.to(device), labels.to(device)
        
            # Set gradients to zero
            optimizer.zero_grad()
        
            # Feedforward
            logps = model(images)
            loss = criterion(logps, labels)
        
            # Backpropagation
            loss.backward()
        
            # Gradient descent
            optimizer.step()
        
            running_loss += loss.item()
        
            if steps % print_every == 0:
            
                # Turn on evaluation, inference mode; turn off dropout
                model.eval()
                valid_loss = 0
                accuracy = 0
            
                # Turn off autograd
                with torch.no_grad():

                    # Validation loop
                    for images, labels in validloader:
                
                        # Move images & labels to GPU if available
                        images, labels = images.to(device), labels.to(device)
        
                        logps = model(images)
                        batch_loss = criterion(logps, labels)
                        valid_loss += batch_loss.item()
                
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_ps, top_class = ps.topk(1, dim=1)
                        equality = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equality.type(torch.FloatTensor)).item()
            
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validloader):.3f}.. ") 
            
                running_loss = 0
            
                # Set model back to training mode
                model.train()   

    # Save the checkpoint
    model.class_to_idx = train_data.class_to_idx

    checkpoint = {'input_size': 25088,
                  'output_size': 102,
                  'model': model,
                  'classifier': model.classifier,
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer,
                  'learning_rate': learning_rate,
                  'epochs': epochs,
                  'class_to_idx': model.class_to_idx
                  }

    torch.save(checkpoint, args.save_dir)
    
    print("\nTraining process is completed!")
    print("Checkpoint is saved at: {}".format(args.save_dir))
    print()

#-----------------------------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()