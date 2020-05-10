import argparse
import json
import os
import numpy as np
import random
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from collections import OrderedDict

from PIL import Image

torch.manual_seed(47)

#-----------------------------------------------------------------------------------------------------------------------------------------------
def arg_parse():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--img_path", type=str, help="Testing image path")
    parser.add_argument("--checkpoint", type=str, help="Saved trained model checkpoint")
    parser.add_argument("--top_k", type=int, default=4, help="Top K most likely classes")
    parser.add_argument("--category_names", type=str, default="cat_to_name.json", help="JSPN object to map category label to name")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to train on")
    
    return parser.parse_args()

#-----------------------------------------------------------------------------------------------------------------------------------------------
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    model = checkpoint['model']   
    
    for param in model.parameters():
        param.requires_grad = False
            
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    
    optimizer = checkpoint['optimizer']
    learning_rate = checkpoint['learning_rate']
    epochs = checkpoint['epochs']
    model.class_to_idx = checkpoint['class_to_idx']    
    
    return model

#-----------------------------------------------------------------------------------------------------------------------------------------------
test_transform = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])])

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Process a PIL image for use in a PyTorch model
    converted_image = Image.open(image).convert("RGB")
    converted_image = test_transform(converted_image)
    return converted_image

#-----------------------------------------------------------------------------------------------------------------------------------------------
def predict(device, image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Implement the code to predict the class from an image file
    model.to(device)
    
    image = process_image(image_path)
    image.unsqueeze_(0)
    
    image = image.to(device)
    
    # Turn off the autograd
    with torch.no_grad():
        
        # Feed-fworwad
        logps = model(image)
                
        # Calculate probability
        ps = torch.exp(logps)
        top_ps, top_class = ps.topk(topk, dim=1)
        
        top_ps = np.array(top_ps[0])
        top_class = np.array(top_class[0])
    
    # Convert indices to actual category names
    index_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_class = [index_to_class[idx] for idx in top_class]
    
    return top_ps, top_class

#-----------------------------------------------------------------------------------------------------------------------------------------------
def main():
    args = arg_parse()
    
    # Label mapping
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    # Load the checkpoint
    model = load_checkpoint(args.checkpoint)
    
    # Class prediction
    device = args.device
    image_path = args.img_path
    top_k = args.top_k
    
    probs, classes = predict(device, image_path, model, top_k)
    names = [cat_to_name[str(idx)] for idx in classes]
    
    # Print
    print("\nTop {} most likely classes are...".format(top_k))
    print("Names: ", names)
    print("Probabilities: ", probs)
    print()
    
#-----------------------------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()