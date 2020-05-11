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


# -----------------------------------------------------------------------------------------------------------------------------------------------
def arg_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--img_path", type=str, help="Testing image path")
    parser.add_argument("--checkpoint", type=str,
                        help="Saved trained model checkpoint")
    parser.add_argument("--top_k", type=int, default=4,
                        help="Top K most likely classes")
    parser.add_argument("--category_names", type=str, default="cat_to_name.json",
                        help="JSPN object to map category label to name")
    parser.add_argument("--gpu", type=str, default="cuda",
                        choices=["cuda", "cpu"], help="Device to train on")

    return parser.parse_args()

# -----------------------------------------------------------------------------------------------------------------------------------------------


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

# -----------------------------------------------------------------------------------------------------------------------------------------------


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # Process a PIL image for use in a PyTorch model
    y_to_x_ratio = image.size[1] / image.size[0]
    trans = transforms.ToTensor()
    x = 256
    y = int(y_to_x_ratio * x)
    image = image.resize((x, y))

    # cropping from center
    center_width = image.size[0] / 2
    center_height = image.size[1] / 2

    cropped_image = image.crop(
        (
            center_width - 112,
            center_height - 112,
            center_width + 112,
            center_height + 112
        )
    )
    # Normalize image
    np_image = np.array(cropped_image)
    np_image = np.array(np_image)/255.0

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    image = (np_image - mean) / std
    image = image.transpose((2, 0, 1))
    return torch.from_numpy(image)

# -----------------------------------------------------------------------------------------------------------------------------------------------


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

# -----------------------------------------------------------------------------------------------------------------------------------------------


def main():
    args = arg_parse()

    # Label mapping
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    # Load the checkpoint
    model = load_checkpoint(args.checkpoint)

    # Check if gpu is available
    args.gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = args.gpu
    image_path = args.img_path
    top_k = args.top_k

    # user spicify the JSON file
    category_names = args_values.category_names

    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)

    probs, classes = predict(device, image_path, model, top_k)
    names = [cat_to_name[str(idx)] for idx in classes]

    # Print
    print("\nTop {} most likely classes are...".format(top_k))
    print("Names: ", names)
    print("Probabilities: ", probs)
    print()


# -----------------------------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
