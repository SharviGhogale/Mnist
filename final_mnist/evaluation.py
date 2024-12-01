# Imports
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from layers import Generator

     #Import Generator Class from Training

# Define the device for the training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model():
    '''
    This function is used to load the saved generator model using PyTorch.
    '''
    model = torch.load('final_generator.pt', map_location=torch.device('cpu'))
    model.to(device)
    model.eval()

    return model

def generate_images(model, num_images):
    '''
    Take the model as input and generate a specified number of images.
    '''
    z_dim = 100
    images = []

    for _ in range(num_images):
        z = torch.randn(1, z_dim, device=device)
        generated_image = model(z).detach().cpu().numpy().reshape(28, 28)
        images.append(generated_image)

    return images

def plot_images(images, grid_size):
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10), tight_layout=True)

    for i, ax in enumerate(axes.flatten()):
        ax.imshow(images[i], cmap='gray')
        ax.axis("off")
    #save the plot
    plt.savefig('generated_images.png')
    plt.show()


if __name__ == "__main__":
    # Instantiate the generator model and load the saved state dictionary
    model = load_model()

    # Generate 25 new images
    num_images = 25
    images = generate_images(model, num_images)

    # Show the generated images in a 5x5 grid
    grid_size = 5
    plot_images(images, grid_size)
