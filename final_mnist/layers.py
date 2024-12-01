import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm  # Import the tqdm module

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
latent_size = 100
batch_size = 128
num_epochs = 200
learning_rate_G = 0.0002
learning_rate_D = 0.0002
beta1 = 0.5
beta2 = 0.999
label_smooth = 0.1
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input is Z, going into a transposed convolution
            nn.ConvTranspose2d(latent_size, 128, 7, 1, 0, bias=False), # This will create a 7x7 image
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # State size: 128 x 7 x 7
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False), # Upsample to 14x14
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # State size: 64 x 14 x 14
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False), # Upsample to 28x28
            nn.Tanh()
            # Final state size: 1 x 28 x 28
        )

    def forward(self, input):
        input = input.view(input.size(0), 100, 1, 1)
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input 28x28 image
            nn.Conv2d(1, 64, 4, 2, 1, bias=False), # Downsample to 14x14
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(2, stride=2), # Pool to 7x7
            # State size: 64 x 7 x 7
            nn.Conv2d(64, 128, 3, 2, 1, bias=False), # Downsample to 4x4
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: 128 x 4 x 4
            nn.Flatten(),
            nn.Linear(128*4*4, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
      return self.main(input).view(-1, 1)  # Keep the output as [batch_size, 1]
