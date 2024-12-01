import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import csv

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

# Image processing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
# MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


# Generator

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input is Z, going into a transposed convolution
            nn.ConvTranspose2d(latent_size, 128, 7, 1, 0, bias=False),  # This will create a 7x7 image
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # State size: 128 x 7 x 7
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),  # Upsample to 14x14
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # State size: 64 x 14 x 14
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),  # Upsample to 28x28
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
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),  # Downsample to 14x14
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(2, stride=2),  # Pool to 7x7
            # State size: 64 x 7 x 7
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),  # Downsample to 4x4
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: 128 x 4 x 4
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1, 1)  # Keep the output as [batch_size, 1]


# Initialize models
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Loss function & optimizers
criterion = nn.BCELoss()
gen_optimizer = optim.Adam(generator.parameters(), lr=learning_rate_G, betas=(beta1, beta2))
disc_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate_D, betas=(beta1, beta2))

losses_g = []
losses_d = []

# Training
for epoch in range(num_epochs):
    # Wrap the train_loader with tqdm for a progress bar
    for i, (images, _) in enumerate(train_loader):
        images = images.to(device)
        batch_size = images.size(0)

        # Train discriminator
        disc_optimizer.zero_grad()
        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)

        # Train discriminator with real images
        real_outputs = discriminator(images)
        real_loss = criterion(real_outputs, real_labels)

        # Train discriminator with fake images
        noise = torch.randn(batch_size, latent_size, 1, 1, device=device)
        fake_images = generator(noise)
        fake_outputs = discriminator(fake_images.detach())
        fake_loss = criterion(fake_outputs, fake_labels)

        disc_loss = real_loss + fake_loss
        disc_loss.backward()
        disc_optimizer.step()

        # Train generator
        gen_optimizer.zero_grad()
        noise = torch.randn(batch_size, latent_size, 1, 1, device=device)
        fake_images = generator(noise)
        outputs = discriminator(fake_images)
        gen_loss = criterion(outputs, real_labels)  # Generator tries to fool discriminator
        gen_loss.backward()
        gen_optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], '
                  f'Discriminator Loss: {disc_loss.item():.4f}, Generator Loss: {gen_loss.item():.4f}')

        losses_g.append(gen_loss.item())
        losses_d.append(disc_loss.item())

# Save generator model
torch.save(generator, './final_generator.pt')
torch.save(generator.state_dict(), './final_generator_weights.pt')

combined_data = list(zip(losses_g, losses_d))
with open('data_model_loss.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Losses_G', 'Losses_D'])
    writer.writerows(combined_data)

# Generate images
np.random.seed(504)
h = w = 28
num_gen = 25

z = np.random.normal(size=[num_gen, latent_size])
z = torch.randn(num_gen, latent_size).to(device)

generated_images = generator(z).cpu().detach().numpy()

# plot of generation
n = np.sqrt(num_gen).astype(np.int32)
I_generated = np.empty((h * n, w * n))
for i in range(n):
    for j in range(n):
        I_generated[i * h:(i + 1) * h, j * w:(j + 1) * w] = generated_images[i * n + j, :].reshape(28, 28)

plt.figure(figsize=(4, 4))
plt.axis("off")
plt.imshow(I_generated, cmap='gray')
plt.show()

with open('data_model_loss.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)
    losses_g = []
    losses_d = []
    for row in reader:
        losses_g.append(float(row[0]))
        losses_d.append(float(row[1]))

losses_g_epoch = [sum(losses_g[i:i + len(train_loader)]) / len(train_loader) for i in
                  range(0, len(losses_g), len(train_loader))]
losses_d_epoch = [sum(losses_d[i:i + len(train_loader)]) / len(train_loader) for i in
                  range(0, len(losses_d), len(train_loader))]

plt.figure()
plt.plot(range(1, len(losses_g_epoch) + 1), losses_g_epoch, label='Generator loss')
plt.plot(range(1, len(losses_d_epoch) + 1), losses_d_epoch, label='Discriminator loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('losses_plot.png')
plt.show()

