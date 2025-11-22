import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    """3D GAN Generator for voxel-based shape generation"""

    def __init__(self, latent_dim=200, output_dim=32, leak_value=0.2, bias=False):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        padd = (0, 0, 0)
        if output_dim == 32:
            padd = (1, 1, 1)

        self.main = nn.Sequential(
            # Input: latent_dim x 1 x 1 x 1
            nn.ConvTranspose3d(latent_dim, 512, kernel_size=4, stride=2, padding=(1, 1, 1), bias=bias),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            # State: 512 x 2 x 2 x 2

            nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=(1, 1, 1), bias=bias),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            # State: 256 x 4 x 4 x 4

            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=(1, 1, 1), bias=bias),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            # State: 128 x 8 x 8 x 8

            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=(1, 1, 1), bias=bias),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            # State: 64 x 16 x 16 x 16

            nn.ConvTranspose3d(64, 1, kernel_size=4, stride=2, padding=padd, bias=bias),
            nn.Sigmoid()
            # Output: 1 x 32 x 32 x 32
        )

    def forward(self, x):
        # Reshape input to (batch, latent_dim, 1, 1, 1)
        x = x.view(-1, self.latent_dim, 1, 1, 1)
        output = self.main(x)
        return output


class Discriminator(nn.Module):
    """3D GAN Discriminator for voxel-based shape discrimination"""

    def __init__(self, input_dim=32, leak_value=0.2, bias=False):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim

        padd = (0, 0, 0)
        if input_dim == 32:
            padd = (1, 1, 1)

        self.main = nn.Sequential(
            # Input: 1 x 32 x 32 x 32
            nn.Conv3d(1, 64, kernel_size=4, stride=2, padding=padd, bias=bias),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(leak_value, inplace=True),
            # State: 64 x 16 x 16 x 16

            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=(1, 1, 1), bias=bias),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(leak_value, inplace=True),
            # State: 128 x 8 x 8 x 8

            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=(1, 1, 1), bias=bias),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(leak_value, inplace=True),
            # State: 256 x 4 x 4 x 4

            nn.Conv3d(256, 512, kernel_size=4, stride=2, padding=(1, 1, 1), bias=bias),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(leak_value, inplace=True),
            # State: 512 x 2 x 2 x 2

            nn.Conv3d(512, 1, kernel_size=4, stride=2, padding=(1, 1, 1), bias=bias),
            nn.Sigmoid()
            # Output: 1 x 1 x 1 x 1
        )

    def forward(self, x):
        output = self.main(x)
        return output.view(-1, 1)


class DiffusionModel(nn.Module):
    """Simple 3D Diffusion Model for shape generation (placeholder for future implementation)"""

    def __init__(self, resolution=32):
        super(DiffusionModel, self).__init__()
        self.resolution = resolution
        # This is a placeholder - full diffusion model would be more complex
        print("Diffusion model placeholder - using GAN for now")

    def forward(self, x, t):
        return x


def weights_init(m):
    """Initialize network weights"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
