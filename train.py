import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from tqdm import tqdm
import argparse

from models.gan_3d import Generator, Discriminator, weights_init
from utils.shape_utils import create_sample_shape

class ShapeNetDataset(Dataset):
    """Dataset loader for ShapeNet voxel data"""

    def __init__(self, data_dir, resolution=32):
        self.data_dir = data_dir
        self.resolution = resolution
        self.samples = []

        # Load dataset files (assuming .npy format)
        if os.path.exists(data_dir):
            for filename in os.listdir(data_dir):
                if filename.endswith('.npy'):
                    self.samples.append(os.path.join(data_dir, filename))

        # If no data found, create synthetic samples for demo
        if len(self.samples) == 0:
            print("No ShapeNet data found. Creating synthetic samples...")
            self.use_synthetic = True
            self.samples = list(range(100))  # 100 synthetic samples
        else:
            self.use_synthetic = False

        print(f"Dataset initialized with {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.use_synthetic:
            # Generate synthetic shapes
            shape_types = ['cube', 'sphere', 'pyramid']
            shape_type = shape_types[idx % len(shape_types)]
            voxel = create_sample_shape(shape_type, self.resolution)
            voxel = voxel.astype(np.float32)
        else:
            # Load real ShapeNet data
            voxel = np.load(self.samples[idx])
            voxel = voxel.astype(np.float32)

        # Add channel dimension
        voxel = voxel[np.newaxis, ...]

        return torch.from_numpy(voxel)


def train_gan(args):
    """Train 3D GAN on ShapeNet dataset"""

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create dataset and dataloader
    dataset = ShapeNetDataset(args.data_dir, args.resolution)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    # Initialize models
    generator = Generator(
        latent_dim=args.latent_dim,
        output_dim=args.resolution
    ).to(device)

    discriminator = Discriminator(
        input_dim=args.resolution
    ).to(device)

    # Initialize weights
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    # Optimizers
    optimizer_G = optim.Adam(
        generator.parameters(),
        lr=args.lr,
        betas=(args.beta1, 0.999)
    )

    optimizer_D = optim.Adam(
        discriminator.parameters(),
        lr=args.lr,
        betas=(args.beta1, 0.999)
    )

    # Loss function
    criterion = nn.BCELoss()

    # Training loop
    print("Starting training...")
    print("=" * 60)

    for epoch in range(args.num_epochs):
        epoch_g_loss = 0
        epoch_d_loss = 0

        with tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.num_epochs}') as pbar:
            for i, real_shapes in enumerate(pbar):
                batch_size = real_shapes.size(0)
                real_shapes = real_shapes.to(device)

                # Labels
                real_labels = torch.ones(batch_size, 1).to(device)
                fake_labels = torch.zeros(batch_size, 1).to(device)

                # ---------------------
                # Train Discriminator
                # ---------------------
                optimizer_D.zero_grad()

                # Real shapes
                real_output = discriminator(real_shapes)
                d_loss_real = criterion(real_output, real_labels)

                # Fake shapes
                z = torch.randn(batch_size, args.latent_dim).to(device)
                fake_shapes = generator(z)
                fake_output = discriminator(fake_shapes.detach())
                d_loss_fake = criterion(fake_output, fake_labels)

                # Total discriminator loss
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                optimizer_D.step()

                # -----------------
                # Train Generator
                # -----------------
                optimizer_G.zero_grad()

                # Generate fake shapes
                z = torch.randn(batch_size, args.latent_dim).to(device)
                fake_shapes = generator(z)
                fake_output = discriminator(fake_shapes)

                # Generator loss
                g_loss = criterion(fake_output, real_labels)
                g_loss.backward()
                optimizer_G.step()

                # Update progress bar
                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()

                pbar.set_postfix({
                    'G_Loss': f'{g_loss.item():.4f}',
                    'D_Loss': f'{d_loss.item():.4f}'
                })

        # Print epoch statistics
        avg_g_loss = epoch_g_loss / len(dataloader)
        avg_d_loss = epoch_d_loss / len(dataloader)

        print(f"Epoch [{epoch+1}/{args.num_epochs}] - "
              f"G_Loss: {avg_g_loss:.4f}, D_Loss: {avg_d_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            torch.save(
                generator.state_dict(),
                os.path.join(args.checkpoint_dir, f'generator_epoch_{epoch+1}.pth')
            )
            torch.save(
                discriminator.state_dict(),
                os.path.join(args.checkpoint_dir, f'discriminator_epoch_{epoch+1}.pth')
            )
            print(f"Saved checkpoint at epoch {epoch+1}")

    # Save final model
    torch.save(generator.state_dict(), os.path.join(args.checkpoint_dir, 'generator.pth'))
    torch.save(discriminator.state_dict(), os.path.join(args.checkpoint_dir, 'discriminator.pth'))
    print("Training completed!")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train 3D GAN on ShapeNet')

    parser.add_argument('--data_dir', type=str, default='data/shapenet',
                        help='Path to ShapeNet dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='Beta1 for Adam optimizer')
    parser.add_argument('--latent_dim', type=int, default=200,
                        help='Dimension of latent space')
    parser.add_argument('--resolution', type=int, default=32,
                        help='Voxel resolution')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')

    args = parser.parse_args()

    train_gan(args)
