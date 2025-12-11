# dcgan_fashion_mnist.py
# PyTorch DCGAN for Fashion-MNIST (28x28)

import os, torch, torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, 

# Hyperparameters

latent_dim = 100
batch_size = 128
epochs = 20
lr = 2e-4
beta1 = 0.5
beta2 = 0.999
sample_dir = "samples_fashion_mnist"
os.makedirs(sample_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Data

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # to [-1, 1]
])

dataset = datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=transform
)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

# Models

class Generator(nn.Module):
    """
    Input:  (N, 100, 1, 1)
    Output: (N, 1, 28, 28)
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # (N, 100, 1, 1) -> (N, 128, 7, 7)
            nn.ConvTranspose2d(latent_dim, 128, kernel_size=7, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # (N, 128, 7, 7) -> (N, 64, 14, 14)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # (N, 64, 14, 14) -> (N, 1, 28, 28)
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    """
    Input:  (N, 1, 28, 28)
    Output: (N, 1) logits
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # (N, 1, 28, 28) -> (N, 64, 14, 14)
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # (N, 64, 14, 14) -> (N, 128, 7, 7)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # (N, 128, 7, 7) -> (N, 1, 1, 1)
            nn.Conv2d(128, 1, kernel_size=7, stride=1, padding=0, bias=False)
        )

    def forward(self, x):
        logits = self.net(x).view(-1, 1)
        return logits

G = Generator().to(device)
D = Discriminator().to(device)

# Loss & Optimizers

criterion = nn.BCEWithLogitsLoss()
optG = torch.optim.Adam(G.parameters(), lr=lr, betas=(beta1, beta2))
optD = torch.optim.Adam(D.parameters(), lr=lr, betas=(beta1, beta2))

# Fixed noise for monitoring G's progress
fixed_z = torch.randn(64, latent_dim, 1, 1, device=device)

# Training

def real_fake_labels(n, real=True):
    # Optional label smoothing for real labels (helps stability a bit)
    if real:
        return torch.full((n, 1), 0.9, device=device)  # smooth real labels to 0.9
    else:
        return torch.zeros((n, 1), device=device)

for epoch in range(1, epochs + 1):
    G.train(); D.train()
    for real_imgs, _ in loader:
        real_imgs = real_imgs.to(device)
        n = real_imgs.size(0)

        # ---------------- D step ----------------
        # Real
        D.zero_grad(set_to_none=True)
        labels_real = real_fake_labels(n, real=True)
        logits_real = D(real_imgs)
        loss_real = criterion(logits_real, labels_real)

        # Fake
        z = torch.randn(n, latent_dim, 1, 1, device=device)
        fake_imgs = G(z).detach()
        labels_fake = real_fake_labels(n, real=False)
        logits_fake = D(fake_imgs)
        loss_fake = criterion(logits_fake, labels_fake)

        loss_D = loss_real + loss_fake
        loss_D.backward()
        optD.step()

        # ---------------- G step ----------------
        G.zero_grad(set_to_none=True)
        z = torch.randn(n, latent_dim, 1, 1, device=device)
        gen_imgs = G(z)
        # try to fool D -> want D(gen_imgs) to classify as real (1)
        logits_gen = D(gen_imgs)
        labels_gen = real_fake_labels(n, real=True)
        loss_G = criterion(logits_gen, labels_gen)
        loss_G.backward()
        optG.step()

    # Save samples
    G.eval()
    with torch.no_grad():
        samples = G(fixed_z)
        # bring from [-1,1] to [0,1] for saving
        utils.save_image(samples, os.path.join(sample_dir, f"epoch_{epoch:03d}.png"),
                         nrow=8, normalize=True, value_range=(-1, 1))
    print(f"Epoch {epoch:03d}/{epochs} | D: {loss_D.item():.4f} | G: {loss_G.item():.4f}")

print("Training complete. Sample images saved in:", sample_dir)
