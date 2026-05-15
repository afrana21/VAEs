import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch import nn


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=512, latent_dim=10):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )

        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = x.view(x.size(0), -1)

        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)

        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)

        return recon, mu, logvar


model = VAE().to(DEVICE)
model.load_state_dict(torch.load("vae_mnist.pth", map_location=DEVICE))
model.eval()

transform = transforms.ToTensor()

dataset = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)

image, label = dataset[0]

with torch.no_grad():
    recon, _, _ = model(image.unsqueeze(0).to(DEVICE))

recon = recon.view(28, 28).cpu().numpy()

plt.figure(figsize=(6, 3))

plt.subplot(1, 2, 1)
plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"Original ({label})")

plt.subplot(1, 2, 2)
plt.imshow(recon, cmap="gray")
plt.title("Reconstructed")

plt.tight_layout()
plt.show()
