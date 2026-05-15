import json
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
LATENT_DIM = 10
EPOCHS = 10
LR = 1e-3


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


def loss_function(recon_x, x, mu, logvar):
    x = x.view(x.size(0), -1)

    recon_loss = nn.functional.binary_cross_entropy(
        recon_x, x, reduction="sum"
    )

    kl_div = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    return recon_loss + kl_div


def train():
    transform = transforms.ToTensor()

    dataset = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model = VAE(latent_dim=LATENT_DIM).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    logs = []

    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0

        for images, _ in loader:
            images = images.to(DEVICE)

            optimizer.zero_grad()

            recon, mu, logvar = model(images)

            loss = loss_function(recon, images, mu, logvar)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader.dataset)

        logs.append({
            "epoch": epoch + 1,
            "train_loss": round(avg_loss, 4)
        })

        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "vae_mnist.pth")

    with open("baseline_metrics.json", "w") as f:
        json.dump({
            "dataset": "MNIST",
            "epochs": EPOCHS,
            "latent_dim": LATENT_DIM,
            "final_train_loss": logs[-1]["train_loss"]
        }, f, indent=4)

    pd_logs = __import__("pandas").DataFrame(logs)
    pd_logs.to_csv("training_logs.csv", index=False)

    print("Training completed successfully.")


if __name__ == "__main__":
    train()
