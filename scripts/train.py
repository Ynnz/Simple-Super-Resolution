import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Add model path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "models"))
from srcnn import SRCNN


class SuperResolutionDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.down_up = transforms.Compose([
            transforms.Resize(16),
            transforms.Resize(32)
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        hr_img, _ = self.dataset[idx]
        lr_img = self.down_up(hr_img)
        return lr_img, hr_img


def main():
    # Load CIFAR-10 and use only a subset for speed
    transform = transforms.ToTensor()
    cifar10 = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    subset = Subset(cifar10, range(1000))  # Use only 1000 images due to limited computing power
    dataset = SuperResolutionDataset(subset)

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    print(f"Loaded subset with {len(dataset)} samples")

    model = SRCNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        print(f"Epoch {epoch+1}/{num_epochs}")

        for batch_idx, (lr, hr) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            preds = model(lr)
            loss = criterion(preds, hr)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"✅ Epoch {epoch+1} completed, Avg Loss: {avg_loss:.6f}")

    # Save model
    os.makedirs("checkpoints", exist_ok=True)
    save_path = "checkpoints/srcnn_cifar10_subset.pth"
    torch.save(model.state_dict(), save_path)
    print(f"✅ Training complete. Model saved to {save_path}")


if __name__ == "__main__":
    main()