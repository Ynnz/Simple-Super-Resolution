import os
import sys
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# Add model path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "models"))
from srcnn import SRCNN


# Load the model
model = SRCNN()
model.load_state_dict(torch.load("checkpoints/srcnn_cifar10_subset.pth"))
model.eval()

# Load some CIFAR-10 test images
transform = transforms.ToTensor()
testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
hr_img, _ = testset[0]  # Just pick one image

# Create a low-res version
down_up = transforms.Compose([
    transforms.Resize(16),
    transforms.Resize(32)
])
lr_img = down_up(hr_img)

# Run the model
with torch.no_grad():
    input_tensor = lr_img.unsqueeze(0)  # add batch dimension
    output = model(input_tensor).squeeze(0)  # remove batch dimension

# Helper to visualize
def imshow(tensor, title):
    img = tensor.permute(1, 2, 0).numpy()
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')

# Plot low-res, super-res, and ground truth
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
imshow(lr_img, "Low-Res Input")
plt.subplot(1, 3, 2)
imshow(output, "Super-Resolved Output")
plt.subplot(1, 3, 3)
imshow(hr_img, "High-Res Ground Truth")
plt.tight_layout()
plt.show()