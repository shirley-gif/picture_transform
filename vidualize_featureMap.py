'''
展示VGG19模型的特征图
# Visualize feature maps from VGG19 model   

# This script loads an image, processes it, and visualizes the feature maps
# extracted from a specific layer of the VGG19 model.   
# It uses PyTorch and torchvision for model loading and image processing.
# Visualize feature maps from VGG19 model

'''

from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt

# Load and preprocess image
img_path = "data/images/neural-style/dancing.jpg"
img = Image.open(img_path).convert("RGB")

# Resize and normalize to fit VGG input
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # convert to [0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                         std=[0.229, 0.224, 0.225])
])

img_tensor = transform(img).unsqueeze(0)  # add batch dimension

# Load pretrained VGG19 model
vgg = models.vgg19(pretrained=True).features.eval()

# Get feature map from a specific layer (e.g., conv4_2 is at index 21 in VGG19)
with torch.no_grad():
    for i, layer in enumerate(vgg):
        img_tensor = layer(img_tensor)
        if i == 21:  # conv4_2
            feature_maps = img_tensor.squeeze(0)  # remove batch dim
            break

# Visualize the first 6 feature maps
fig, axes = plt.subplots(3, 5 , figsize=(12, 8))
for i, ax in enumerate(axes.flat):
    fmap = feature_maps[i].cpu()
    ax.imshow(fmap, cmap='viridis')
    ax.axis("off")
    ax.set_title(f"Channel {i}")

plt.tight_layout()
plt.show()
