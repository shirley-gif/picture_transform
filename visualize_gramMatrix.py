'''
展示 VGG19 网络前几层的 Gram Matrix
# Visualize the Gram Matrix of VGG19's early layers 

# This script loads an image, processes it, and visualizes the Gram Matrix
# extracted from the first few layers of the VGG19 model.
# It uses PyTorch and torchvision for model loading and image processing.


'''

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# 图像路径：换成你自己的
IMAGE_PATH = "Pc_Banner_new-scaled.jpg"

# 1. 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
image = transform(Image.open(IMAGE_PATH).convert("RGB")).unsqueeze(0)

# 2. 加载 VGG19 网络的前几层（比如 relu1_2）
vgg = models.vgg19(pretrained=True).features[:4].eval()

# 3. 获取特征图
with torch.no_grad():
    features = vgg(image)

# 4. Gram Matrix 计算
def gram_matrix(input_tensor):
    b, c, h, w = input_tensor.size()
    features = input_tensor.view(b * c, h * w)
    G = torch.mm(features, features.t())
    return G / (b * c * h * w)

gram = gram_matrix(features).squeeze().cpu().numpy()

# 5. 可视化
plt.figure(figsize=(6, 5))
sns.heatmap(gram, cmap="coolwarm", cbar=False)
plt.title("Gram Matrix (Style Representation)")
plt.tight_layout()
plt.show()