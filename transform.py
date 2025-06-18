'''

运行风格迁移算法，生成新的图像

1. 加载内容图（你的主图）

2. 加载风格图（参考图，如古典油画）

3. 提取两者的特征（通过 VGG19 网络）

4. 优化内容图的像素，使其风格趋近于风格图

'''

from PIL import Image
import torchvision.transforms as transforms
import torch
import torchvision.models as models
import torch.nn as nn

# 预处理函数
def image_loader(image_name, imsize=512):
    loader = transforms.Compose([
        transforms.Resize(imsize),
        transforms.ToTensor()])
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(torch.float)

# 加载内容图和风格图
content_img = image_loader("your_content.jpg")
style_img = image_loader("your_style.jpg")

assert content_img.size() == style_img.size(), "尺寸必须一致"

# 使用 VGG19 特征提取器
cnn = models.vgg19(pretrained=True).features.eval()

# 内容和风格损失模块（省略细节）
# 引用官方教程实现：https://pytorch.org/tutorials/advanced/neural_style_tutorial.html

# 运行风格迁移函数（封装后）
output = run_style_transfer(cnn, content_img, style_img)

# 保存图片
unloader = transforms.ToPILImage()
image = output.cpu().clone().squeeze(0)
image = unloader(image)
image.save("stylized_output.jpg")
