## Latent Image Similarity using ANNOY and PyTorch

## Overview

This project aims to implement image similarity using the ANNOY library and PyTorch. The major steps involved in this project include finalizing an embedding model, determining the number of dimensions and the metric for ANNOY, followed by deciding the number of trees ANNOY should build. 

## Model

For the image embedding model, we will consider three popular models: ResNet50, ResNet18, and VGG16. These models are widely used for image recognition tasks and have been pre-trained on large-scale datasets. Each model has its own architecture and characteristics. Here's a brief overview of the models:

- ResNet50: ResNet50 is a deep convolutional neural network with 50 layers. It has achieved excellent performance on various image recognition tasks.

- ResNet18: ResNet18 is a shallower version of ResNet50 with 18 layers. It offers a good balance between model complexity and performance.

- VGG16: VGG16 is another popular deep convolutional neural network. It has a total of 16 layers and has shown strong performance on image classification tasks.

Below is an example code snippet to use ResNet50 for image embedding:

```python
import torch
import torchvision.models as models

# Load the pre-trained ResNet50 model
weights = models.ResNet50_Weights.IMAGENET1K_V2
model = models.resnet50(weights=weights)

# removing the last layer
model.fc = nn.Identity()
model.eval()

# transform
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# Get output tensor
image = Image.open(image_path)
input_tensor = transform(image).unsqueeze(0)
output_tensor = model(input_tensor)
