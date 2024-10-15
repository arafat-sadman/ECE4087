import torch
from PIL import Image
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt
import torch
model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=3, out_channels=1, init_features=32, pretrained=True)

# Load pretrained ESRGAN model (you might need to download the model weights)
#model_path = 'path/to/pretrained_esrgan.pth'
#model = torch.load(model_path)
model.eval()

# Function to load and preprocess image
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),  # Resize to fit the model input
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalization
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Function to postprocess and save the image
def postprocess_image(tensor):
    tensor = tensor.squeeze(0).detach().numpy()
    tensor = tensor.transpose(1, 2, 0)
    tensor = (tensor * 255).clip(0, 255).astype("uint8")
    return Image.fromarray(tensor)

# Load and enhance an image
input_image_path = 'check.jpg'
input_image = preprocess_image(input_image_path)

with torch.no_grad():
    output_image = model(input_image)

# Convert tensor back to image and save
output_image = postprocess_image(output_image)
output_image.save('path/to/save/enhanced_image.png')

# Display input and output images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(Image.open(input_image_path))
plt.title('Input Image')
plt.subplot(1, 2, 2)
plt.imshow(output_image)
plt.title('Enhanced Image')
plt.show()
