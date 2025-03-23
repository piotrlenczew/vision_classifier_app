import torch
import torchvision.models as models
from PIL import Image

# Load a pre-trained model (e.g., ResNet18)
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.to("cuda")  # Move to CPU
model.eval()  # Set to evaluation mode

# transformations required by the model https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html#torchvision.models.ResNet18_Weights 
transform = models.ResNet18_Weights.IMAGENET1K_V1.transforms

# Get class labels from the model's metadata
class_labels = models.ResNet18_Weights.IMAGENET1K_V1.meta["categories"]

def classify_image(image):
    """Classify an image and return the predicted class."""
    img = Image.open(image).convert("RGB")
    img = transform(img).unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        output = model(img)
        predicted_class = torch.argmax(output, dim=1).item()
    
    return predicted_class  # Modify to return a class label if needed
