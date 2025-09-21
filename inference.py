from models.ViT import ViT
from PIL import Image
import torch
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CIFAR-10 classes
classes = ['plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']



#image preprocessing
transform = transforms.Compose([
    transforms.Resize((32, 32)),  
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616]
    )
])

model = ViT(
    image_size = 32,
    patch_size=4,
    num_classes=10,
    emb_dims=128,
    depth=4,
    num_heads=4,
    dropout=0.1,
    mlp_ratio=2
).to(device)

# Load weights
model.load_state_dict(torch.load("best_vit_cifar10.pth", map_location=device))
model.eval()


# Inference on a single image
def predict_image(img_path):
    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # add batch dimension

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return classes[predicted.item()]
    
if __name__ == "__main__":
    img_path = "sample.jpg"  
    prediction = predict_image(img_path)
    print(f"Predicted class: {prediction}")