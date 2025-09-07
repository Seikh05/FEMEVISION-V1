import torch
import torch.nn as nn
import torchvision.models as models 
import torchvision.transforms as T
import cv2
from PIL import Image

class GenderResNet50(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # remove fc
        self.fc = nn.Linear(2048, 1)
# ...existing code...

    def forward(self, x):
        feat = self.backbone(x).flatten(1)
        return self.fc(feat)

def load_gender_model(weights_path):
    model = GenderResNet50(pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

def infer_gender(model, crop_bgr):
    image = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    tensor_img = transform(image)
    if not isinstance(tensor_img, torch.Tensor):
        raise TypeError("Transform did not return a torch.Tensor. Check input image and transform.")
    x = tensor_img.unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits).item()
    return "Male" if prob > 0.5 else "Female"