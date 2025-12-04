import torch
from PIL import Image
import requests
from torchvision import transforms, models

# Load a resnet18 model using the local torchvision API (compatible with installed versions)
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).eval()

# Download human-readable labels for ImageNet.
response = requests.get("https://git.io/JJkYN")
labels = [l for l in response.text.split("\n") if l]

# Standard ImageNet preprocessing
preprocess = transforms.Compose([
       transforms.Resize(256),
       transforms.CenterCrop(224),
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict(inp: Image.Image):
       if inp is None:
              return {}
       img_t = preprocess(inp).unsqueeze(0)
       with torch.no_grad():
              out = model(img_t)[0]
              probs = torch.nn.functional.softmax(out, dim=0)
              topk = torch.topk(probs, 3)
       # return only the top-k labels to keep output small
       return {labels[int(idx)]: float(probs[int(idx)]) for idx in topk.indices}

import gradio as gr

gr.Interface(fn=predict,
                      inputs=gr.Image(type="pil"),
                      outputs=gr.Label(num_top_classes=3)).launch()

