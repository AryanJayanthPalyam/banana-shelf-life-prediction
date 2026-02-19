import io
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import models, transforms

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MultiOutputResNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        for p in self.base.parameters():
            p.requires_grad = False

        in_features = self.base.fc.in_features
        self.base.fc = nn.Identity()

        self.reg_head1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.cls_head1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6)
        )

    def forward(self, x):
        feat = self.base(x)
        out_reg = self.reg_head1(feat)
        out_cls = self.cls_head1(feat)
        return out_reg, out_cls



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiOutputResNet()
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()


transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
])

variety_map_inv = {
    0: 'yellaki',
    1: 'robusta',
    2: 'Redbanana',
    3: 'poovan',
    4: 'karpooravalli',
    5: 'nendran'
}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_days, pred_cls = model(tensor)

    pred_days = float(pred_days.item())
    pred_cls = torch.argmax(pred_cls, dim=1).item()

    return {
        "predicted_variety": variety_map_inv[pred_cls],
        "predicted_days_until_death": round(pred_days, 2)
    }
