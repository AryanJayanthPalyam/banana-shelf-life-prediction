import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image

VARIETY_MAP = {'yellaki': 0, 'robusta': 1, 'Redbanana': 2}
VARIETY_LIST = list(VARIETY_MAP.keys())

# Model class (match your training model architecture)
class BananaCNNMulti(torch.nn.Module):
    def __init__(self, n_varieties=3, embed_dim=8):
        super().__init__()
        self.resnet = models.resnet18(weights=None)
        self.resnet.fc = torch.nn.Identity()
        self.variety_embed = torch.nn.Embedding(n_varieties, embed_dim)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(512 + embed_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )
    def forward(self, x_img, x_variety):
        img_feat = self.resnet(x_img)
        var_feat = self.variety_embed(x_variety)
        x = torch.cat([img_feat, var_feat], dim=1)
        return self.fc(x).squeeze(-1)

@st.cache_resource
def load_model():
    model = BananaCNNMulti(n_varieties=len(VARIETY_LIST))
    model.load_state_dict(torch.load('banana_cnn_trained.pth', map_location='cpu'))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

st.title("Banana Ripeness Prediction")

uploaded_file = st.file_uploader("Upload a banana image", type=["jpg", "jpeg", "png"])
variety = st.selectbox("Select Banana Variety", VARIETY_LIST)

if uploaded_file is not None and variety:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    input_tensor = transform(image).unsqueeze(0)
    variety_tensor = torch.tensor([VARIETY_MAP[variety]], dtype=torch.long)
    
    with torch.no_grad():
        prediction = model(input_tensor, variety_tensor)
        pred_days = prediction.item()
        
    if pred_days <= 0:
        st.error("Predicted: ROTTEN")
    else:
        st.success(f"Predicted days until overripeness: {pred_days:.2f} days")
