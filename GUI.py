import os
import random
from typing import List
import pandas as pd
from PIL import Image, ImageFilter
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import torch.nn.functional as F
import streamlit as st

torch.backends.cudnn.benchmark = True

# --------------------------
# 1) Charset and utils
# --------------------------
CHARSET = list(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    ".,-;:!?()[]{}'\"/\\@#&%+*=<> "
)
BLANK_IDX = len(CHARSET)

def encode_text(s: str) -> List[int]:
    ids = []
    for ch in s:
        try:
            ids.append(CHARSET.index(ch))
        except ValueError:
            continue
    return ids

def greedy_decode(log_probs):
    preds = log_probs.argmax(dim=2)
    results = []
    for seq in preds:
        prev = BLANK_IDX
        text = []
        for idx in seq:
            idx = idx.item()
            if idx != prev and idx != BLANK_IDX:
                text.append(CHARSET[idx])
            prev = idx
        results.append("".join(text))
    return results

def beam_decode(log_probs, beam_width=10):
    results = []
    B, T, C = log_probs.shape
    for b in range(B):
        beams = [("", 0.0, BLANK_IDX)]
        for t in range(T):
            new_beams = []
            probs = log_probs[b, t]
            topk_probs, topk_idx = probs.topk(beam_width)
            for seq, score, last in beams:
                for p, idx in zip(topk_probs, topk_idx):
                    p = p.item()
                    idx = idx.item()
                    new_score = score + p
                    if idx == BLANK_IDX:
                        new_beams.append((seq, new_score, last))
                    else:
                        if idx != last:
                            char = CHARSET[idx] if idx < len(CHARSET) else ""
                            new_beams.append((seq + char, new_score, idx))
                        else:
                            new_beams.append((seq, new_score, last))
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:beam_width]
        best_seq = max(beams, key=lambda x: x[1] / max(1, len(x[0])))[0]
        results.append(best_seq)
    return results

# --------------------------
# 2) Dataset
# --------------------------
class HandwritingWordDataset(Dataset):
    def __init__(self, csv_file: str, img_root: str = None, transform=None,img_size=(64,256), train=True):
        df = pd.read_csv(csv_file)
        df = df.dropna(subset=["FILENAME", "IDENTITY"])
        self.df = df.reset_index(drop=True)
        self.img_root = img_root
        self.transform = transform
        self.img_size = img_size
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row['FILENAME']
        if self.img_root:
            path = os.path.join(self.img_root, path)
        img = Image.open(path).convert("L")
        img = img.resize(self.img_size, Image.BILINEAR)
        if self.transform:
            img = self.transform(img)
        text = str(row['IDENTITY'])
        label = encode_text(text)
        return {
            "image": img,
            "text": text,
            "label": torch.tensor(label, dtype=torch.long),
            "label_length": len(label)
        }

def collate_fn(batch):
    images = [b['image'] for b in batch]
    texts = [b['text'] for b in batch]
    labels = [b['label'] for b in batch]
    label_lengths = [b['label_length'] for b in batch]
    images = torch.stack(images, dim=0)
    labels_cat = torch.cat(labels).to(torch.long) if labels else torch.tensor([], dtype=torch.long)
    return {
        'images': images,
        'texts': texts,
        'labels': labels_cat,
        'label_lengths': torch.tensor(label_lengths, dtype=torch.long)
    }

# --------------------------
# 3) Model
# --------------------------
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((4, 1))
        )
        self.rnn = nn.LSTM(512, 256, bidirectional=True, num_layers=2, dropout=0.2)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        features = self.cnn(x)
        features = features.mean(2)
        features = features.permute(2,0,1)
        out,_ = self.rnn(features)
        out = self.fc(out)
        return F.log_softmax(out, dim=2)

# --------------------------
# 4) Transform
# --------------------------
def default_transforms(img_size=(64,256), train=True):
    h, w = img_size
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((h,w)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

# --------------------------
# 5) Prediction
# --------------------------
def predict_img(img: Image.Image, checkpoint_path: str, device='cpu', img_size=(64,256)):
    device = torch.device(device)
    num_classes = len(CHARSET)+1
    model = CRNN(num_classes=num_classes).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    transform = default_transforms(img_size, train=False)
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        log_probs = model(img_tensor)
        log_probs = log_probs.permute(1,0,2)
        pred_text = beam_decode(log_probs, beam_width=10)[0]
        probs = log_probs.exp()
        conf = probs.max(dim=-1)[0].mean().item()
    return pred_text, conf

# --------------------------
# 6) Streamlit GUI
# --------------------------
st.title("Handwriting OCR Recognition")

uploaded_file = st.file_uploader("Upload a handwriting image", type=["png","jpg","jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("L")
    st.image(img, caption="Uploaded Image", width=700)

    if st.button("Predict"):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        checkpoint_path = "ocr_epoch_last.pth"

        if not os.path.exists(checkpoint_path):
            st.error(f"Checkpoint not found: {checkpoint_path}")
        else:
            pred_text, conf = predict_img(img, checkpoint_path, device=device)
            st.subheader("Predicted Text:")
            st.text(pred_text)
            st.subheader("Confidence:")
            st.text(f"{conf*100:.2f}%")
