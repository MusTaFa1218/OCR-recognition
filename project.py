import os
import random
from typing import List
import pandas as pd
from PIL import Image, ImageFilter
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from torchvision.models import resnet18, ResNet18_Weights 
import torch.nn.functional as F 
torch.backends.cudnn.benchmark = True

CHARSET = list(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    ".,-;:!?()[]{}'\"/\\@#&%+*=<> "
)
BLANK_IDX = len(CHARSET)

# --------------------------
# 2) Utilities: encode / decode for CTC
# --------------------------
def encode_text(s: str) -> List[int]:
    """Encode a string into list of integer indices according to CHARSET.
       Unknown chars are ignored (or could be mapped to a special token)."""
    ids = []
    for ch in s:
        try:
            ids.append(CHARSET.index(ch))
        except ValueError:

            continue
    return ids

def greedy_decode(log_probs):
    """
    log_probs: Tensor (B, T, C)
    returns: list[str]
    """
    preds = log_probs.argmax(dim=2)  # (B, T)
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
    """
    log_probs: Tensor (B, T, C)  — log-softmax outputs
    returns: list[str]
    """
    results = []

    B, T, C = log_probs.shape

    for b in range(B):
        beams = [("", 0.0, BLANK_IDX)]

        for t in range(T):
            new_beams = []

            probs = log_probs[b, t]  # (C,)

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


train_csv = "/kaggle/input/handwriting-recognition/written_name_train_v2.csv"
val_csv   = "/kaggle/input/handwriting-recognition/written_name_validation_v2.csv"
test_csv  = "/kaggle/input/handwriting-recognition/written_name_test_v2.csv"
train_root = "/kaggle/input/handwriting-recognition/train_v2/train"
val_root   = "/kaggle/input/handwriting-recognition/validation_v2/validation"
test_root  = "/kaggle/input/handwriting-recognition/test_v2/test"

batch_size = 64
epochs = 150
lr = 0.0001 
device = 'cuda' if torch.cuda.is_available() else 'cpu' 
img_size = (64, 256)
checkpoint_dir = '/kaggle/working/checkpoints_2'  
checkpoint_name ='ocr_epoch_last.pth'

class HandwritingWordDataset(Dataset):
    def __init__(self, csv_file: str, img_root: str = None, transform=None,img_size=img_size , train=True):

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

        # Resize
        img = img.resize(self.img_size, Image.BILINEAR)

        # Apply transforms (ToTensor + Normalize)
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
    """
    Collate function to produce padded batch for CTC training.
    - images stacked
    - labels concatenated into 1D tensor (required format for nn.CTCLoss)
    - input_lengths: sequence length (timesteps) produced by the model for each image
    - target_lengths: length of each label sequence
    """
    images = [b['image'] for b in batch]
    texts = [b['text'] for b in batch]
    labels = [b['label'] for b in batch]
    label_lengths = [b['label_length'] for b in batch]

    images = torch.stack(images, dim=0)
    if len(labels) == 0:
        labels_cat = torch.tensor([], dtype=torch.long)
    else:
        labels_cat = torch.cat(labels).to(torch.long)

    return {
        'images': images,
        'texts': texts,
        'labels': labels_cat,
        'label_lengths': torch.tensor(label_lengths, dtype=torch.long)
    }


class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        
        # CNN 
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
            nn.MaxPool2d((2, 1)), # Reduce height only

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((4, 1))  # Final collapse of height to 1
            
        )
        # RNN 
        self.rnn = nn.LSTM(512, 256, bidirectional=True, num_layers=2, dropout=0.2)
        
        # FC
        self.fc = nn.Linear(512, num_classes)


    def forward(self, x):
        # x: [Batch, 1, 32, 256]
        features = self.cnn(x) 
        # features: [Batch, 512, 1, 64]
        
        features = features.mean(2)
        features = features.permute(2, 0, 1)
        
        out, _ = self.rnn(features)
        out = self.fc(out)
        return F.log_softmax(out, dim=2) 
    

CER_INTERVAL = 100


def train_one_epoch(model, device, loader, optimizer, ctc_loss_fn):
    model.train()

    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        images = batch['images'].to(device)
        labels = batch['labels'].to(device)
        label_lengths = batch['label_lengths'].to(device)

        optimizer.zero_grad(set_to_none=True)

        log_probs = model(images)   # (T, B, C)
        T, B, C = log_probs.shape
        input_lengths = torch.full(
            (B,), T, dtype=torch.long, device=device
        )

        loss = ctc_loss_fn(
            log_probs.float(),
            labels,
            input_lengths,
            label_lengths
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * B
        total_samples += B

    avg_loss = total_loss / max(1, total_samples)
    return avg_loss



VAL_CER_INTERVAL = 20

def evaluate(model, device, loader, ctc_loss_fn):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            images = batch['images'].to(device)
            labels = batch['labels'].to(device)
            label_lengths = batch['label_lengths'].to(device)

            log_probs = model(images)
            T, B, C = log_probs.shape

            input_lengths = torch.full(
                (B,), T, dtype=torch.long, device=device
            )

            loss = ctc_loss_fn(
                log_probs,
                labels,
                input_lengths,
                label_lengths
            )

            total_loss += loss.item() * B
            total_samples += B

    avg_loss = total_loss / max(1, total_samples)
    return avg_loss


# --------------------------
# 6) Checkpoint utilities
# -------------------------- 
def save_checkpoint(state: dict, filename: str):
    os.makedirs(os.path.dirname(filename), exist_ok=True) 
    torch.save(state, filename) 

def load_checkpoint_if_exists(model, optimizer, filename: str, device):
    if filename and os.path.exists(filename):
        checkpoint = torch.load(filename, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        if optimizer is not None and 'optimizer_state' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Loaded checkpoint {filename} (resuming from epoch {start_epoch})")
        return start_epoch, checkpoint.get('best_val_loss', None)
    else:
        return 0, None


# --------------------------
# 8) Transformations 
# --------------------------
def default_transforms(img_size=img_size, train=True):
    h, w = img_size
    if train:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomRotation(degrees=2.5, fill=255),
            transforms.RandomAffine(
                degrees=5, 
                translate=(0.1, 0.1), 
                scale=(0.8, 1.2), 
                shear=10, 
                fill=255
            ),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            transforms.ColorJitter(brightness=0.3, contrast=0.5),
            transforms.Resize((h, w)), 
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((h, w)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])



def test_and_print(model, device, loader, ctc_loss_fn, max_print=3):
    model.eval()

    total_loss = 0.0
    total_samples = 0
    printed = 0


    with torch.no_grad():
        for batch in loader:
            images = batch['images'].to(device)
            labels = batch['labels'].to(device)
            label_lengths = batch['label_lengths'].to(device)

            log_probs = model(images)  # (T, B, C)
            T, B, _ = log_probs.shape
            input_lengths = torch.full(
                (B,), T, dtype=torch.long, device=device
            )

            loss = ctc_loss_fn(
                log_probs,
                labels,
                input_lengths,
                label_lengths
            )

            total_loss += loss.item() * B
            total_samples += B

            preds = greedy_decode(log_probs.permute(1, 0, 2))

            targets = batch['texts']



            if printed < max_print:
                for i in range(min(B, max_print - printed)):
                    print("GT:", targets[i])
                    print("PR:", preds[i])
                    print("-" * 30)
                    printed += 1

    avg_loss = total_loss / max(1, total_samples)
    return avg_loss



def predict_img(img_path: str, checkpoint_path: str, device='cpu', img_size=(32, 256)):
    device = torch.device(device)

    # build model
    num_classes = len(CHARSET) + 1
    model = CRNN(num_classes=num_classes).to(device)

    # load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    img = Image.open(img_path).convert("L")
    transform = default_transforms(img_size, train=True)
    img = transform(img).unsqueeze(0).to(device) 
    import matplotlib.pyplot as plt

    plt.imshow(img[0, 0].cpu().numpy(), cmap='gray')
    plt.axis('off')

    with torch.no_grad():
        log_probs = model(img)                  # (T, 1, C)
        log_probs = log_probs.permute(1, 0, 2)  # (1, T, C)

        pred = beam_decode(log_probs, beam_width=10)[0]

        # -------- Confidence --------
        probs = log_probs.exp()
        conf = probs.max(dim=-1)[0].mean().item()

    return pred, conf


if __name__ == "__main__":

    print("Using device:", device)

    #------------------ Dataset ---------------------
    # train dataset
    train_ds = HandwritingWordDataset(
        csv_file=train_csv,
        img_root=train_root,
        transform=default_transforms(img_size, train=True),
        img_size=img_size,
        train=True
    )


    indices = torch.randperm(len(train_ds), generator=torch.Generator().manual_seed(42)) [:100000]
    train_subset = Subset(train_ds, indices.tolist())

    # validation dataset    
    val_ds = HandwritingWordDataset(
        csv_file=val_csv,
        img_root=val_root,
        transform=default_transforms(img_size, train=False),
        img_size=img_size,
        train=False
    )
    # test dataset 
    test_ds = HandwritingWordDataset(
        csv_file=test_csv,
        img_root=test_root,
        transform=default_transforms(img_size, train=False),
        img_size=img_size,
        train=False
    )

    # ---------------------------- Dataloaders ------------------------------

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,collate_fn=collate_fn
                             ,num_workers=3, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,collate_fn=collate_fn
                           ,num_workers=2, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_ds,batch_size=batch_size,shuffle=False,collate_fn=collate_fn
                            ,num_workers=2, pin_memory=True, persistent_workers=True)

    # Model 
    num_classes = len(CHARSET) + 1  # + blank for CTC
    model = CRNN(num_classes).to(device)

    # loss function & Optimizer
    ctc_loss_fn = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    # load checkpoints 
    ckpt_path = os.path.join(checkpoint_dir,checkpoint_name)
    start_epoch, best_val_loss = load_checkpoint_if_exists(model, optimizer, ckpt_path, device) 
    if start_epoch == 0: 
        start_epoch = 1



    # ------------------------ train ---------------------
    best_acc = 0.0
    for epoch in range(start_epoch, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        train_loss, train_acc = train_one_epoch(
            model, device, train_loader, optimizer, ctc_loss_fn
        )

        print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.2f}%")

        ckpt = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'train_loss': train_loss
        }

        save_checkpoint(ckpt, os.path.join(checkpoint_dir, f'ocr_epoch_{epoch}.pth'))
        save_checkpoint(ckpt, ckpt_path)
