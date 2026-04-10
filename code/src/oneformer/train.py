import os
import cv2
import torch
import copy
import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms
from transformers import AutoImageProcessor, OneFormerForUniversalSegmentation
import segmentation_models_pytorch as smp
import torch.optim as optim
from segmentation_models_pytorch.utils.metrics import IoU, Fscore
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms  # ok to keep
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation
import sys
sys.path.append('/workspace/mycode/CODE')
from code.models.utils import AnySmokeSegDataset

# -----------------------
# Globals / config
# -----------------------
SEQ_LEN = 77                # CLIP-style sequence length used by OneFormer text inputs
IMAGE_SIZE = (512, 512)     # Training/eval resolution
LABEL_NAME = "smoke"
LABEL_INDEX = 1             # We'll map "smoke" -> 1, reserving 0 for "background"
id2label = {LABEL_INDEX: LABEL_NAME}
label2id = {LABEL_NAME: LABEL_INDEX}

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('/workspace/mycode/oneformer2')


# -----------------------
# Metrics
# -----------------------
def compute_metrics(outputs, masks, threshold=0.5, eps=1e-6):
    """
    outputs: raw logits, shape (B,1,H,W)
    masks: ground truth in {0,1}, shape (B, H, W)
    """
    probs = torch.sigmoid(outputs).detach()
    preds = (probs > threshold).float()
    masks = masks.float()

    preds_flat = preds.view(-1)
    masks_flat = masks.view(-1)

    TP = (preds_flat * masks_flat).sum()
    FP = (preds_flat * (1 - masks_flat)).sum()
    FN = ((1 - preds_flat) * masks_flat).sum()

    precision = TP / (TP + FP + eps)
    recall    = TP / (TP + FN + eps)
    iou       = TP / (TP + FP + FN + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    mse       = torch.mean((probs.view(-1) - masks_flat) ** 2)

    return {
        'iou': iou.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item(),
        'mse': mse.item()
    }

# -----------------------
# Text task inputs (with safe fallback)
# -----------------------
def make_task_inputs(tokenizer, batch_size: int, device, seq_len: int = SEQ_LEN) -> torch.Tensor:
    """
    Build (B, seq_len) integer task_inputs. If a tokenizer is provided, use it;
    otherwise, fall back to CLIP-ish BOS/EOS + pad.
    """
    if tokenizer is not None:
        ti = tokenizer(["semantic"] * batch_size,
                       padding="max_length",
                       truncation=True,
                       max_length=seq_len,
                       return_tensors="pt")
        return ti["input_ids"].to(device)

    # Fallback (common CLIP IDs): seq_len=77, BOS=49406, EOS=49407, pad=0
    bos, eos = 49406, 49407
    x = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
    x[:, 0] = bos
    x[:, 1] = eos
    return x

# -----------------------
# Query fusion helper
# -----------------------
def fuse_per_pixel_logits(outputs, label_index: int, out_size=IMAGE_SIZE):
    """
    Convert OneFormer query outputs into a single per-pixel logit map for one class.

    outputs.masks_queries_logits: (B, Q, Hm, Wm)
    outputs.class_queries_logits: (B, Q, C)

    We weight each mask query by P(class == target_label) and sum over queries.
    """
    # (B, Q, Hm, Wm)
    mask_logits = outputs.masks_queries_logits
    # (B, Q, C)
    class_logits = outputs.class_queries_logits

    # Convert class logits to probabilities and pick the column for our label
    # (B, Q)
    class_probs_for_label = torch.softmax(class_logits, dim=-1)[..., label_index]

    # Upsample masks to the final image size
    # (B, Q, H, W)
    up_mask = torch.nn.functional.interpolate(
        mask_logits, size=out_size, mode="bilinear", align_corners=False
    )

    # Broadcast (B, Q) -> (B, Q, 1, 1) for weighting, then sum over Q
    weights = class_probs_for_label.unsqueeze(-1).unsqueeze(-1)
    # (B, 1, H, W)
    per_pixel_logits = (up_mask * weights).sum(dim=1, keepdim=True)
    return per_pixel_logits

# -----------------------
# Evaluation (fixed)
# -----------------------
def evaluate(model, loader, device):
    model.eval()
    metrics_sum = {'iou':0.0, 'precision':0.0, 'recall':0.0, 'f1':0.0, 'mse':0.0}
    n_batches = 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks  = masks.to(device)  # shape (B, H, W) expected by compute_metrics

            # Build task inputs (safe with tokenizer=None)
            task_inputs = make_task_inputs(tokenizer, images.size(0), device, seq_len=SEQ_LEN)

            # Forward pass
            outputs = model(pixel_values=images, task_inputs=task_inputs)

            # Fuse queries -> per-pixel logits (B,1,H,W)
            per_pixel_logits = fuse_per_pixel_logits(outputs, LABEL_INDEX, IMAGE_SIZE)

            # Compute batch metrics
            batch_metrics = compute_metrics(per_pixel_logits, masks)
            for k, v in batch_metrics.items():
                metrics_sum[k] += v
            n_batches += 1

    # Average
    for k in metrics_sum:
        metrics_sum[k] /= max(n_batches, 1)
    return metrics_sum

# -----------------------
# Training
# -----------------------
def train():
    # Hyperparameters
    num_epochs = 20
    batch_size = 8
    lr = 1e-4

    train_dir      = "/workspace/mycode/AnySmokeDataset/AnySmokeTrain"
    val_dir        = "/workspace/mycode/AnySmokeDataset/AnySmokeTest"
    val_small_dir  = "/workspace/mycode/AnySmokeDataset/AnySmokeTestSmall"
    val_medium_dir = "/workspace/mycode/AnySmokeDataset/AnySmokeTestMedium"
    val_large_dir  = "/workspace/mycode/AnySmokeDataset/AnySmokeTestLarge"

    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_ds       = AnySmokeSegDataset(train_dir, transform=transform)
    val_ds         = AnySmokeSegDataset(val_dir, transform=transform)
    val_small_ds   = AnySmokeSegDataset(val_small_dir, transform=transform)
    val_medium_ds  = AnySmokeSegDataset(val_medium_dir, transform=transform)
    val_large_ds   = AnySmokeSegDataset(val_large_dir, transform=transform)



 ### Ablation code start
    # num_samples = len(train_ds)
    # indices     = list(range(num_samples))
    # np.random.seed(42)
    # np.random.shuffle(indices)
    
    # Scale = 0.8
    # split = int(Scale * num_samples)
    # print(Scale)
    # train_idx = indices[:split]
    # sampler = SubsetRandomSampler(train_idx)

    # train_loader = DataLoader(
    #     train_ds,
    #     batch_size=batch_size,
    #     sampler=sampler,
    #     num_workers=4
    # )

    # print(len(train_loader)*batch_size)

    train_loader       = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4)
    val_loader         = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    val_small_loader   = DataLoader(val_small_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    val_medium_loader  = DataLoader(val_medium_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    val_large_loader   = DataLoader(val_large_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    MODEL_NAME = '/workspace/mycode/oneformer2'
    print(MODEL_NAME)

    model = OneFormerForUniversalSegmentation.from_pretrained(
        MODEL_NAME,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_val_iou = 0.0
    print("Start training ...")

    for epoch in tqdm(range(1, num_epochs + 1)):
        model.train()
        running_loss = 0.0

        for images, masks in train_loader:
            images = images.to(device)
            # For loss, we need (B,1,H,W)
            masks_logits_shape = masks.to(device).unsqueeze(1).float()

            optimizer.zero_grad()

            task_inputs = make_task_inputs(tokenizer, images.size(0), device, seq_len=SEQ_LEN)
            outputs = model(pixel_values=images, task_inputs=task_inputs)

            # (B,1,H,W)
            per_pixel_logits = fuse_per_pixel_logits(outputs, LABEL_INDEX, IMAGE_SIZE)

            loss = criterion(per_pixel_logits, masks_logits_shape)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / max(len(train_loader), 1)

        # -------- Validation --------
        val_metrics = evaluate(model, val_loader, device)
        val_iou = val_metrics['iou']

        print(
            f"Epoch [{epoch}/{num_epochs}] "
            f"Train Loss: {avg_train_loss:.4f} "
            f"Val IoU: {val_iou:.4f} "
            f"Precision: {val_metrics['precision']:.4f} "
            f"Recall: {val_metrics['recall']:.4f} "
            f"F1: {val_metrics['f1']:.4f} "
            f"MSE: {val_metrics['mse']:.6f}"
        )

        # Size-binned validation
        val_large_metrics  = evaluate(model, val_large_loader, device)
        print("large mask performance")
        print(
            f"Val IoU: {val_large_metrics['iou']:.4f} "
            f"Precision: {val_large_metrics['precision']:.4f} "
            f"Recall: {val_large_metrics['recall']:.4f} "
            f"F1: {val_large_metrics['f1']:.4f} "
            f"MSE: {val_large_metrics['mse']:.6f}"
        )

        val_medium_metrics = evaluate(model, val_medium_loader, device)
        print("medium mask performance")
        print(
            f"Val IoU: {val_medium_metrics['iou']:.4f} "
            f"Precision: {val_medium_metrics['precision']:.4f} "
            f"Recall: {val_medium_metrics['recall']:.4f} "
            f"F1: {val_medium_metrics['f1']:.4f} "
            f"MSE: {val_medium_metrics['mse']:.6f}"
        )

        val_small_metrics  = evaluate(model, val_small_loader, device)
        print("small mask performance")
        print(
            f"Val IoU: {val_small_metrics['iou']:.4f} "
            f"Precision: {val_small_metrics['precision']:.4f} "
            f"Recall: {val_small_metrics['recall']:.4f} "
            f"F1: {val_small_metrics['f1']:.4f} "
            f"MSE: {val_small_metrics['mse']:.6f}"
        )

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            # Optional: save a checkpoint here if desired
            # torch.save(model.state_dict(), "best_oneformer_smoke.pth")

    print(f"Training complete. Best Val IoU: {best_val_iou:.4f}")

    final_path = "./1028_model.pth"
    torch.save(model.state_dict(), final_path)

if __name__ == "__main__":
    train()
