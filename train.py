"""
Entraînement Mask R-CNN pour segmentation des toitures cadastrales
Dataset: Images aériennes annotées avec CVAT (format COCO)
Classes: toiture_tole_ondulee, toiture_tole_bac, toiture_tuile, toiture_dalle
"""

import os
import json
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import v2 as T
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask_utils
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Chemins (à adapter)
    "images_dir": ".C:/Users/NEBRATA/Desktop/Memoire/modeles/segmentation/dataset1/default/images",
    "annotations_file": "C:/Users/NEBRATA/Desktop/Memoire/modeles/segmentation/dataset1/annotations/instances_default.json",
    # "images_dir": "./data/images",
    # "annotations_file": "./data/annotations/instances_default.json",
    "output_dir": "./output",
    
    # Classes (dans l'ordre de CVAT)
    "classes": [
        "__background__",      # 0 - toujours en premier
        "toiture_tole_ondulee",  # 1
        "toiture_tole_bac",      # 2
        "toiture_tuile",         # 3
        "toiture_dalle"          # 4
    ],
    
    # Hyperparamètres
    "num_epochs": 25,
    "batch_size": 2,           # 2-4 pour GPU 8GB, augmenter si plus de VRAM
    "learning_rate": 0.005,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "lr_step_size": 8,
    "lr_gamma": 0.1,
    
    # Dataset
    "train_split": 0.85,       # 85% train, 15% validation
    "num_workers": 2,
    
    # Sauvegarde
    "save_every": 5,           # Sauvegarder tous les N epochs
}


# =============================================================================
# DATASET
# =============================================================================

class CadastralDataset(torch.utils.data.Dataset):
    """Dataset pour segmentation cadastrale depuis annotations COCO/CVAT"""
    
    def __init__(self, images_dir, annotations_file, transforms=None):
        self.images_dir = images_dir
        self.transforms = transforms
        
        # Charger annotations COCO
        self.coco = COCO(annotations_file)
        self.image_ids = list(self.coco.imgs.keys())
        
        # Mapping catégories COCO -> indices locaux
        self.cat_ids = self.coco.getCatIds()
        self.cat_mapping = {cat_id: idx + 1 for idx, cat_id in enumerate(self.cat_ids)}
        
        print(f"Dataset chargé: {len(self.image_ids)} images")
        print(f"Catégories: {[self.coco.cats[c]['name'] for c in self.cat_ids]}")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.imgs[img_id]
        
        # Charger l'image
        img_path = os.path.join(self.images_dir, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")
        
        # Récupérer les annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        boxes = []
        labels = []
        masks = []
        areas = []
        
        for ann in anns:
            # Ignorer les annotations invalides
            if ann.get('iscrowd', 0):
                continue
            
            # Boîte englobante [x, y, width, height] -> [x1, y1, x2, y2]
            x, y, w, h = ann['bbox']
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            
            # Classe (remapper vers indices locaux)
            labels.append(self.cat_mapping[ann['category_id']])
            
            # Masque
            if 'segmentation' in ann:
                if isinstance(ann['segmentation'], list):
                    # Polygone -> RLE -> masque binaire
                    rles = coco_mask_utils.frPyObjects(
                        ann['segmentation'],
                        img_info['height'],
                        img_info['width']
                    )
                    rle = coco_mask_utils.merge(rles)
                    mask = coco_mask_utils.decode(rle)
                else:
                    # Déjà en RLE
                    mask = coco_mask_utils.decode(ann['segmentation'])
                masks.append(mask)
            
            # Aire
            areas.append(ann.get('area', w * h))
        
        # Gérer le cas sans annotations
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, img_info['height'], img_info['width']), dtype=torch.uint8)
            areas = torch.zeros((0,), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
            areas = torch.as_tensor(areas, dtype=torch.float32)
        
        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([img_id]),
            "area": areas,
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64)
        }
        
        # Convertir image en tensor
        image = T.ToTensor()(image)
        
        # Appliquer transformations
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        
        return image, target


# =============================================================================
# TRANSFORMATIONS (Augmentation)
# =============================================================================

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob
    
    def __call__(self, image, target):
        if np.random.random() < self.prob:
            image = torch.flip(image, [-1])
            
            # Ajuster les boîtes
            if "boxes" in target and len(target["boxes"]) > 0:
                width = image.shape[-1]
                boxes = target["boxes"]
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                target["boxes"] = boxes
            
            # Ajuster les masques
            if "masks" in target and len(target["masks"]) > 0:
                target["masks"] = torch.flip(target["masks"], [-1])
        
        return image, target


class RandomVerticalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob
    
    def __call__(self, image, target):
        if np.random.random() < self.prob:
            image = torch.flip(image, [-2])
            
            if "boxes" in target and len(target["boxes"]) > 0:
                height = image.shape[-2]
                boxes = target["boxes"]
                boxes[:, [1, 3]] = height - boxes[:, [3, 1]]
                target["boxes"] = boxes
            
            if "masks" in target and len(target["masks"]) > 0:
                target["masks"] = torch.flip(target["masks"], [-2])
        
        return image, target


def get_transforms(train=True):
    transforms = []
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
        transforms.append(RandomVerticalFlip(0.5))
    return Compose(transforms)


# =============================================================================
# MODÈLE
# =============================================================================

def get_model(num_classes):
    """Créer un Mask R-CNN fine-tuné pour N classes"""
    
    # Charger le modèle pré-entraîné sur COCO
    model = maskrcnn_resnet50_fpn_v2(weights="DEFAULT")
    
    # Remplacer le classificateur de boîtes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Remplacer le prédicteur de masques
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    
    return model


# =============================================================================
# ENTRAÎNEMENT
# =============================================================================

def collate_fn(batch):
    """Fonction de collation pour DataLoader"""
    return tuple(zip(*batch))


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """Entraîner une epoch"""
    model.train()
    
    total_loss = 0
    loss_classifier = 0
    loss_box_reg = 0
    loss_mask = 0
    loss_objectness = 0
    loss_rpn_box = 0
    
    pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}")
    
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        # Accumuler les pertes
        total_loss += losses.item()
        loss_classifier += loss_dict.get('loss_classifier', torch.tensor(0)).item()
        loss_box_reg += loss_dict.get('loss_box_reg', torch.tensor(0)).item()
        loss_mask += loss_dict.get('loss_mask', torch.tensor(0)).item()
        loss_objectness += loss_dict.get('loss_objectness', torch.tensor(0)).item()
        loss_rpn_box += loss_dict.get('loss_rpn_box_reg', torch.tensor(0)).item()
        
        pbar.set_postfix({
            'loss': f"{losses.item():.4f}",
            'mask': f"{loss_dict.get('loss_mask', torch.tensor(0)).item():.4f}"
        })
    
    n = len(data_loader)
    return {
        'total': total_loss / n,
        'classifier': loss_classifier / n,
        'box_reg': loss_box_reg / n,
        'mask': loss_mask / n,
        'objectness': loss_objectness / n,
        'rpn_box': loss_rpn_box / n
    }


@torch.no_grad()
def evaluate(model, data_loader, device):
    """Évaluer sur le set de validation"""
    model.eval()
    
    total_loss = 0
    
    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # En mode eval, on doit passer en mode train temporairement pour avoir les pertes
        model.train()
        loss_dict = model(images, targets)
        model.eval()
        
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()
    
    return total_loss / len(data_loader)


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Sauvegarder un checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("MASK R-CNN - Segmentation des Toitures Cadastrales")
    print("=" * 60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Créer le dossier de sortie
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
    # Dataset
    print("\nChargement du dataset...")
    full_dataset = CadastralDataset(
        CONFIG["images_dir"],
        CONFIG["annotations_file"],
        transforms=None
    )
    
    # Split train/val
    train_size = int(CONFIG["train_split"] * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train: {len(train_dataset)} images")
    print(f"Val: {len(val_dataset)} images")
    
    # Appliquer les transformations
    train_dataset.dataset.transforms = get_transforms(train=True)
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Modèle
    print("\nCréation du modèle...")
    num_classes = len(CONFIG["classes"])
    model = get_model(num_classes)
    model.to(device)
    print(f"Classes: {CONFIG['classes']}")
    
    # Optimiseur
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=CONFIG["learning_rate"],
        momentum=CONFIG["momentum"],
        weight_decay=CONFIG["weight_decay"]
    )
    
    # Scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=CONFIG["lr_step_size"],
        gamma=CONFIG["lr_gamma"]
    )
    
    # Historique des pertes
    history = {
        'train_loss': [],
        'val_loss': [],
        'lr': []
    }
    
    best_val_loss = float('inf')
    
    # Entraînement
    print("\n" + "=" * 60)
    print("DÉBUT DE L'ENTRAÎNEMENT")
    print("=" * 60)
    
    for epoch in range(CONFIG["num_epochs"]):
        # Train
        train_losses = train_one_epoch(model, optimizer, train_loader, device, epoch)
        
        # Validation
        val_loss = evaluate(model, val_loader, device)
        
        # Scheduler
        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Historique
        history['train_loss'].append(train_losses['total'])
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)
        
        # Affichage
        print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")
        print(f"  Train Loss: {train_losses['total']:.4f} (mask: {train_losses['mask']:.4f})")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  LR: {current_lr:.6f}")
        
        # Sauvegarder le meilleur modèle
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                os.path.join(CONFIG["output_dir"], "best_model.pth")
            )
            print("  ✓ Meilleur modèle sauvegardé!")
        
        # Sauvegardes périodiques
        if (epoch + 1) % CONFIG["save_every"] == 0:
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                os.path.join(CONFIG["output_dir"], f"checkpoint_epoch_{epoch+1}.pth")
            )
    
    # Sauvegarder le modèle final
    save_checkpoint(
        model, optimizer, CONFIG["num_epochs"]-1, val_loss,
        os.path.join(CONFIG["output_dir"], "final_model.pth")
    )
    
    # Sauvegarder l'historique
    with open(os.path.join(CONFIG["output_dir"], "history.json"), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot des courbes de perte
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Courbes de perte - Mask R-CNN Cadastral')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(CONFIG["output_dir"], "loss_curves.png"), dpi=150)
    plt.close()
    
    print("\n" + "=" * 60)
    print("ENTRAÎNEMENT TERMINÉ")
    print(f"Meilleure Val Loss: {best_val_loss:.4f}")
    print(f"Modèles sauvegardés dans: {CONFIG['output_dir']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
