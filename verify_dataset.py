"""
Vérification du dataset CVAT/COCO avant entraînement
Lance ce script pour valider tes annotations
"""

import os
import json
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
from pycocotools import mask as coco_mask_utils
from dotenv import load_dotenv


def verify_dataset(images_dir, annotations_file):
    """Vérifier l'intégrité du dataset"""
    
    print("=" * 60)
    print("VÉRIFICATION DU DATASET CVAT/COCO")
    print("=" * 60)
    
    errors = []
    warnings = []
    
    # 1. Vérifier que le fichier d'annotations existe
    if not os.path.exists(annotations_file):
        print(f"❌ ERREUR: Fichier d'annotations introuvable: {annotations_file}")
        return False
    
    # 2. Charger les annotations
    try:
        coco = COCO(annotations_file)
        print(f"✓ Annotations chargées: {annotations_file}")
    except Exception as e:
        print(f"❌ ERREUR: Impossible de charger les annotations: {e}")
        return False
    
    # 3. Afficher les statistiques générales
    print(f"\n📊 STATISTIQUES GÉNÉRALES")
    print("-" * 40)
    print(f"  Images: {len(coco.imgs)}")
    print(f"  Annotations: {len(coco.anns)}")
    print(f"  Catégories: {len(coco.cats)}")
    
    # 4. Afficher les catégories
    print(f"\n📂 CATÉGORIES")
    print("-" * 40)
    for cat_id, cat in coco.cats.items():
        ann_count = len(coco.getAnnIds(catIds=[cat_id]))
        print(f"  [{cat_id}] {cat['name']}: {ann_count} annotations")
    
    # 5. Vérifier les images
    print(f"\n🖼️  VÉRIFICATION DES IMAGES")
    print("-" * 40)
    
    missing_images = []
    valid_images = 0
    
    for img_id, img_info in coco.imgs.items():
        img_path = os.path.join(images_dir, img_info['file_name'])
        
        if not os.path.exists(img_path):
            missing_images.append(img_info['file_name'])
            continue
        
        # Vérifier les dimensions
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                if w != img_info['width'] or h != img_info['height']:
                    warnings.append(
                        f"Dimensions incorrectes pour {img_info['file_name']}: "
                        f"annotation ({img_info['width']}x{img_info['height']}) vs "
                        f"réel ({w}x{h})"
                    )
            valid_images += 1
        except Exception as e:
            errors.append(f"Erreur lecture {img_info['file_name']}: {e}")
    
    print(f"  Images valides: {valid_images}/{len(coco.imgs)}")
    
    if missing_images:
        print(f"  ❌ Images manquantes: {len(missing_images)}")
        for img in missing_images[:5]:
            print(f"     - {img}")
        if len(missing_images) > 5:
            print(f"     ... et {len(missing_images)-5} autres")
    
    # 6. Vérifier les annotations
    print(f"\n📝 VÉRIFICATION DES ANNOTATIONS")
    print("-" * 40)
    
    invalid_bbox = 0
    invalid_segmentation = 0
    empty_segmentation = 0
    
    for ann_id, ann in coco.anns.items():
        # Vérifier bbox
        if 'bbox' in ann:
            x, y, w, h = ann['bbox']
            if w <= 0 or h <= 0:
                invalid_bbox += 1
        
        # Vérifier segmentation
        if 'segmentation' in ann:
            seg = ann['segmentation']
            if isinstance(seg, list):
                if len(seg) == 0:
                    empty_segmentation += 1
                elif any(len(poly) < 6 for poly in seg):
                    invalid_segmentation += 1
    
    print(f"  Bbox invalides (w<=0 ou h<=0): {invalid_bbox}")
    print(f"  Segmentations invalides: {invalid_segmentation}")
    print(f"  Segmentations vides: {empty_segmentation}")
    
    # 7. Statistiques par image
    print(f"\n📈 DISTRIBUTION DES ANNOTATIONS")
    print("-" * 40)
    
    anns_per_image = []
    for img_id in coco.imgs:
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns_per_image.append(len(ann_ids))
    
    anns_per_image = np.array(anns_per_image)
    print(f"  Min annotations/image: {anns_per_image.min()}")
    print(f"  Max annotations/image: {anns_per_image.max()}")
    print(f"  Moyenne: {anns_per_image.mean():.1f}")
    print(f"  Images sans annotation: {np.sum(anns_per_image == 0)}")
    
    # 8. Warnings et erreurs
    if warnings:
        print(f"\n⚠️  WARNINGS ({len(warnings)})")
        print("-" * 40)
        for w in warnings[:10]:
            print(f"  {w}")
    
    if errors:
        print(f"\n❌ ERREURS ({len(errors)})")
        print("-" * 40)
        for e in errors[:10]:
            print(f"  {e}")
    
    # 9. Résumé
    print(f"\n" + "=" * 60)
    if len(errors) == 0 and len(missing_images) == 0:
        print("✅ DATASET VALIDE - Prêt pour l'entraînement!")
        return True
    else:
        print("❌ DATASET CONTIENT DES ERREURS - Corriger avant entraînement")
        return False


def visualize_sample(images_dir, annotations_file, num_samples=3):
    """Visualiser quelques échantillons annotés"""
    
    coco = COCO(annotations_file)
    
    # Sélectionner des images avec annotations
    img_ids = []
    for img_id in coco.imgs:
        if len(coco.getAnnIds(imgIds=[img_id])) > 0:
            img_ids.append(img_id)
    
    np.random.shuffle(img_ids)
    img_ids = img_ids[:num_samples]
    
    fig, axes = plt.subplots(1, num_samples, figsize=(6*num_samples, 6))
    if num_samples == 1:
        axes = [axes]
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for ax, img_id in zip(axes, img_ids):
        img_info = coco.imgs[img_id]
        img_path = os.path.join(images_dir, img_info['file_name'])
        
        image = Image.open(img_path)
        ax.imshow(image)
        
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ann_ids)
        
        for i, ann in enumerate(anns):
            color = colors[ann['category_id'] % 10]
            
            # Dessiner le masque
            if 'segmentation' in ann:
                if isinstance(ann['segmentation'], list):
                    for seg in ann['segmentation']:
                        poly = np.array(seg).reshape(-1, 2)
                        ax.fill(poly[:, 0], poly[:, 1], alpha=0.4, color=color)
                        ax.plot(poly[:, 0], poly[:, 1], color=color, linewidth=2)
            
            # Dessiner la bbox
            if 'bbox' in ann:
                x, y, w, h = ann['bbox']
                rect = plt.Rectangle((x, y), w, h, 
                                     fill=False, edgecolor=color, linewidth=2)
                ax.add_patch(rect)
                
                cat_name = coco.cats[ann['category_id']]['name']
                ax.text(x, y-5, cat_name, fontsize=8, color='white',
                       bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
        
        ax.set_title(f"{img_info['file_name']}\n{len(anns)} annotations")
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig("dataset_samples.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Échantillons sauvegardés: dataset_samples.png")


if __name__ == "__main__":
    import argparse

    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Vérifier le dataset CVAT")
    parser.add_argument("--images", type=str, 
                    default=os.getenv("SEGMENTATION_DATASET_IMAGES_DIR"),
                    help="Dossier des images")
    parser.add_argument("--annotations", type=str, 
                    default=os.getenv("SEGMENTATION_DATASET_ANNOTATIONS_FILE"),
                    help="Fichier d'annotations COCO JSON")
    parser.add_argument("--visualize", action="store_true",
                       help="Visualiser des échantillons")
    parser.add_argument("--num-samples", type=int, default=3,
                       help="Nombre d'échantillons à visualiser")
    
    args = parser.parse_args()


    print("Images:", os.getenv("SEGMENTATION_DATASET_IMAGES_DIR"))
    print("Annotations:", os.getenv("SEGMENTATION_DATASET_ANNOTATIONS_FILE"))
    
    valid = verify_dataset(args.images, args.annotations)
    
    if valid and args.visualize:
        visualize_sample(args.images, args.annotations, args.num_samples)
