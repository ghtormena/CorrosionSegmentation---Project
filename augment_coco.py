import os
import json
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
from pycocotools.coco import COCO
import albumentations as A
from albumentations import (
    HorizontalFlip, VerticalFlip, Rotate, GaussianBlur, GaussNoise, RandomResizedCrop
)


# Caminhos
INPUT_IMAGE_DIR = "images"
ANNOTATION_PATH = "/home/giovanna/Documentos/IC/versao1_ds_anotado/DSv1/annotations/instances_default.json"
OUTPUT_IMAGE_DIR = "/home/giovanna/Documentos/IC/versao1_ds_anotado/DSv1/augmented/images"
OUTPUT_ANN_PATH = "/home/giovanna/Documentos/IC/versao1_ds_anotado/DSv1/augmented/annotations/augmented_coco.json"
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_ANN_PATH), exist_ok=True)

# Augmentations
transform = A.Compose([
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.2),
    Rotate(limit=15, p=0.5),
    RandomResizedCrop(size=(512, 512), scale=(0.9, 1.0), p=0.5),
    GaussianBlur(blur_limit=3, p=0.3),
    GaussNoise(std_range=(0.1, 0.2), p=0.3),
    A.ColorJitter(brightness=0.2, saturation=0.2, contrast=0.0, hue=0.0, p=0.3),
], additional_targets={'mask': 'mask'})

# Carregar COCO
coco = COCO(ANNOTATION_PATH)
new_images = []
new_annotations = []
ann_id = max(ann['id'] for ann in coco.dataset['annotations']) + 1
img_id = max(img['id'] for img in coco.dataset['images']) + 1

for img in tqdm(coco.dataset['images']):
    file_name = img['file_name']
    img_path = os.path.join(INPUT_IMAGE_DIR, file_name)
    image = cv2.imread(img_path)
    height, width = image.shape[:2]
    
    # Copiar imagem original e ajustar o nome
    only_filename = os.path.basename(file_name)  # Ex: 'image_0000.jpg'
    new_path = os.path.join(OUTPUT_IMAGE_DIR, only_filename)

    cv2.imwrite(new_path, image)  # Salva a imagem original na pasta sem subpasta

    # Atualiza o campo file_name no JSON final
    img['file_name'] = only_filename
    new_images.append(img)



    ann_ids = coco.getAnnIds(imgIds=img['id'])
    anns = coco.loadAnns(ann_ids)

    for i in range(5):  # número de augmentations por imagem
        mask = np.zeros((height, width), dtype=np.uint8)
        for ann in anns:
            rle = coco.annToMask(ann)
            mask = np.maximum(mask, rle * 255)

        augmented = transform(image=image, mask=mask)
        aug_image = augmented['image']
        aug_mask = augmented['mask']
        aug_file = f"{Path(file_name).stem}_aug{i}.jpg"
        aug_path = os.path.join(OUTPUT_IMAGE_DIR, aug_file)
        cv2.imwrite(aug_path, aug_image)

        # Atualiza entrada da imagem
        new_images.append({
            'id': img_id,
            'file_name': aug_file,
            'width': aug_image.shape[1],
            'height': aug_image.shape[0]
        })

        # Extrair novos contornos
        contours, _ = cv2.findContours(aug_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < 50:
                continue
            segmentation = contour.flatten().tolist()
            x, y, w, h = cv2.boundingRect(contour)
            new_annotations.append({
                'id': ann_id,
                'image_id': img_id,
                'category_id': anns[0]['category_id'],
                'segmentation': [segmentation],
                'area': float(w * h),
                'bbox': [x, y, w, h],
                'iscrowd': 0
            })
            ann_id += 1
        img_id += 1

# Salvar novo JSON COCO
augmented_coco = coco.dataset.copy()
augmented_coco['images'] = new_images
augmented_coco['annotations'] = new_annotations + coco.dataset['annotations']

with open(OUTPUT_ANN_PATH, 'w') as f:
    json.dump(augmented_coco, f)

print("✅ Dataset aumentado com sucesso!")