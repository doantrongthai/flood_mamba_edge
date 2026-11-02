import os
import json
import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
import cv2

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False

AIFLOODSENSE_CLASSES = FLOODSCENE_CLASSES = {0: 0, 85: 1, 170: 2, 255: 3}


def _find_dir(parent, candidates):
    for c in candidates:
        p = os.path.join(parent, c)
        if os.path.isdir(p):
            return p
    return None


class FloodSegmentationDataset(Dataset):
    def __init__(self, root_dir, split='train', size=128, seed=42, num_classes=1, dataset_type='floodkaggle'):
        self.dataset_type = dataset_type.lower()
        self.size = size
        self.seed = seed
        self.num_classes = num_classes
        self.split = split

        has_split_dir = os.path.isdir(os.path.join(root_dir, split))
        self.root_dir = os.path.join(root_dir, split) if has_split_dir else root_dir

        self.images_dir = _find_dir(self.root_dir, ['images', 'image'])
        if self.images_dir is None:
            self.images_dir = self.root_dir

        self.masks_dir = _find_dir(self.root_dir, ['masks', 'mask', 'labels', 'label'])
        if self.masks_dir is None:
            self.masks_dir = self.root_dir

        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Image directory not found in '{self.root_dir}'. Available: {os.listdir(root_dir) if os.path.exists(root_dir) else 'Directory does not exist'}")

        all_images = sorted([
            img for img in os.listdir(self.images_dir)
            if os.path.splitext(img)[1].lower() in [".png", ".jpg", ".jpeg"]
        ])

        # If dataset directory is flat (no train/val/test subdirectories):
        if not has_split_dir and len(all_images) > 0:
            if self.dataset_type == 'floodvn':
                # FloodVN is specifically an external evaluation benchmark:
                # evaluate on all available images for test/eval
                self.images = all_images
            else:
                rng = random.Random(seed)
                shuffled_indices = list(range(len(all_images)))
                rng.shuffle(shuffled_indices)
                n = len(shuffled_indices)
                n_train = int(n * 0.70)
                n_val = int(n * 0.15)

                if split == 'train':
                    chosen = shuffled_indices[:n_train]
                elif split == 'val':
                    chosen = shuffled_indices[n_train:n_train + n_val]
                else:  # test
                    chosen = shuffled_indices[n_train + n_val:]
                self.images = [all_images[i] for i in sorted(chosen)]
        else:
            self.images = all_images

        # LUT for AIFloodSense / FloodScene multi-class
        self.lut = np.zeros(256, dtype=np.uint8)
        for px, cls in FLOODSCENE_CLASSES.items():
            self.lut[px] = cls

        if HAS_ALBUMENTATIONS:
            if split == 'train':
                self.transform = A.Compose([
                    A.Resize(size, size),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.2),
                    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.3),
                    ToTensorV2()
                ], is_check_shapes=False)
            else:
                self.transform = A.Compose([
                    A.Resize(size, size),
                    ToTensorV2()
                ], is_check_shapes=False)
        else:
            self.transform = None

    def __len__(self):
        return len(self.images)

    def _load_mask(self, base_name, img_shape, img_name=None):
        h, w = img_shape[:2]

        # 1. Check for JSON polygon annotations FIRST (LabelMe / FloodVN format)
        for json_ext in ['.json', '.JSON']:
            json_path = os.path.join(self.masks_dir, base_name + json_ext)
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    mask_img = np.zeros((h, w), dtype=np.uint8)
                    shapes = data.get('shapes', [])
                    for shape in shapes:
                        points = np.array(shape.get('points', []), dtype=np.int32)
                        if len(points) >= 3:
                            cv2.fillPoly(mask_img, [points], color=255)
                    return mask_img
                except Exception:
                    pass

        # 2. Check for standard image mask (png, jpg) in masks_dir
        # Guard: if masks_dir == images_dir, do NOT mistakenly load the input image itself as a mask
        is_same_dir = os.path.abspath(self.masks_dir) == os.path.abspath(self.images_dir)
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.PNG', '.JPG', '.JPEG']:
            candidate_file = base_name + ext
            if is_same_dir and img_name is not None and candidate_file.lower() == img_name.lower():
                continue
            mask_path = os.path.join(self.masks_dir, candidate_file)
            if os.path.exists(mask_path):
                mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask_img is not None:
                    if mask_img.shape[:2] != (h, w):
                        mask_img = cv2.resize(mask_img, (w, h), interpolation=cv2.INTER_NEAREST)
                    return mask_img

        # Fallback empty mask if none found
        return np.zeros((h, w), dtype=np.uint8)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to load image at {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        base_name = os.path.splitext(img_name)[0]
        mask_img = self._load_mask(base_name, image.shape, img_name=img_name)

        # Dataset-specific mask parsing
        if self.dataset_type in ['aifloodsense', 'floodscene']:
            if self.num_classes == 1:
                # Binary: 255 is flood
                mask = (mask_img == 255).astype(np.uint8)
            else:
                # Multi-class: map 0, 85, 170, 255 -> 0, 1, 2, 3
                mask = self.lut[mask_img].astype(np.uint8)

        else:
            # Binary masks (FloodKaggle, FloodVN)
            mask = (mask_img > 127).astype(np.uint8)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image'].float() / 255.0
            mask_out = transformed['mask']
        else:
            # Deterministic OpenCV / torch fallback transform
            image = cv2.resize(image, (self.size, self.size))
            mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
            if self.split == 'train':
                if random.random() > 0.5:
                    image = cv2.flip(image, 1)
                    mask = cv2.flip(mask, 1)
                if random.random() > 0.5:
                    image = cv2.flip(image, 0)
                    mask = cv2.flip(mask, 0)
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask_out = torch.from_numpy(mask)

        if mask_out is None:
            raise ValueError(f"Mask is None after transform at index {idx}")

        if self.num_classes == 1:
            mask_tensor = mask_out.unsqueeze(0).float() if mask_out.ndim == 2 else mask_out.float()
        else:
            mask_tensor = mask_out.long()

        return image, mask_tensor


def seed_worker(worker_id):
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(worker_seed)


def get_dataloaders(dataset, batch_size=4, size=128, seed=42, num_classes=1, dataset_type='floodkaggle'):
    # Resolve directory alias between aifloodsense and floodscene
    if not os.path.exists(dataset):
        if dataset.lower() == 'aifloodsense' and os.path.exists('floodscene'):
            dataset = 'floodscene'
        elif dataset.lower() == 'floodscene' and os.path.exists('aifloodsense'):
            dataset = 'aifloodsense'
    train_dataset = FloodSegmentationDataset(dataset, 'train', size, seed, num_classes, dataset_type=dataset_type)
    val_dataset   = FloodSegmentationDataset(dataset, 'val',   size, seed, num_classes, dataset_type=dataset_type)
    test_dataset  = FloodSegmentationDataset(dataset, 'test',  size, seed, num_classes, dataset_type=dataset_type)

    g = torch.Generator()
    g.manual_seed(seed)

    # Optimal workers for Colab / Edge CPU (avoids IPC/shared memory overhead)
    workers = min(2, os.cpu_count() or 2)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=torch.cuda.is_available(),
        drop_last=True, worker_init_fn=seed_worker,
        generator=g, persistent_workers=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=torch.cuda.is_available(),
        drop_last=False, worker_init_fn=seed_worker,
        persistent_workers=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=torch.cuda.is_available(),
        drop_last=False, worker_init_fn=seed_worker,
        persistent_workers=False
    )

    return train_loader, val_loader, test_loader
