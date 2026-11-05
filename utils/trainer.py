import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# ==============================================================================
# 1. BOUNDARY EXTRACTOR & CUSTOM BOUNDARY LOSS
# ==============================================================================
def extract_boundaries(masks, kernel_size=3):
    # Dilate mask
    dilated = F.max_pool2d(masks, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    # Erode mask
    eroded = -F.max_pool2d(-masks, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    # Get boundary
    boundary = dilated - eroded
    # Hard binary threshold
    boundary = (boundary > 0.5).float()
    return boundary


class BoundaryLoss(nn.Module):
    def __init__(self, pos_weight_val=10.0, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val]))
        self.dice_weight = dice_weight

    def dice_loss(self, pred, target, smooth=1e-5):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice.mean()

    def forward(self, pred_boundary, gt_mask):
        gt_boundary = extract_boundaries(gt_mask)
        loss_bce = self.bce(pred_boundary, gt_boundary)
        loss_dice = self.dice_loss(pred_boundary, gt_boundary)
        return loss_bce + self.dice_weight * loss_dice


class PlainDiceLoss(nn.Module):
    def forward(self, pred, target, smooth=1e-5):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice.mean()


def get_aux_loss(name):
    name = name.lower()
    if name == 'boundary':
        return BoundaryLoss(pos_weight_val=10.0, dice_weight=0.5)
    if name == 'bce':
        return nn.BCEWithLogitsLoss()
    if name == 'dice':
        return PlainDiceLoss()
    raise ValueError(f"Unknown aux loss: {name} (expected one of: boundary, bce, dice)")


# ==============================================================================
# 2. SEED & TRAINER
# ==============================================================================
def set_seed(seed):
    import random
    import numpy as np

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def train_segmentation(model_name, loss_name, size, epochs, batch_size, lr,
                       dataset, output_path, seed, num_classes=1, dataset_type='floodvn',
                       aux_loss_name='boundary', aux_weight=0.4, pretrained=False):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")

    set_seed(seed)

    from utils.dataloader import get_dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset=dataset, batch_size=batch_size, size=size,
        seed=seed, num_classes=num_classes, dataset_type=dataset_type
    )

    print(f"[OK] DataLoaders: num_workers=4, persistent_workers=False")
    print(f"  Train: {len(train_loader)} batches (drop_last=True)")
    print(f"  Val:   {len(val_loader)} batches")
    print(f"  Test:  {len(test_loader)} batches")

    set_seed(seed)

    from models import get_model
    model = get_model(model_name, num_classes=num_classes, seed=seed, input_size=size, pretrained=pretrained)
    model = model.to(device)
    model = model.float()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[OK] Model: {model_name} (pretrained={pretrained})")
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable:    {trainable_params:,}")

    from utils.metrics import calculate_model_complexity, measure_inference_time

    print(f"\n{'='*70}")
    print("MODEL COMPLEXITY ANALYSIS (PAPER ACCURATE)")
    print(f"{'='*70}")

    complexity = calculate_model_complexity(model, input_size=(1, 3, size, size), device=device)

    print(f"Total Params (Train w/ Aux): {complexity['total_params']:,}")
    print(f"Deployable Params (Edge):   {complexity.get('deployable_params', complexity['total_params']):,}")
    print(f"Trainable Parameters:        {complexity['trainable_params']:,}")
    print(f"Model Size (Deployable):     {complexity.get('deployable_size_mb', complexity['model_size_mb']):.2f} MB")
    print(f"Peak Memory Usage:    {complexity['memory_mb']:.2f} MB")
    print(f"GMACs:                {complexity.get('gmacs', complexity['gflops']/2):.4f}")
    print(f"GFLOPs:               {complexity['gflops']:.4f}")

    print(f"\n{'='*70}")
    print("INFERENCE PERFORMANCE")
    print(f"{'='*70}")

    inference_stats = measure_inference_time(model, input_size=(1, 3, size, size), device=device)

    print(f"Average Inference Time: {inference_stats['avg_time_s']*1000:.4f} ms (± {inference_stats['std_time_s']*1000:.4f} ms)")
    print(f"Min Inference Time:     {inference_stats['min_time_s']*1000:.4f} ms")
    print(f"Max Inference Time:     {inference_stats['max_time_s']*1000:.4f} ms")
    print(f"FPS:                    {inference_stats['fps']:.2f}")
    print(f"Latency:                {inference_stats['latency_ms']:.4f} ms")
    print(f"{'='*70}\n")

    set_seed(seed)

    from losses import get_loss
    criterion = get_loss(loss_name, num_classes=num_classes)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        fused=False
    )

    print(f"[OK] Optimizer: Adam (fused=False for determinism)")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    os.makedirs(output_path, exist_ok=True)
    ds_name = os.path.basename(os.path.normpath(dataset))
    save_path = os.path.join(output_path, f'{model_name}_{loss_name}_{ds_name}_s{seed}.pth')

    best_val_loss = float('inf')

    # -------------------------------------------------------------
    # DETECT AUX BRANCH & PICK AUX LOSS
    # -------------------------------------------------------------
    model.train()
    try:
        dummy_x = torch.randn(2, 3, size, size).to(device)
        with torch.no_grad():
            dummy_out = model(dummy_x)
        has_aux = isinstance(dummy_out, (list, tuple))
    except Exception:
        has_aux = False

    aux_criterion = get_aux_loss(aux_loss_name).to(device) if (has_aux and num_classes == 1) else None
    if has_aux and num_classes > 1:
        # For multi-class models with aux head (like FloodScene), aux head outputs num_classes
        aux_criterion = criterion

    # -------------------------------------------------------------
    # LOSS CONFIGURATION LOGGER
    # -------------------------------------------------------------
    print(f"\n{'='*70}")
    print("LOSS CONFIGURATION LOGGER")
    print(f"{'='*70}")
    print(f"Main Loss : {loss_name.upper()}")
    if has_aux:
        print(f"Aux Loss  : {aux_loss_name.upper() if num_classes == 1 else loss_name.upper()}")
        print(f"  + Weight : {aux_weight}")
    else:
        print(f"Aux Loss  : None")
    print(f"{'='*70}\n")

    print(f"\n{'='*70}")
    print(f"TRAINING START - {epochs} EPOCHS")
    print(f"{'='*70}")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]",
                    leave=False, ncols=100)

        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(device, non_blocking=False)
            masks = masks.to(device, non_blocking=False)

            optimizer.zero_grad(set_to_none=False)

            outputs = model(images)

            if isinstance(outputs, (list, tuple)):
                loss = criterion(outputs[0], masks)
                if aux_criterion is not None:
                    for aux_out in outputs[1:]:
                        if aux_out is not None:
                            aux_loss = aux_criterion(aux_out, masks)
                            loss += aux_weight * aux_loss
            else:
                loss = criterion(outputs, masks)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]",
                                      leave=False, ncols=100):
                images = images.to(device, non_blocking=False)
                masks = masks.to(device, non_blocking=False)

                outputs = model(images)

                if isinstance(outputs, (list, tuple)):
                    outputs = outputs[0]

                loss = criterion(outputs, masks)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train: {avg_train_loss:.6f} | "
              f"Val: {avg_val_loss:.6f}", end='')
        scheduler.step()
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'seed': seed,
                'pytorch_version': torch.__version__,
                'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
                'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                'torch_rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                'cuda_rng_state_all': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                'complexity': complexity,
                'inference_stats': inference_stats,
                'config': {
                    'model': model_name,
                    'loss': loss_name,
                    'dataset': dataset,
                    'batch_size': batch_size,
                    'lr': lr,
                    'size': size,
                    'num_classes': num_classes,
                    'aux_loss': aux_loss_name if has_aux else None,
                    'aux_weight': aux_weight if has_aux else None,
                    'pretrained': pretrained
                }
            }

            torch.save(checkpoint, save_path, _use_new_zipfile_serialization=True)
            print(" [OK] Best")
        else:
            print()

    print(f"\n{'='*70}")
    print("TESTING & EVALUATION")
    print(f"{'='*70}")

    checkpoint = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Testing", ncols=100):
            images = images.to(device, non_blocking=False)
            masks = masks.to(device, non_blocking=False)

            outputs = model(images)

            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]

            loss = criterion(outputs, masks)
            test_loss += loss.item()

            if num_classes == 1:
                preds = torch.sigmoid(outputs)
            else:
                preds = torch.softmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(masks.cpu().numpy())

    from utils.metrics import calculate_all_metrics

    metrics = calculate_all_metrics(all_preds, all_labels, num_classes)
    miou = metrics['miou']
    dice = metrics['dice']
    pixel_acc = metrics['pixel_accuracy']
    precision = metrics['precision']
    recall = metrics['recall']
    fwiou = metrics['fwiou']
    boundary_iou = metrics['boundary_iou']

    avg_test_loss = test_loss / len(test_loader)

    print(f"\n{'='*70}")
    print("FINAL RESULTS - COMPREHENSIVE PAPER METRICS")
    print(f"{'='*70}")
    print(f"Test Loss:        {avg_test_loss:.6f}")
    print(f"mIoU:             {miou:.6f}")
    print(f"Dice / F1 Score:  {dice:.6f}")
    print(f"Pixel Accuracy:   {pixel_acc:.6f}")
    print(f"Precision:        {precision:.6f}")
    print(f"Recall:           {recall:.6f}")
    print(f"FWIoU:            {fwiou:.6f}")
    print(f"Boundary IoU:     {boundary_iou:.6f}")
    print(f"Best Val Loss:    {best_val_loss:.6f}")
    print(f"\nLaTeX Quick-Copy:")
    print(f"mIoU: ${miou:.4f}$, Dice: ${dice:.4f}$, Prec: ${precision:.4f}$, Rec: ${recall:.4f}$, B-IoU: ${boundary_iou:.4f}$")
    print(f"\n{'='*70}")
    print("MODEL COMPLEXITY & EFFICIENCY")
    print(f"{'='*70}")
    print(f"Total Params (Train):     {complexity['total_params']:,}")
    print(f"Deployable Params (Edge): {complexity.get('deployable_params', complexity['total_params']):,}")
    print(f"Deployable Model Size:    {complexity.get('deployable_size_mb', complexity['model_size_mb']):.2f} MB")
    print(f"Memory Usage:     {complexity['memory_mb']:.2f} MB")
    print(f"GMACs:            {complexity.get('gmacs', complexity['gflops']/2):.4f}")
    print(f"GFLOPs:           {complexity['gflops']:.4f}")
    print(f"\n{'='*70}")
    print("INFERENCE PERFORMANCE")
    print(f"{'='*70}")
    print(f"Avg Inference:    {inference_stats['avg_time_s']*1000:.4f} ms")
    print(f"FPS:              {inference_stats['fps']:.2f}")
    print(f"Latency:          {inference_stats['latency_ms']:.4f} ms")
    print(f"\n{'='*70}")
    print(f"Saved:            {save_path}")
    print(f"{'='*70}\n")

    return {
        'test_loss': avg_test_loss,
        'miou': miou,
        'dice': dice,
        'f1': dice,
        'pixel_accuracy': pixel_acc,
        'precision': precision,
        'recall': recall,
        'fwiou': fwiou,
        'boundary_iou': boundary_iou,
        'best_val_loss': best_val_loss,
        'model_path': save_path,
        'complexity': complexity,
        'inference_stats': inference_stats
    }
