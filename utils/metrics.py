import torch
import numpy as np
import cv2
import time


def _prepare_masks(all_preds, all_labels, num_classes, threshold=0.5):
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    if num_classes == 1:
        if all_preds.ndim == 4:
            all_preds = all_preds.squeeze(1)
        if all_labels.ndim == 4:
            all_labels = all_labels.squeeze(1)

        preds_binary = (all_preds > threshold).astype(np.uint8)
        labels_binary = (all_labels > threshold).astype(np.uint8)
        return preds_binary, labels_binary
    else:
        preds_class = np.argmax(all_preds, axis=1)
        if all_labels.ndim == 4:
            all_labels = all_labels.squeeze(1)
        labels_class = all_labels.astype(np.int64)
        return preds_class, labels_class


def calculate_miou(all_preds, all_labels, num_classes, threshold=0.5):
    preds, labels = _prepare_masks(all_preds, all_labels, num_classes, threshold)

    if num_classes == 1:
        preds_flat = preds.reshape(-1)
        labels_flat = labels.reshape(-1)

        intersection = np.logical_and(preds_flat, labels_flat).sum()
        union = np.logical_or(preds_flat, labels_flat).sum()

        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        return float(intersection / union)
    else:
        preds_flat = preds.reshape(-1)
        labels_flat = labels.reshape(-1)

        ious = []
        for cls in range(num_classes):
            pred_mask = (preds_flat == cls)
            label_mask = (labels_flat == cls)

            intersection = np.logical_and(pred_mask, label_mask).sum()
            union = np.logical_or(pred_mask, label_mask).sum()

            if union > 0:
                ious.append(float(intersection / union))

        return float(np.mean(ious)) if ious else 0.0


def calculate_dice_score(all_preds, all_labels, num_classes, threshold=0.5):
    preds, labels = _prepare_masks(all_preds, all_labels, num_classes, threshold)

    if num_classes == 1:
        preds_flat = preds.reshape(-1)
        labels_flat = labels.reshape(-1)

        intersection = np.logical_and(preds_flat, labels_flat).sum()
        dice = (2.0 * intersection) / (preds_flat.sum() + labels_flat.sum() + 1e-8)
        return float(dice)
    else:
        preds_flat = preds.reshape(-1)
        labels_flat = labels.reshape(-1)

        dice_scores = []
        for cls in range(num_classes):
            pred_mask = (preds_flat == cls)
            label_mask = (labels_flat == cls)

            intersection = np.logical_and(pred_mask, label_mask).sum()
            dice = (2.0 * intersection) / (pred_mask.sum() + label_mask.sum() + 1e-8)

            if pred_mask.sum() > 0 or label_mask.sum() > 0:
                dice_scores.append(float(dice))

        return float(np.mean(dice_scores)) if dice_scores else 0.0


def calculate_pixel_accuracy(all_preds, all_labels, num_classes, threshold=0.5):
    preds, labels = _prepare_masks(all_preds, all_labels, num_classes, threshold)
    preds_flat = preds.reshape(-1)
    labels_flat = labels.reshape(-1)

    correct = (preds_flat == labels_flat).sum()
    total = all_preds_flat_size = preds_flat.size
    return float(correct / total)


def calculate_precision(all_preds, all_labels, num_classes, threshold=0.5):
    preds, labels = _prepare_masks(all_preds, all_labels, num_classes, threshold)

    if num_classes == 1:
        preds_flat = preds.reshape(-1)
        labels_flat = labels.reshape(-1)

        tp = np.logical_and(preds_flat == 1, labels_flat == 1).sum()
        fp = np.logical_and(preds_flat == 1, labels_flat == 0).sum()

        if tp + fp == 0:
            return 1.0 if labels_flat.sum() == 0 else 0.0
        return float(tp / (tp + fp))
    else:
        preds_flat = preds.reshape(-1)
        labels_flat = labels.reshape(-1)

        precisions = []
        for cls in range(num_classes):
            tp = np.logical_and(preds_flat == cls, labels_flat == cls).sum()
            fp = np.logical_and(preds_flat == cls, labels_flat != cls).sum()
            if tp + fp > 0:
                precisions.append(float(tp / (tp + fp)))

        return float(np.mean(precisions)) if precisions else 0.0


def calculate_recall(all_preds, all_labels, num_classes, threshold=0.5):
    preds, labels = _prepare_masks(all_preds, all_labels, num_classes, threshold)

    if num_classes == 1:
        preds_flat = preds.reshape(-1)
        labels_flat = labels.reshape(-1)

        tp = np.logical_and(preds_flat == 1, labels_flat == 1).sum()
        fn = np.logical_and(preds_flat == 0, labels_flat == 1).sum()

        if tp + fn == 0:
            return 1.0 if preds_flat.sum() == 0 else 0.0
        return float(tp / (tp + fn))
    else:
        preds_flat = preds.reshape(-1)
        labels_flat = labels.reshape(-1)

        recalls = []
        for cls in range(num_classes):
            tp = np.logical_and(preds_flat == cls, labels_flat == cls).sum()
            fn = np.logical_and(preds_flat != cls, labels_flat == cls).sum()
            if tp + fn > 0:
                recalls.append(float(tp / (tp + fn)))

        return float(np.mean(recalls)) if recalls else 0.0


def calculate_fwiou(all_preds, all_labels, num_classes, threshold=0.5):
    """
    Frequency Weighted Intersection over Union (FWIoU):
    Weights each class IoU by frequency of ground truth pixels.
    """
    preds, labels = _prepare_masks(all_preds, all_labels, num_classes, threshold)
    preds_flat = preds.reshape(-1)
    labels_flat = labels.reshape(-1)

    total_pixels = labels_flat.size
    if total_pixels == 0:
        return 0.0

    if num_classes == 1:
        # Binary: class 0 (background) and class 1 (flood)
        n0 = (labels_flat == 0).sum()
        n1 = (labels_flat == 1).sum()

        u0 = np.logical_or(preds_flat == 0, labels_flat == 0).sum()
        i0 = np.logical_and(preds_flat == 0, labels_flat == 0).sum()
        iou0 = float(i0 / u0) if u0 > 0 else 0.0

        u1 = np.logical_or(preds_flat == 1, labels_flat == 1).sum()
        i1 = np.logical_and(preds_flat == 1, labels_flat == 1).sum()
        iou1 = float(i1 / u1) if u1 > 0 else 0.0

        return float((n0 * iou0 + n1 * iou1) / total_pixels)
    else:
        weighted_iou_sum = 0.0
        for cls in range(num_classes):
            n_cls = (labels_flat == cls).sum()
            if n_cls > 0:
                pred_mask = (preds_flat == cls)
                label_mask = (labels_flat == cls)
                intersection = np.logical_and(pred_mask, label_mask).sum()
                union = np.logical_or(pred_mask, label_mask).sum()
                iou = float(intersection / union) if union > 0 else 0.0
                weighted_iou_sum += n_cls * iou

        return float(weighted_iou_sum / total_pixels)


def calculate_boundary_iou(all_preds, all_labels, num_classes, threshold=0.5, dilation=3):
    """
    Boundary IoU (Bowen et al., CVPR 2021):
    Measures contour alignment between predicted mask and ground truth mask.
    """
    preds, labels = _prepare_masks(all_preds, all_labels, num_classes, threshold)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilation, dilation))

    n_samples = preds.shape[0]
    sample_ious = []

    for i in range(n_samples):
        p = preds[i].astype(np.uint8)
        l = labels[i].astype(np.uint8)

        if num_classes == 1:
            p_bin = (p == 1).astype(np.uint8)
            l_bin = (l == 1).astype(np.uint8)

            p_bound = cv2.dilate(p_bin, kernel) - cv2.erode(p_bin, kernel)
            l_bound = cv2.dilate(l_bin, kernel) - cv2.erode(l_bin, kernel)

            inter = np.logical_and(p_bound, l_bound).sum()
            union = np.logical_or(p_bound, l_bound).sum()

            if union > 0:
                sample_ious.append(float(inter / union))
            elif l_bin.sum() == 0 and p_bin.sum() == 0:
                sample_ious.append(1.0)
        else:
            cls_ious = []
            for cls in range(num_classes):
                p_c = (p == cls).astype(np.uint8)
                l_c = (l == cls).astype(np.uint8)

                if l_c.sum() > 0 or p_c.sum() > 0:
                    p_bound = cv2.dilate(p_c, kernel) - cv2.erode(p_c, kernel)
                    l_bound = cv2.dilate(l_c, kernel) - cv2.erode(l_c, kernel)
                    inter = np.logical_and(p_bound, l_bound).sum()
                    union = np.logical_or(p_bound, l_bound).sum()
                    if union > 0:
                        cls_ious.append(float(inter / union))
            if cls_ious:
                sample_ious.append(float(np.mean(cls_ious)))

    return float(np.mean(sample_ious)) if sample_ious else 0.0


def calculate_all_metrics(all_preds, all_labels, num_classes=1, threshold=0.5):
    """
    Compute comprehensive paper-ready metrics in one call.
    """
    miou = calculate_miou(all_preds, all_labels, num_classes, threshold)
    dice = calculate_dice_score(all_preds, all_labels, num_classes, threshold)
    pixel_acc = calculate_pixel_accuracy(all_preds, all_labels, num_classes, threshold)
    precision = calculate_precision(all_preds, all_labels, num_classes, threshold)
    recall = calculate_recall(all_preds, all_labels, num_classes, threshold)
    fwiou = calculate_fwiou(all_preds, all_labels, num_classes, threshold)
    boundary_iou = calculate_boundary_iou(all_preds, all_labels, num_classes, threshold)

    return {
        'miou': miou,
        'dice': dice,
        'f1': dice,
        'pixel_accuracy': pixel_acc,
        'precision': precision,
        'recall': recall,
        'fwiou': fwiou,
        'boundary_iou': boundary_iou,
    }


def calculate_model_complexity(model, input_size=(1, 3, 512, 512), device='cuda'):
    """
    Accurately calculates total params, trainable params, memory allocation,
    GMACs, and GFLOPs (taking into account Conv2d groups for depthwise convolutions).
    """
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    deployable_params = sum(p.numel() for name, p in model.named_parameters() if 'aux' not in name.lower())
    model_size_mb = total_params * 4 / (1024 ** 2)
    deployable_size_mb = deployable_params * 4 / (1024 ** 2)

    device_obj = torch.device(device)
    dummy_input = torch.randn(input_size, device=device_obj)

    if torch.cuda.is_available() and 'cuda' in str(device):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    with torch.no_grad():
        _ = model(dummy_input)

    if torch.cuda.is_available() and 'cuda' in str(device):
        torch.cuda.synchronize()
        memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        memory_allocated = 0.0

    total_macs = 0
    hooks = []

    def count_ops_hook(module, input, output):
        nonlocal total_macs
        if isinstance(module, torch.nn.Conv2d):
            batch_size = input[0].size(0)
            output_height = output.size(2)
            output_width = output.size(3)
            # ✓ FIXED: Divide by module.groups for accurate Depthwise Separable Conv FLOPs
            cin_per_group = module.in_channels // module.groups
            kernel_ops = module.kernel_size[0] * module.kernel_size[1] * cin_per_group
            output_ops = output_height * output_width * module.out_channels
            total_macs += batch_size * kernel_ops * output_ops
        elif isinstance(module, torch.nn.Linear):
            batch_size = input[0].size(0)
            total_macs += batch_size * module.in_features * module.out_features

    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            hooks.append(module.register_forward_hook(count_ops_hook))

    with torch.no_grad():
        _ = model(dummy_input)

    for hook in hooks:
        hook.remove()

    gmacs = total_macs / 1e9
    # Standard deep learning convention: 1 MAC = 2 FLOPs (1 multiply + 1 accumulate)
    gflops = (total_macs * 2) / 1e9

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'deployable_params': deployable_params,
        'model_size_mb': float(model_size_mb),
        'deployable_size_mb': float(deployable_size_mb),
        'memory_mb': float(memory_allocated),
        'gmacs': float(gmacs),
        'gflops': float(gflops),
        'macs_gflops': float(gmacs),  # backwards compatibility
    }


def measure_inference_time(model, input_size=(1, 3, 512, 512), device='cuda', warmup=10, iterations=100):
    model.eval()
    device_obj = torch.device(device)
    dummy_input = torch.randn(input_size, device=device_obj)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    if torch.cuda.is_available() and 'cuda' in str(device):
        torch.cuda.synchronize()

    times = []

    with torch.no_grad():
        for _ in range(iterations):
            if torch.cuda.is_available() and 'cuda' in str(device):
                torch.cuda.synchronize()

            start = time.perf_counter()
            _ = model(dummy_input)

            if torch.cuda.is_available() and 'cuda' in str(device):
                torch.cuda.synchronize()

            end = time.perf_counter()
            times.append(end - start)

    times = np.array(times)

    avg_time = float(np.mean(times))
    std_time = float(np.std(times))
    min_time = float(np.min(times))
    max_time = float(np.max(times))

    fps = float(1.0 / avg_time) if avg_time > 0 else 0.0
    latency_ms = float(avg_time * 1000)

    return {
        'avg_time_s': avg_time,
        'std_time_s': std_time,
        'min_time_s': min_time,
        'max_time_s': max_time,
        'fps': fps,
        'latency_ms': latency_ms,
    }
