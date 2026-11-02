import os
import argparse
import json
import time
import torch
import numpy as np

from models import get_model
from utils.dataloader import get_dataloaders
from utils.metrics import calculate_all_metrics


def evaluate_cross_dataset(model_name, weights_path, source_dataset, target_datasets,
                           size=128, batch_size=4, num_classes=1, seed=42, output_dir='outputs',
                           download=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 75)
    print("GENERALIZATION BENCHMARK: IN-DOMAIN, CROSS-DOMAIN & EXTERNAL TEST")
    print("=" * 75)
    print(f"Model Architecture    : {model_name}")
    print(f"Trained Checkpoint    : {weights_path}")
    print(f"Source Domain (Train) : {source_dataset.upper()}")
    print(f"Target Domains (Test) : {[d.upper() for d in target_datasets]}")
    print(f"Inference Device      : {device}")
    print("=" * 75 + "\n")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Checkpoint not found at: '{weights_path}'")

    if download:
        from benchmark import download_dataset
        for ds in target_datasets:
            if not os.path.exists(ds):
                print(f"[*] Target dataset '{ds}' not found on disk. Initiating download...")
                download_dataset(ds)

    # Initialize model and load weights
    model = get_model(model_name, num_classes=num_classes, seed=seed, input_size=size)
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    domain_results = {}

    for target_ds in target_datasets:
        if target_ds.lower() == source_dataset.lower():
            eval_tag = "IN-DOMAIN"
        elif target_ds.lower() == 'floodvn':
            eval_tag = "EXTERNAL TEST"
        else:
            eval_tag = "CROSS-DOMAIN"

        print(f"\n>>> [{eval_tag}] Evaluating: {source_dataset.upper()} -> {target_ds.upper()}...")
        try:
            _, _, test_loader = get_dataloaders(
                dataset=target_ds, batch_size=batch_size, size=size,
                seed=seed, num_classes=num_classes, dataset_type=target_ds
            )
        except Exception as e:
            print(f"[!] Could not load dataset '{target_ds}': {e}")
            continue

        all_preds = []
        all_labels = []
        inference_times = []

        with torch.no_grad():
            for images, masks in test_loader:
                images = images.to(device)
                
                start_t = time.perf_counter()
                outputs = model(images)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                inference_times.append(time.perf_counter() - start_t)

                if isinstance(outputs, (list, tuple)):
                    outputs = outputs[0]

                if num_classes == 1:
                    preds = torch.sigmoid(outputs)
                else:
                    preds = torch.softmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(masks.cpu().numpy())

        metrics = calculate_all_metrics(all_preds, all_labels, num_classes)
        avg_latency = float(np.mean(inference_times) / batch_size * 1000)
        metrics['latency_ms'] = avg_latency
        metrics['fps'] = float(1000.0 / avg_latency if avg_latency > 0 else 0)

        domain_results[target_ds] = {
            'eval_type': eval_tag,
            'metrics': metrics
        }

        print(f"    mIoU:         {metrics['miou']:.4f}")
        print(f"    Dice / F1:    {metrics['dice']:.4f}")
        print(f"    Precision:    {metrics['precision']:.4f}")
        print(f"    Recall:       {metrics['recall']:.4f}")
        print(f"    Pixel Acc:    {metrics['pixel_accuracy']:.4f}")
        print(f"    Boundary IoU: {metrics['boundary_iou']:.4f}")
        print(f"    Latency:      {avg_latency:.2f} ms ({metrics['fps']:.1f} FPS)")

    print("\n" + "=" * 80)
    print("CROSS-DATASET GENERALIZATION SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Target Domain':<15} | {'Type':<15} | {'mIoU':<8} | {'Dice':<8} | {'Precision':<10} | {'Recall':<8} | {'B-IoU':<8}")
    print("-" * 80)
    for ds_name, res in domain_results.items():
        m = res['metrics']
        print(f"{ds_name.upper():<15} | {res['eval_type']:<15} | {m['miou']:<8.4f} | {m['dice']:<8.4f} | {m['precision']:<10.4f} | {m['recall']:<8.4f} | {m['boundary_iou']:<8.4f}")
    print("=" * 80 + "\n")

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f'{model_name}_cross_dataset_{source_dataset}.json')
    with open(out_file, 'w') as f:
        json.dump({
            'model': model_name,
            'source_dataset': source_dataset,
            'weights': weights_path,
            'input_size': size,
            'results': domain_results
        }, f, indent=2)
    print(f"[OK] Generalization benchmark saved: {out_file}")
    return domain_results


def main():
    parser = argparse.ArgumentParser(description="Cross-Dataset Generalization Evaluation Suite")
    parser.add_argument('--model',           type=str, default='mambafloodlite', help="Model name")
    parser.add_argument('--weights',         type=str, required=True,            help="Path to trained .pth weights")
    parser.add_argument('--source_dataset',  type=str, default='floodkaggle',    help="Source domain used in training")
    parser.add_argument('--target_datasets', type=str, nargs='+', default=['floodkaggle', 'aifloodsense'],
                        help="List of target datasets to evaluate on")
    parser.add_argument('--size',            type=int, default=128,              help="Input dimension")
    parser.add_argument('--batch_size',      type=int, default=4)
    parser.add_argument('--num_classes',     type=int, default=1)
    parser.add_argument('--seed',            type=int, default=42)
    parser.add_argument('--output_dir',      type=str, default='outputs')
    parser.add_argument('--download',        action='store_true',                help="Download missing datasets")
    args = parser.parse_args()

    evaluate_cross_dataset(
        model_name=args.model,
        weights_path=args.weights,
        source_dataset=args.source_dataset,
        target_datasets=args.target_datasets,
        size=args.size,
        batch_size=args.batch_size,
        num_classes=args.num_classes,
        seed=args.seed,
        output_dir=args.output_dir,
        download=args.download
    )


if __name__ == '__main__':
    main()
