import argparse

import torch
import yaml
from tqdm import tqdm

from dataset import load_data
from models.modeling import model_init
from utils import (
    gen_confusion_matrix,
    eval_metrics,
    set_seed
)


def test(model, test_loader, device, num_classes):
    """Run testing on the model."""
    with torch.no_grad():
        model.eval()
        confusion_matrix = torch.zeros([num_classes, num_classes], device=device)

        for data, label, _ in tqdm(test_loader):
            data, label = data.to(device), label.long().to(device)

            predictions, superpixel_map = model(data, label)

            supertoken_indices = torch.arange(predictions.size(1), device=device)

            for i in range(predictions.size(0)):
                mask = (superpixel_map[i].unsqueeze(-1) == supertoken_indices).float()

                saliency_result = (mask @ predictions[i]).argmax(dim=-1)

                # Update confusion matrix
                confusion_matrix += gen_confusion_matrix(num_classes, saliency_result, label[i])

        return confusion_matrix.cpu().numpy()


def main():
    # Load configuration
    cfg = yaml.safe_load(open(args.config, "r"))

    # Device setup
    device_name = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    num_classes = cfg["num_classes"]

    # Set seed for reproducibility
    set_seed(args.seed)

    # Load model
    model = model_init(cfg)[0]
    model = model.to(device)
    model.load_state_dict(torch.load(args.pretrained_model), strict=False)

    # Load test data
    test_loader = load_data(args, cfg['img_size'], "ts")

    # Run test and evaluate metrics
    confusion_matrix = test(model, test_loader, device, num_classes)
    mean_f1, oa, kappa, miou, class_f1 = eval_metrics(confusion_matrix, mode="ts")

    # Print metrics
    print(f"Mean F1: {mean_f1:.4f}, OA: {oa:.4f}, Kappa: {kappa:.4f}, mIoU: {miou:.4f}")
    print(f"Class F1: {class_f1}")


if __name__ == '__main__':
    """Main entry point for testing."""
    parser = argparse.ArgumentParser(description="PyTorch WHU_OHS Dataset Test")
    parser.add_argument("--config", default="models/yamls/best.yaml", type=str, help="Config file path")
    parser.add_argument("--data_root", type=str, required=True, help="Data root directory")
    parser.add_argument("--split_dir", default="txts", type=str, help="Split directory")
    parser.add_argument("--seed", default=233, type=int, help="Random seed")
    parser.add_argument("--device", default="cuda", type=str, help="Device for inference")
    parser.add_argument("--batch_size", default=12, type=int, help="Mini-batch size")
    parser.add_argument("--pretrained_model", type=str, required=True, help="Path to the pretrained model")
    args = parser.parse_args()

    main()
