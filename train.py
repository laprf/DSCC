import argparse
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from models.modeling import model_init
from dataset import load_data
from utils import set_seed, gen_confusion_matrix, eval_metrics


def train(model, optimizer, criterion, data_loader):
    model.train()

    for iteration, (inputs, labels, _) in enumerate(data_loader):
        inputs, labels = inputs.to(DEVICE), labels.long().to(DEVICE)
        optimizer.zero_grad()

        predictions, train_labels, merge_loss = model(inputs, labels)

        # Compute token loss
        loss = criterion(predictions.permute(0, 2, 1), train_labels) + merge_loss
        loss.backward()
        optimizer.step()


def validate(model, validation_loader):
    with torch.no_grad():
        model.eval()

        # Pre-compute shared values
        num_classes = NUM_CLASSES
        confusion_matrix = torch.zeros([NUM_CLASSES, NUM_CLASSES], device=DEVICE)

        for inputs, labels, _ in validation_loader:
            inputs, labels = inputs.to(DEVICE), labels.long().to(DEVICE)

            # Forward pass
            predictions, supertoken_map = model(inputs, labels)

            batch_size, num_pixels = supertoken_map.size(0), supertoken_map.size(1) * supertoken_map.size(2)
            supertoken_indices = torch.arange(predictions.size(1), device=DEVICE)
            expanded_supertoken_map = supertoken_map.view(batch_size, num_pixels, 1).expand(-1, -1, predictions.size(1))
            mask = (expanded_supertoken_map == supertoken_indices).float().to(predictions.dtype)
            saliency_results = torch.bmm(mask, predictions).argmax(dim=-1)
            confusion_matrix += gen_confusion_matrix(num_classes, saliency_results.flatten(), labels.flatten())

        confusion_matrix_np = confusion_matrix.cpu().numpy()
        return eval_metrics(confusion_matrix_np, mode='val')



def main():
    """Main training and validation loop."""
    # Initialize variables
    min_f1_score = 0.0
    model, optimizer, criterion, scheduler = model_init(cfg, args)
    model = model.to(DEVICE)

    # Data loaders
    train_loader = load_data(args, cfg['img_size'], 'tr')
    val_loader = load_data(args, cfg['img_size'], 'val')

    # Training and validation loop
    with tqdm(total=args.epoch_num, desc="Training Progress") as progress_bar:
        for epoch in range(args.epoch_num):
            train(model, optimizer, criterion, train_loader)
            val_f1 = validate(model, val_loader)
            scheduler.step()

            # Save the best model
            if val_f1 > min_f1_score:
                min_f1_score = val_f1
                model_save_path = EXPERIMENT_DIR / "best_model.pth"
                torch.save(model.state_dict(), model_save_path)

            progress_bar.update(1)


if __name__ == '__main__':
    # Argument parser
    parser = argparse.ArgumentParser(description="PyTorch WHU_OHS Training Script")
    parser.add_argument("--config", default="models/yamls/best.yaml", type=str,
                        help="Path to the configuration file (default: config.yaml)")
    parser.add_argument("--exp_name", type=str, default='exp', help="Experiment name")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory of the dataset")
    parser.add_argument("--split_dir", default="txts", type=str,
                        help="Directory containing dataset split files")
    parser.add_argument("--output_root", default="DataStorage", type=str,
                        help="Root directory for training artifacts")
    parser.add_argument("--seed", default=233, type=int, help="Random seed")
    parser.add_argument("--device", default="cuda", type=str,
                        help="Device for training")
    parser.add_argument("--num_workers", default=4, type=int,
                        help="Number of dataloader worker processes")
    parser.add_argument("--batch_size", default=12, type=int, help="Mini-batch size (default: 8)")
    parser.add_argument("--epoch_num", default=150, type=int, help="Number of training epochs (default: 100)")
    parser.add_argument("--lr", default=1e-4, type=float, help="Initial learning rate (default: 5e-4)")
    parser.add_argument("--gpu", default="0", type=str, help="GPU device number")
    args = parser.parse_args()

    DEVICE = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    config_path = Path(args.config)
    EXPERIMENT_DIR = Path(args.output_root) / args.exp_name

    # Load configuration
    with config_path.open("r") as config_file:
        cfg = yaml.load(config_file, Loader=yaml.FullLoader)

    # Create necessary directories
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    (EXPERIMENT_DIR / "config.yaml").write_text(config_path.read_text())

    # Initialize global variables
    NUM_CLASSES = cfg["num_classes"]
    set_seed(args.seed)

    # Start main function
    main()
