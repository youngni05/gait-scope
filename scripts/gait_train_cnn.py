"""Step 3B: train a small 1D-CNN on MediaPipe pose sequences (train/val only).

Run from gait_project root:
    python -m pip install numpy scikit-learn torch
    python scripts/gait_train_cnn.py --data data/processed/gait_step2 --out experiments/cnn
    python scripts/gait_train_cnn.py --data data/processed/gait_step2 --out experiments/cnn/cnn_d_dropout050 --lr 0.0003 --dropout 0.50
    python scripts/gait_train_cnn.py --data data/processed/gait_step2 --out experiments/cnn/cnn_e_weightdecay001 --lr 0.0003 --weight-decay 0.01

Does not load test.npz. Train-derived feature scaling and weighted loss are used.
Input: [records, 128, 99] = 33 joints x (x, y, confidence-validity mask).
"""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class GaitCNN(nn.Module):
    def __init__(self, features=99, dropout=0.35):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(features, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(128, 2))

    def forward(self, x):
        # [batch, time, features] -> [batch, features, time]
        return self.classifier(self.encoder(x.transpose(1, 2)))


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in ("X", "y", "record_key", "original_label")}


def scale_from_train(X_train, X_val):
    """Normalize only XY channels using training data; keep mask channels [0,1]."""
    coord_columns = np.array([j for j in range(99) if j % 3 != 2])
    mean = np.zeros(99, dtype=np.float32)
    std = np.ones(99, dtype=np.float32)
    mean[coord_columns] = X_train[:, :, coord_columns].mean(axis=(0, 1))
    std[coord_columns] = X_train[:, :, coord_columns].std(axis=(0, 1))
    std[coord_columns] = np.maximum(std[coord_columns], 1e-3)

    def transform(X):
        X = (X.astype(np.float32) - mean) / std
        X[:, :, coord_columns] = np.clip(X[:, :, coord_columns], -8.0, 8.0)
        return X

    return transform(X_train), transform(X_val), mean, std


def metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "normal_recall": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "abnormal_precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "abnormal_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "abnormal_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "confusion_matrix_normal_abnormal": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


@torch.no_grad()
def predict(model, X, batch_size, device):
    model.eval()
    probabilities = []
    for start in range(0, len(X), batch_size):
        batch = torch.from_numpy(X[start:start + batch_size]).to(device)
        probabilities.append(torch.softmax(model(batch), dim=1).cpu().numpy())
    return np.concatenate(probabilities, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Train 1D-CNN for binary gait classification")
    parser.add_argument("--data", type=Path, default=Path("data/processed/gait_step2"))
    parser.add_argument("--out", type=Path, default=Path("experiments/cnn"))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1 or args.lr <= 0 or not 0 <= args.dropout < 1 or args.weight_decay < 0:
        parser.error("Invalid training hyperparameters")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train = load_npz(args.data / "train.npz")
    val = load_npz(args.data / "val.npz")
    if set(train["record_key"].tolist()) & set(val["record_key"].tolist()):
        raise ValueError("Train/validation record keys overlap")
    if train["X"].shape[1:] != (128, 99) or val["X"].shape[1:] != (128, 99):
        raise ValueError("Expected pose arrays shaped [N,128,99]")
    if not np.isfinite(train["X"]).all() or not np.isfinite(val["X"]).all():
        raise ValueError("Non-finite features found")
    if set(np.unique(train["y"])) != {0, 1} or set(np.unique(val["y"])) != {0, 1}:
        raise ValueError("Expected both binary labels in train and validation")

    X_train, X_val, mean, std = scale_from_train(train["X"], val["X"])
    y_train, y_val = train["y"].astype(np.int64), val["y"].astype(np.int64)
    label_counts = np.bincount(y_train, minlength=2)
    class_weights = len(y_train) / (2.0 * label_counts)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    model = GaitCNN(dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=loader_generator, num_workers=0)

    args.out.mkdir(parents=True, exist_ok=True)
    history = []
    best_rank = (-1.0, -1.0)
    best_epoch, stale_epochs = 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_X)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_y)

        probabilities = predict(model, X_val, args.batch_size, device)
        predictions = probabilities.argmax(axis=1)
        score = metrics(y_val, predictions)
        rank = (score["macro_f1"], score["balanced_accuracy"])
        history.append({"epoch": epoch, "train_loss": total_loss / len(y_train), **{k: v for k, v in score.items() if not isinstance(v, list)}})

        if rank > best_rank:
            best_rank, best_epoch, stale_epochs = rank, epoch, 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "feature_mean": torch.from_numpy(mean),
                "feature_std": torch.from_numpy(std),
                "epoch": epoch,
                "model_config": {"features": 99, "architecture": "GaitCNN", "dropout": args.dropout},
                "seed": args.seed,
                "training_hyperparameters": {"lr": args.lr, "weight_decay": args.weight_decay, "batch_size": args.batch_size, "patience": args.patience},
                "label_map": {"normal": 0, "abnormal": 1},
            }, args.out / "best_cnn.pt")
        else:
            stale_epochs += 1

        print(f"Epoch {epoch:03d} | loss={total_loss/len(y_train):.4f} | val_macro_f1={score['macro_f1']:.4f} | val_bal_acc={score['balanced_accuracy']:.4f}")
        if stale_epochs >= args.patience:
            print(f"Early stopping after {epoch} epochs (best epoch: {best_epoch})")
            break

    # Restore the best validation checkpoint, not the final epoch.
    checkpoint = torch.load(args.out / "best_cnn.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    probabilities = predict(model, X_val, args.batch_size, device)
    pred = probabilities.argmax(axis=1)
    final_scores = metrics(y_val, pred)

    with (args.out / "training_history.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    with (args.out / "validation_predictions.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["record_key", "original_label", "true_binary", "predicted_binary", "prob_abnormal"])
        for key, label, actual, predicted, probability in zip(val["record_key"], val["original_label"], y_val, pred, probabilities[:, 1]):
            writer.writerow([str(key), str(label), int(actual), int(predicted), float(probability)])

    summary = {
        "model": "1D-CNN", "train_records": len(y_train), "validation_records": len(y_val),
        "feature_shape": [128, 99], "device": str(device), "seed": args.seed,
        "hyperparameters": {"lr": args.lr, "dropout": args.dropout, "weight_decay": args.weight_decay, "batch_size": args.batch_size, "patience": args.patience, "max_epochs": args.epochs},
        "class_weights_normal_abnormal": class_weights.tolist(),
        "selection_criterion": "maximum validation macro_f1, tie break balanced_accuracy",
        "best_epoch": best_epoch, "epochs_run": len(history), "validation_metrics": final_scores,
        "architecture": f"Conv1d(99,64,k5)-BN-ReLU-MaxPool-Conv1d(64,128,k5)-BN-ReLU-GlobalAvgPool-Dropout({args.dropout})-Linear(128,2)",
        "warning": "Test split was not read. Record-level validation does not ensure participant-independent performance.",
    }
    (args.out / "validation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nFINAL VALIDATION SUMMARY")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved to: {args.out.resolve()}")


if __name__ == "__main__":
    main()
