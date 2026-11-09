"""Train the BiLSTM-Attention temporal classifier on YOLO feature sequences.

Input is whatever scripts/export_sequences.py produced: windows of SEQ_LEN
frames, each frame the seven-float vector the client actually transmits. The
model therefore learns from exactly the representation it will see at runtime.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

from drowsim.config import CLASSES
from drowsim.model import FocalLoss, build_model


def load_split(path: Path, split: str):
    data = np.load(path / f"{split}.npz")
    x = torch.from_numpy(data["x"]).float()
    y = torch.from_numpy(data["y"]).long()
    return TensorDataset(x, y), y.numpy()


def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(y.numpy())
    return np.concatenate(preds), np.concatenate(targets)


def main() -> int:
    p = argparse.ArgumentParser(description="Train BiLSTM-Attention")
    p.add_argument("--data", default="data/sequences")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--out", default="models/bilstm_attention.pt")
    args = p.parse_args()

    root = Path(args.data)
    if not (root / "train.npz").exists():
        print(f"{root}/train.npz not found - run scripts/export_sequences.py first")
        return 1

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    train_ds, train_y = load_split(root, "train")
    val_ds, _ = load_split(root, "val")
    test_ds, _ = load_split(root, "test")

    print(f"[bilstm] device={device}")
    print(f"[bilstm] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    print(f"[bilstm] class counts (train): {np.bincount(train_y, minlength=3).tolist()}")

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch)
    test_dl = DataLoader(test_ds, batch_size=args.batch)

    model = build_model(hidden=args.hidden).to(device)
    print(f"[bilstm] trainable parameters: {model.num_parameters:,}")

    # Focal loss rather than weighted cross-entropy, matching the original
    # DrowSim setup: FL3D is roughly 7:1.6:1 alert:microsleep:yawning.
    criterion = FocalLoss(alpha=1.0, gamma=2.0)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=5)

    best_macro_f1, best_state, patience = 0.0, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimiser.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimiser.step()
            total += loss.item() * len(x)

        preds, targets = evaluate(model, val_dl, device)
        report = classification_report(
            targets, preds, target_names=CLASSES, output_dict=True, zero_division=0
        )
        macro_f1 = report["macro avg"]["f1-score"]
        scheduler.step(macro_f1)

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}  loss {total / len(train_ds):.4f}  "
                  f"val macro-F1 {macro_f1:.4f}  (best {best_macro_f1:.4f})")
        if patience >= 15:
            print(f"  early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    preds, targets = evaluate(model, test_dl, device)

    # Video-disjoint, not person-disjoint: FL3D exposes video ids only, and
    # NITYMED recorded several videos per driver.
    print("\n=== Test set (video-disjoint, unseen videos) ===")
    print(classification_report(targets, preds, target_names=CLASSES,
                                digits=4, zero_division=0))
    print("confusion matrix (rows = true):")
    print(confusion_matrix(targets, preds))

    report = classification_report(targets, preds, target_names=CLASSES,
                                   output_dict=True, zero_division=0)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "hidden": args.hidden,
                "classes": list(CLASSES)}, args.out)
    Path("runs").mkdir(exist_ok=True)
    Path("runs/bilstm_test_report.json").write_text(json.dumps(report, indent=2))
    print(f"\n[bilstm] weights -> {args.out}")
    print(f"[bilstm] metrics -> runs/bilstm_test_report.json")
    print(f"[bilstm] accuracy {report['accuracy']:.4f}  "
          f"macro-F1 {report['macro avg']['f1-score']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
