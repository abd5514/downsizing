"""BiLSTM with additive attention over the YOLO feature sequence.

The rule engine handles microsleep. This model exists to read the slower,
contextual signal - the difference between an alert driver and one who is
yawning - from a window of frames rather than from any single frame.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from drowsim.config import FEATURE_DIM, SEQ_LEN


class Attention(nn.Module):
    """Additive (Bahdanau-style) attention that collapses the time axis.

    Without it every frame in the window contributes equally, which blurs the
    onset of a yawn across the frames on either side of it.
    """

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, H)
        weights = F.softmax(self.score(x).squeeze(-1), dim=1)   # (B, T)
        context = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (B, H)
        return context, weights


class BiLSTMAttention(nn.Module):
    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        hidden: int = 96,
        num_classes: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden, num_layers=1, batch_first=True, bidirectional=True
        )
        self.attention = Attention(hidden * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        out, _ = self.lstm(x)                 # (B, T, 2H)
        context, weights = self.attention(out)
        logits = self.fc(self.dropout(context))
        if return_attention:
            return logits, weights
        return logits

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FocalLoss(nn.Module):
    """Focal loss, as used in the original DrowSim training setup.

    FL3D is roughly 7:1.6:1 alert:microsleep:yawning. Focal loss down-weights
    the easy majority-class frames instead of applying one fixed weight per
    class, which is what plain weighted cross-entropy would do.
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, reduction="none")
        pt = torch.exp(-ce)
        return (self.alpha * (1.0 - pt) ** self.gamma * ce).mean()


def build_model(**kwargs) -> BiLSTMAttention:
    return BiLSTMAttention(**kwargs)


if __name__ == "__main__":
    m = build_model()
    x = torch.randn(4, SEQ_LEN, FEATURE_DIM)
    y = m(x)
    print(f"input  {tuple(x.shape)}")
    print(f"output {tuple(y.shape)}")
    print(f"trainable parameters: {m.num_parameters:,}")
