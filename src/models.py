"""Fusion models for CrisisMMD humanitarian classification."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EarlyFusionMLP(nn.Module):
    """Concatenate pooled image and text features and classify."""

    def __init__(
        self, image_dim=1280, text_dim=768, hidden=256, dropout=0.3,
        n_classes=5,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(image_dim + text_dim + 1, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, image, text, missing):
        """Return class logits for a batch of image-text pairs."""
        x = torch.cat([image, text, missing.unsqueeze(1)], dim=1)
        return self.net(x)


class ModalityHead(nn.Module):
    """A small classifier for one modality."""

    def __init__(self, in_dim, hidden=256, dropout=0.3, n_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        """Return class logits."""
        return self.net(x)


class SingleModalityMLP(nn.Module):
    """A one-modality MLP control with the same head size as fusion."""

    def __init__(self, in_dim, hidden=256, dropout=0.2, n_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        """Return class logits."""
        return self.net(x)


class DeepSingleModalityMLP(nn.Module):
    """A two-hidden-layer single-modality capacity control."""

    def __init__(
        self, in_dim, hidden1=768, hidden2=256, dropout=0.2,
        n_classes=5,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, n_classes),
        )

    def forward(self, x):
        """Return class logits."""
        return self.net(x)


class LateFusionModel(nn.Module):
    """Weighted average of image and text class probabilities."""

    def __init__(
        self, image_dim=1280, text_dim=768, hidden=256, dropout=0.3,
        n_classes=5,
    ):
        super().__init__()
        self.image_head = ModalityHead(
            image_dim, hidden, dropout, n_classes
        )
        self.text_head = ModalityHead(
            text_dim, hidden, dropout, n_classes
        )
        self.logit_weight = nn.Parameter(torch.tensor(0.0))

    def forward(self, image, text, missing):
        """Return combined log-probabilities and modality logits."""
        image_logits = self.image_head(image)
        text_logits = self.text_head(text)
        weight = torch.sigmoid(self.logit_weight)
        weight = weight * (1.0 - missing)
        image_prob = F.softmax(image_logits, dim=1)
        text_prob = F.softmax(text_logits, dim=1)
        prob = weight.unsqueeze(1) * image_prob
        prob = prob + (1.0 - weight).unsqueeze(1) * text_prob
        return prob.clamp_min(1e-8).log(), image_logits, text_logits


class CrossAttentionBlock(nn.Module):
    """A transformer block with cross-attention and a feed-forward net."""

    def __init__(self, hidden=256, heads=4, dropout=0.2):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            hidden, heads, dropout=dropout, batch_first=True
        )
        self.norm_attn = nn.LayerNorm(hidden)
        self.norm_ff = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, 4 * hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden, hidden),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value, key_padding_mask):
        """Attend from query tokens to key/value tokens."""
        attn, _ = self.attn(
            query, key_value, key_value,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.norm_attn(query + self.dropout(attn))
        x = self.norm_ff(x + self.dropout(self.ff(x)))
        return x


class CrossAttentionFusion(nn.Module):
    """Bidirectional cross-attention over image and text tokens."""

    def __init__(
        self, image_dim=1280, text_dim=768, hidden=256, heads=4,
        dropout=0.2, n_classes=5, max_image_tokens=49,
        max_text_tokens=64,
    ):
        super().__init__()
        self.image_proj = nn.Linear(image_dim, hidden)
        self.text_proj = nn.Linear(text_dim, hidden)
        self.image_pos = nn.Parameter(
            torch.zeros(1, max_image_tokens, hidden)
        )
        self.text_pos = nn.Parameter(
            torch.zeros(1, max_text_tokens, hidden)
        )
        self.missing_token = nn.Parameter(torch.zeros(1, 1, hidden))
        self.text_to_image = CrossAttentionBlock(hidden, heads, dropout)
        self.image_to_text = CrossAttentionBlock(hidden, heads, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, image_tokens, image_mask, text_tokens, text_mask):
        """Return class logits for a batch of token sequences."""
        image = self.image_proj(image_tokens) + self.image_pos
        text = self.text_proj(text_tokens) + self.text_pos
        missing = ~image_mask.any(dim=1)
        if missing.any():
            image = image.clone()
            image_mask = image_mask.clone()
            image[missing, 0] = self.missing_token[0, 0]
            image_mask[missing, 0] = True
        text_context = self.text_to_image(
            text, image, ~image_mask
        )
        image_context = self.image_to_text(
            image, text, ~text_mask
        )
        text_pool = _masked_mean(text_context, text_mask)
        image_pool = _masked_mean(image_context, image_mask)
        return self.classifier(torch.cat([text_pool, image_pool], dim=1))


def _masked_mean(x, mask):
    """Mean over valid tokens."""
    weights = mask.unsqueeze(-1).float()
    return (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class DualHeadModel(nn.Module):
    """Shared fusion representation with text and image label heads."""

    def __init__(
        self, image_dim=1280, text_dim=768, hidden=256, dropout=0.3,
        n_classes=5,
    ):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(image_dim + text_dim + 1, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.text_head = nn.Linear(hidden, n_classes)
        self.image_head = nn.Linear(hidden, n_classes)

    def forward(self, image, text, missing):
        """Return text-label and image-label logits."""
        x = torch.cat([image, text, missing.unsqueeze(1)], dim=1)
        shared = self.shared(x)
        return self.text_head(shared), self.image_head(shared)


def count_parameters(model):
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
