"""Stage B objective contracts and CPU-friendly loss helpers."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


ALLOWED_STAGE_B_OBJECTIVES = ("bce", "supcon", "proxy_anchor")
STAGE_B_ENCODER_FAMILY = "Sphere-Traj-LSTM"


def validate_stage_b_objective_plan(
    *,
    objectives: Sequence[str],
    encoder_family: str,
    representation_branches: Sequence[str] | None = None,
    optimize_detector: bool = False,
) -> dict[str, object]:
    """Validate the frozen Stage B objective and encoder contract."""

    objective_names = tuple(str(objective) for objective in objectives)
    if not objective_names:
        raise ValueError("Stage B requires at least one allowed objective")
    unsupported = [name for name in objective_names if name not in ALLOWED_STAGE_B_OBJECTIVES]
    if unsupported:
        raise ValueError(
            "Stage B objective is not allowed: "
            + ", ".join(unsupported)
            + f"; allowed={list(ALLOWED_STAGE_B_OBJECTIVES)}"
        )

    if encoder_family != STAGE_B_ENCODER_FAMILY:
        raise ValueError(f"Stage B encoder must be {STAGE_B_ENCODER_FAMILY}; got {encoder_family}")

    branch_values = representation_branches if representation_branches is not None else (STAGE_B_ENCODER_FAMILY,)
    branches = tuple(str(branch) for branch in branch_values)
    for branch in branches:
        if branch != STAGE_B_ENCODER_FAMILY:
            raise ValueError(
                "Stage B representation branch is disallowed; only "
                f"{STAGE_B_ENCODER_FAMILY} is allowed, got {branch}"
            )

    if bool(optimize_detector):
        raise ValueError("Stage B must not optimize the final detector")

    return {
        "objectives": objective_names,
        "encoder_family": encoder_family,
        "representation_branches": branches,
        "optimize_detector": False,
    }


def bce_with_logits_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    pos_weight: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Binary cross-entropy baseline loss for Stage B CPU tests."""

    targets = labels.float().view_as(logits.float())
    if pos_weight is not None and not isinstance(pos_weight, torch.Tensor):
        pos_weight = torch.tensor(float(pos_weight), dtype=logits.dtype, device=logits.device)
    return F.binary_cross_entropy_with_logits(logits.float(), targets, pos_weight=pos_weight)


def bce_baseline_scores(logits: torch.Tensor) -> torch.Tensor:
    """Convert BCE logits to anomaly probabilities."""

    return torch.sigmoid(logits.float())


def build_bce_baseline_head(input_dim: int) -> torch.nn.Linear:
    """Create a minimal linear BCE head."""

    if int(input_dim) <= 0:
        raise ValueError("input_dim must be positive")
    return torch.nn.Linear(int(input_dim), 1)


def supcon_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 0.07,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Supervised contrastive loss over normalized embeddings."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    features = _as_2d_float_tensor(embeddings, name="embeddings")
    target = labels.view(-1)
    if target.shape[0] != features.shape[0]:
        raise ValueError("labels length must match embeddings")

    features = F.normalize(features, dim=1)
    logits = torch.matmul(features, features.T) / float(temperature)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    batch_size = features.shape[0]
    self_mask = torch.eye(batch_size, dtype=torch.bool, device=features.device)
    positive_mask = target.view(-1, 1).eq(target.view(1, -1)) & ~self_mask
    logits_mask = (~self_mask).float()
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + eps)
    positives_per_anchor = positive_mask.sum(dim=1)
    valid = positives_per_anchor > 0
    if not bool(valid.any()):
        return features.sum() * 0.0
    per_anchor = -(positive_mask.float() * log_prob).sum(dim=1) / positives_per_anchor.clamp_min(1)
    return per_anchor[valid].mean()


def proxy_anchor_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    proxies: torch.Tensor | None = None,
    num_classes: int | None = None,
    margin: float = 0.1,
    alpha: float = 32.0,
) -> torch.Tensor:
    """Proxy Anchor loss with class-mean proxy fallback."""

    features = _as_2d_float_tensor(embeddings, name="embeddings")
    target = labels.long().view(-1)
    if target.shape[0] != features.shape[0]:
        raise ValueError("labels length must match embeddings")
    if target.numel() == 0:
        raise ValueError("labels must not be empty")

    features = F.normalize(features, dim=1)
    class_count = int(num_classes) if num_classes is not None else int(target.max().item()) + 1
    if class_count <= 0:
        raise ValueError("num_classes must be positive")
    if proxies is None:
        proxy_tensor = _class_mean_proxies(features, target, class_count)
    else:
        proxy_tensor = _as_2d_float_tensor(proxies, name="proxies").to(features.device)
        if proxy_tensor.shape != (class_count, features.shape[1]):
            raise ValueError("proxies shape must be (num_classes, embedding_dim)")
        proxy_tensor = F.normalize(proxy_tensor, dim=1)

    similarities = torch.matmul(features, proxy_tensor.T)
    one_hot = F.one_hot(target, num_classes=class_count).float()
    positive_classes = one_hot.sum(dim=0) > 0

    positive_term = torch.log1p(
        (torch.exp(-float(alpha) * (similarities - float(margin))) * one_hot).sum(dim=0)
    )
    negative_term = torch.log1p(
        (torch.exp(float(alpha) * (similarities + float(margin))) * (1.0 - one_hot)).sum(dim=0)
    )
    positive_loss = (
        positive_term[positive_classes].mean()
        if bool(positive_classes.any())
        else similarities.sum() * 0.0
    )
    negative_loss = negative_term.mean()
    return positive_loss + negative_loss


def compute_stage_b_loss(
    objective: str,
    embeddings_or_logits: torch.Tensor,
    labels: torch.Tensor,
    **kwargs: object,
) -> torch.Tensor:
    """Dispatch to the requested Stage B loss helper."""

    if objective == "bce":
        return bce_with_logits_loss(embeddings_or_logits, labels, **kwargs)
    if objective == "supcon":
        return supcon_loss(embeddings_or_logits, labels, **kwargs)
    if objective == "proxy_anchor":
        return proxy_anchor_loss(embeddings_or_logits, labels, **kwargs)
    raise ValueError(f"unsupported Stage B objective: {objective}")


def _as_2d_float_tensor(value: torch.Tensor, *, name: str) -> torch.Tensor:
    tensor = value.float()
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be a 2D tensor")
    if tensor.shape[0] == 0 or tensor.shape[1] == 0:
        raise ValueError(f"{name} must not be empty")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must be finite")
    return tensor


def _class_mean_proxies(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    proxies = torch.zeros(num_classes, features.shape[1], dtype=features.dtype, device=features.device)
    for class_id in range(num_classes):
        mask = labels == class_id
        if bool(mask.any()):
            proxies[class_id] = features[mask].mean(dim=0)
        else:
            proxies[class_id, class_id % features.shape[1]] = 1.0
    return F.normalize(proxies, dim=1)
