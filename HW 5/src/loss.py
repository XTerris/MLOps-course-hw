"""Causal LM loss только на позициях ответа из маски ДЗ 4.

У issue длинный промпт и ответ из 3–4 токенов. Проекция в словарь на
маскированных позициях не нужна: это тот же loss, без лишних логитов.
"""

import torch
from torch.nn import functional as F

from src.data import LABEL_PAD_ID


def answer_loss(model, batch: dict) -> tuple[torch.Tensor, int]:
    targets = batch["labels"][:, 1:]
    count = int((targets != LABEL_PAD_ID).sum())
    if not count:
        raise ValueError("в батче нет токенов ответа после causal shift")
    first = int(torch.nonzero((targets != LABEL_PAD_ID).any(dim=0))[0, 0])
    inputs = {k: v for k, v in batch.items() if k != "labels"}
    # Непрерывный хвост использует slice: backward не требует
    # недетерминированного index_put_with_accumulate на MPS.
    keep = batch["input_ids"].shape[1] - first
    logits = model(**inputs, logits_to_keep=keep, use_cache=False).logits[:, :-1, :]
    selected = targets[:, first:]
    total = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]), selected.reshape(-1),
        ignore_index=LABEL_PAD_ID, reduction="sum",
    )
    return total, count
