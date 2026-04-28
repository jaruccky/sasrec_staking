import torch
import torch.nn.functional as F


def compute_sampled_ce_loss(hidden, labels, negatives, item_emb, padding_idx=0):
    labels_safe = labels.clone()
    labels_safe[labels_safe == -100] = padding_idx

    emb_pos = item_emb(labels_safe)
    logits_pos = (hidden * emb_pos).sum(dim=-1)

    if negatives.ndim == 2:
        emb_neg = item_emb(negatives)
        logits_neg = torch.bmm(hidden, emb_neg.transpose(1, 2))
    else:
        emb_neg = item_emb(negatives)
        logits_neg = torch.matmul(hidden.unsqueeze(2), emb_neg.transpose(2, 3)).squeeze(2)

    logits = torch.cat([logits_pos.unsqueeze(-1), logits_neg], dim=-1)

    targets = labels.clone()
    targets[targets != -100] = 0

    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
    return loss


def compute_sampled_bce_loss(hidden, labels, negatives, item_emb, padding_idx=0):
    labels_safe = labels.clone()
    labels_safe[labels_safe == -100] = padding_idx

    emb_pos = item_emb(labels_safe)
    logits_pos = (hidden * emb_pos).sum(dim=-1)

    if negatives.ndim == 2:
        emb_neg = item_emb(negatives)
        logits_neg = torch.bmm(hidden, emb_neg.transpose(1, 2))
    else:
        emb_neg = item_emb(negatives)
        logits_neg = torch.matmul(hidden.unsqueeze(2), emb_neg.transpose(2, 3)).squeeze(2)

    logits = torch.cat([logits_pos.unsqueeze(-1), logits_neg], dim=-1)

    targets = torch.zeros_like(logits)
    targets[:, :, 0] = 1.0

    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    mask = (labels != -100).unsqueeze(-1).float()
    loss = (loss * mask).sum() / mask.sum() / logits.size(-1)
    return loss


def compute_full_softmax_loss(logits, labels):
    return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
