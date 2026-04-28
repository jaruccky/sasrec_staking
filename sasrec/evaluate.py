import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


@torch.no_grad()
def evaluate(model, user_sequences, user_targets, num_items, max_length,
             device, k=10, batch_size=256, filter_seen=True):
    model.eval()

    hit, ndcg, mrr = 0.0, 0.0, 0.0
    num_users = 0

    user_ids = list(user_targets.keys())
    for start in range(0, len(user_ids), batch_size):
        batch_users = user_ids[start:start + batch_size]

        input_seqs = []
        targets = []
        histories = []
        for uid in batch_users:
            seq = user_sequences[uid]
            input_seq = seq[-max_length:]
            input_seqs.append(torch.tensor(input_seq, dtype=torch.long))
            targets.append(user_targets[uid])
            histories.append(set(seq))

        input_ids = pad_sequence(input_seqs, batch_first=True, padding_value=0).to(device)

        hidden = model(input_ids)

        lengths = (input_ids != 0).sum(dim=1) - 1
        rows = torch.arange(len(batch_users), device=device)
        last_hidden = hidden[rows, lengths]

        all_scores = torch.matmul(last_hidden, model.item_emb.weight.T)

        all_scores = all_scores.cpu().numpy()
        for i, uid in enumerate(batch_users):
            scores = all_scores[i].copy()
            target = targets[i]

            if filter_seen:
                seen = np.array(list(histories[i]), dtype=np.intp)
                scores[seen] = -np.inf
            scores[0] = -np.inf

            target_score = all_scores[i][target]
            rank = (scores > target_score).sum() + 1

            if rank <= k:
                hit += 1.0
                ndcg += 1.0 / np.log2(rank + 1)
                mrr += 1.0 / rank

            num_users += 1

        if (start // batch_size) % 50 == 0:
            print(f"  Evaluated {num_users}/{len(user_ids)} users...", flush=True)

    metrics = {
        f'HR@{k}': hit / num_users,
        f'NDCG@{k}': ndcg / num_users,
        f'MRR@{k}': mrr / num_users,
    }
    return metrics


@torch.no_grad()
def validate_fast(model, val_sequences, val_targets, num_items, max_length,
                  device, k=10, batch_size=256, max_users=10000):
    user_ids = list(val_targets.keys())
    if len(user_ids) > max_users:
        user_ids = np.random.choice(user_ids, size=max_users, replace=False).tolist()

    subset_targets = {uid: val_targets[uid] for uid in user_ids}
    return evaluate(model, val_sequences, subset_targets, num_items, max_length,
                    device, k=k, batch_size=batch_size, filter_seen=True)
