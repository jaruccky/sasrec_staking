import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from sasrec.model import SASRec
from sasrec.data import (
    download_and_preprocess,
    load_data,
    split_leave_one_out,
    CausalLMDataset,
    PaddingCollateFn,
)
from sasrec.losses import (
    compute_sampled_ce_loss,
    compute_sampled_bce_loss,
    compute_full_softmax_loss,
)
from sasrec.evaluate import evaluate, validate_fast


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    loss_type="cross_entropy",
    use_sampling=False,
    grad_clip=None,
):
    model.train()

    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)

        hidden = model(input_ids)
        # hidden: [B, L, H]

        if use_sampling:
            negatives = batch["negatives"].to(device)

            if loss_type == "cross_entropy":
                loss = compute_sampled_ce_loss(
                    hidden=hidden,
                    labels=labels,
                    negatives=negatives,
                    item_emb=model.item_emb,
                )
            elif loss_type == "bce":
                loss = compute_sampled_bce_loss(
                    hidden=hidden,
                    labels=labels,
                    negatives=negatives,
                    item_emb=model.item_emb,
                )
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")

        else:
            logits = hidden @ model.item_emb.weight.T
            # logits: [B, L, item_num + 1]

            loss = compute_full_softmax_loss(
                logits=logits,
                labels=labels,
            )

        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument(
        "--dataset", type=str, default="ml-20m", choices=["ml-1m", "ml-20m"]
    )
    parser.add_argument("--data_dir", type=str, default="data")

    parser.add_argument("--hidden_units", type=int, default=256)
    parser.add_argument("--num_blocks", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=200)

    parser.add_argument(
        "--loss", type=str, default="cross_entropy", choices=["cross_entropy", "bce"]
    )
    parser.add_argument("--num_negatives", type=int, default=None)
    parser.add_argument("--full_negative_sampling", action="store_true")

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--grad_clip", type=float, default=None)

    parser.add_argument("--val_size", type=int, default=10000)
    parser.add_argument("--top_k", type=int, nargs="+", default=[10])

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="checkpoints")

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("SASRec training")
    print("=" * 80)

    set_seed(args.seed)

    if args.data_path is None:
        args.data_path = download_and_preprocess(
            dataset_name=args.dataset,
            output_dir=args.data_dir,
        )

    print("\nConfig:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    user_sequences, num_items = load_data(args.data_path)

    (
        train_sequences,
        val_sequences,
        val_targets,
        test_sequences,
        test_targets,
    ) = split_leave_one_out(user_sequences)

    train_sequences = list(train_sequences.values())

    use_sampling = args.num_negatives is not None

    train_dataset = CausalLMDataset(
        user_sequences=train_sequences,
        max_length=args.max_length,
        num_negatives=args.num_negatives,
        full_negative_sampling=args.full_negative_sampling,
        num_items=num_items,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=PaddingCollateFn(),
        pin_memory=True,
        drop_last=False,
    )

    model = SASRec(
        item_num=num_items,
        maxlen=args.max_length,
        hidden_units=args.hidden_units,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        dropout_rate=args.dropout_rate,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {num_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    best_path = os.path.join(args.save_dir, "best_model.pt")

    best_ndcg = -1.0
    epochs_no_improve = 0

    print("\n" + "=" * 80)
    print("Start training")
    print("=" * 80 + "\n")

    total_start = time.time()

    for epoch in range(1, args.max_epochs + 1):
        epoch_start = time.time()

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            loss_type=args.loss,
            use_sampling=use_sampling,
            grad_clip=args.grad_clip,
        )

        val_metrics = validate_fast(
            model=model,
            user_sequences=val_sequences,
            targets=val_targets,
            num_items=num_items,
            max_length=args.max_length,
            device=device,
            k=10,
            batch_size=256,
            max_users=args.val_size,
        )

        val_hr = val_metrics["HR@10"]
        val_ndcg = val_metrics["NDCG@10"]

        epoch_time = time.time() - epoch_start

        if val_ndcg > best_ndcg:
            best_ndcg = val_ndcg
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_path)
            marker = " * saved"
        else:
            epochs_no_improve += 1
            marker = ""

        print(
            f"Epoch {epoch:03d}/{args.max_epochs} | "
            f"loss={train_loss:.4f} | "
            f"val HR@10={val_hr:.4f} | "
            f"val NDCG@10={val_ndcg:.4f} | "
            f"time={epoch_time:.1f}s"
            f"{marker}"
        )

        if epochs_no_improve >= args.patience:
            print(f"\nEarly stopping: no improvement for {args.patience} epochs")
            break

    total_time = time.time() - total_start

    print("\n" + "=" * 80)
    print("Training finished")
    print("=" * 80)
    print(f"Best val NDCG@10: {best_ndcg:.4f}")
    print(f"Total time: {total_time:.1f}s / {total_time / 60:.1f}min")

    print("\n" + "=" * 80)
    print("Load best model")
    print("=" * 80)

    model.load_state_dict(torch.load(best_path, map_location=device))

    print("\n" + "=" * 80)
    print("Test evaluation")
    print("=" * 80)

    for k in args.top_k:
        print(f"\n--- Test @{k} ---")

        test_metrics = evaluate(
            model=model,
            user_sequences=test_sequences,
            targets=test_targets,
            num_items=num_items,
            max_length=args.max_length,
            device=device,
            k=k,
            batch_size=256,
            filter_seen=True,
        )

        for name, value in test_metrics.items():
            print(f"  {name}: {value:.4f}")

    print("\n" + "=" * 80)
    print("Done")
    print("=" * 80)


if __name__ == "__main__":
    main()
