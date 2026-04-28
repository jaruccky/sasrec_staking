import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


MOVIELENS_URLS = {
    'ml-1m':  'https://files.grouplens.org/datasets/movielens/ml-1m.zip',
    'ml-20m': 'https://files.grouplens.org/datasets/movielens/ml-20m.zip',
}


def download_and_preprocess(dataset_name='ml-20m', output_dir='data', min_interactions=5):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{dataset_name}.txt'

    if output_path.exists():
        print(f"Preprocessed data already exists: {output_path}")
        return str(output_path)

    url = MOVIELENS_URLS[dataset_name]
    zip_path = output_dir / f'{dataset_name}.zip'

    if not zip_path.exists():
        print(f"Downloading {dataset_name} from {url} ...")
        urllib.request.urlretrieve(url, zip_path)
        print(f"  Saved to {zip_path} ({zip_path.stat().st_size / 1e6:.0f} MB)")
    else:
        print(f"Zip already downloaded: {zip_path}")

    print(f"Extracting ratings...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        if dataset_name == 'ml-1m':
            ratings_file = 'ml-1m/ratings.dat'
        else:
            ratings_file = 'ml-20m/ratings.csv'
        zf.extract(ratings_file, output_dir)

    print(f"Loading ratings...")
    if dataset_name == 'ml-1m':
        df = pd.read_csv(
            output_dir / ratings_file, sep='::', header=None,
            names=['user_id', 'item_id', 'rating', 'timestamp'],
            engine='python')
    else:
        df = pd.read_csv(output_dir / ratings_file)
        df.columns = ['user_id', 'item_id', 'rating', 'timestamp']

    print(f"  Raw ratings: {len(df):,}")
    print(f"  Raw users: {df['user_id'].nunique():,}")
    print(f"  Raw items: {df['item_id'].nunique():,}")

    df = df.sort_values(['user_id', 'timestamp'])

    user_counts = df.groupby('user_id').size()
    valid_users = user_counts[user_counts >= min_interactions].index
    df = df[df['user_id'].isin(valid_users)]
    print(f"  After filtering users with <{min_interactions} interactions:")
    print(f"    Interactions: {len(df):,}")
    print(f"    Users: {df['user_id'].nunique():,}")
    print(f"    Items: {df['item_id'].nunique():,}")

    user_map = {uid: idx + 1 for idx, uid in enumerate(df['user_id'].unique())}
    item_map = {iid: idx + 1 for idx, iid in enumerate(df['item_id'].unique())}
    df['user_id'] = df['user_id'].map(user_map)
    df['item_id'] = df['item_id'].map(item_map)

    print(f"Writing preprocessed data to {output_path} ...")
    with open(output_path, 'w') as f:
        for _, row in df.iterrows():
            f.write(f"{row['user_id']} {row['item_id']}\n")

    print(f"  Done! Final stats:")
    print(f"    Users: {df['user_id'].nunique():,}")
    print(f"    Items: {df['item_id'].max():,}")
    print(f"    Interactions: {len(df):,}")

    return str(output_path)


def load_data(data_path):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, sep=' ', header=None, names=['user_id', 'item_id'])

    user_sequences = {
        int(uid): [int(x) for x in items]
        for uid, items in df.groupby('user_id')['item_id'].agg(list).items()
    }

    num_items = int(df['item_id'].max())
    num_users = df['user_id'].nunique()
    num_interactions = len(df)

    print(f"  Users: {num_users:,}")
    print(f"  Items: {num_items:,}")
    print(f"  Interactions: {num_interactions:,}")
    print(f"  Avg seq length: {num_interactions / num_users:.1f}")

    return user_sequences, num_items


def split_leave_one_out(user_sequences):
    train_sequences = {}
    val_targets = {}
    test_sequences = {}
    test_targets = {}

    for uid, seq in user_sequences.items():
        if len(seq) < 3:
            continue

        test_targets[uid] = seq[-1]
        val_targets[uid] = seq[-2]

        train_sequences[uid] = seq[:-2]
        test_sequences[uid] = seq[:-1]

    val_sequences = {uid: seq[:-2] for uid, seq in user_sequences.items() if len(seq) >= 3}

    print(f"  Train users: {len(train_sequences):,}")
    print(f"  Val users:   {len(val_targets):,}")
    print(f"  Test users:  {len(test_targets):,}")

    return train_sequences, val_sequences, val_targets, test_sequences, test_targets


class CausalLMDataset(Dataset):
    def __init__(self, user_sequences, max_length=200,
                 num_negatives=None, full_negative_sampling=True,
                 num_items=None):
        self.user_sequences = user_sequences
        self.max_length = max_length
        self.num_negatives = num_negatives
        self.full_negative_sampling = full_negative_sampling
        self.num_items = num_items

        if num_negatives:
            self.all_items = np.arange(1, num_items + 1)

    def __len__(self):
        return len(self.user_sequences)

    def __getitem__(self, idx):
        item_sequence = self.user_sequences[idx]

        if len(item_sequence) > self.max_length + 1:
            item_sequence = item_sequence[-self.max_length - 1:]

        input_ids = np.array(item_sequence[:-1], dtype=np.int64)
        labels = np.array(item_sequence[1:], dtype=np.int64)

        result = {'input_ids': input_ids, 'labels': labels}

        if self.num_negatives:
            result['negatives'] = self._sample_negatives(item_sequence)

        return result

    def _sample_negatives(self, item_sequence):
        user_items_set = set(item_sequence)
        seq_len = len(item_sequence) - 1

        if self.full_negative_sampling:
            total = self.num_negatives * seq_len
            negs = []
            while len(negs) < total:
                candidates = np.random.randint(1, self.num_items + 1, size=total - len(negs))
                candidates = candidates[~np.isin(candidates, list(user_items_set))]
                negs.extend(candidates.tolist())
            negs = np.array(negs[:total], dtype=np.int64).reshape(seq_len, self.num_negatives)
        else:
            negs = []
            while len(negs) < self.num_negatives:
                candidates = np.random.randint(1, self.num_items + 1,
                                               size=self.num_negatives - len(negs))
                candidates = candidates[~np.isin(candidates, list(user_items_set))]
                negs.extend(candidates.tolist())
            negs = np.array(negs[:self.num_negatives], dtype=np.int64)

        return negs


class PaddingCollateFn:
    def __init__(self, padding_value=0, labels_padding_value=-100):
        self.padding_value = padding_value
        self.labels_padding_value = labels_padding_value

    def __call__(self, batch):
        collated = {}
        for key in batch[0]:
            if np.isscalar(batch[0][key]):
                collated[key] = torch.tensor([ex[key] for ex in batch])
                continue
            pad_val = self.labels_padding_value if key == 'labels' else self.padding_value
            values = [torch.tensor(ex[key]) for ex in batch]
            collated[key] = pad_sequence(values, batch_first=True, padding_value=pad_val)
        return collated
