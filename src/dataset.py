import torch
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
import os

def train_tokenizer(paths, vocab_size=16000):
    os.makedirs("resources", exist_ok=True)

    model_prefix = os.path.join("resources", "tokenizer")

    def iterator():
        for path in paths:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield line

    spm.SentencePieceTrainer.train(
        sentence_iterator=iterator(),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="unigram",
        character_coverage=1.0,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
    )

class Tokenizer:
    def __init__(
        self, 
        model_path, 
        paths, 
        vocab_size=16000
    ):
        if model_path is None:
            train_tokenizer(paths, vocab_size=vocab_size)
            model_path = os.path.join("resources", "tokenizer.model")

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

        self.pad_id = self.sp.pad_id()
        self.unk_id = self.sp.unk_id()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()

    def encode(self, text, add_bos=True, add_eos=True):
        ids = self.sp.encode(text, out_type=int)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids):
        return self.sp.decode(ids)

    def vocab_size(self):
        return self.sp.get_piece_size()

class MTDataset(Dataset):
    def __init__(self, src_path, tgt_path, tokenizer, max_length=256):
        with open(src_path, "r", encoding="utf-8") as f:
            src_lines = f.readlines()

        with open(tgt_path, "r", encoding="utf-8") as f:
            tgt_lines = f.readlines()

        assert len(src_lines) == len(tgt_lines)

        self.samples = []
        self.pad_id = tokenizer.pad_id
        self.bos_id = tokenizer.bos_id
        self.eos_id = tokenizer.eos_id
        self.max_length = max_length

        for src, tgt in zip(src_lines, tgt_lines):
            src = src.strip()
            tgt = tgt.strip()

            src_ids = tokenizer.encode(src, add_bos=False, add_eos=False)
            tgt_ids = tokenizer.encode(tgt, add_bos=True, add_eos=True)

            input_ids = src_ids + tgt_ids
            input_ids = input_ids[:max_length]

            labels = input_ids.copy()

            boundary = min(len(src_ids) + 1, len(labels))
            for i in range(boundary):
                labels[i] = -100

            self.samples.append({
                "input_ids": input_ids,
                "labels": labels,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "input_ids": torch.tensor(sample["input_ids"], dtype=torch.long),
            "labels": torch.tensor(sample["labels"], dtype=torch.long),
        }


def collate_fn(batch, pad_id):
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]

    max_len = max(x.size(0) for x in input_ids)

    padded_inputs = []
    padded_labels = []
    attention_masks = []

    for inp, lab in zip(input_ids, labels):
        pad_length = max_len - inp.size(0)

        padded_inputs.append(
            torch.cat([inp, torch.full((pad_length,), pad_id, dtype=torch.long)])
        )

        padded_labels.append(
            torch.cat([lab, torch.full((pad_length,), -100, dtype=torch.long)])
        )

        attention_masks.append(
            torch.cat([torch.ones(inp.size(0)), torch.zeros(pad_length)])
        )

    return {
        "input_ids": torch.stack(padded_inputs),
        "labels": torch.stack(padded_labels),
        "attention_mask": torch.stack(attention_masks),
    }


def create_dataloader(dataset, batch_size=32, shuffle=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_fn(batch, dataset.pad_id),
    )