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
        unk_id=3,
        bos_id=1,
        eos_id=2,
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

            boundary = min(len(src_ids), len(labels))
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

class MTSeq2SeqDataset(Dataset):
    def __init__(
        self,
        src_path,
        tgt_path,
        src_tokenizer,
        tgt_tokenizer,
        max_length=256
    ):
        with open(src_path, "r", encoding="utf-8") as f:
            src_lines = f.readlines()

        with open(tgt_path, "r", encoding="utf-8") as f:
            tgt_lines = f.readlines()

        assert len(src_lines) == len(tgt_lines)

        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

        # dùng chung pad
        self.pad_id = src_tokenizer.pad_id

        self.bos_id = tgt_tokenizer.bos_id
        self.eos_id = tgt_tokenizer.eos_id

        self.max_length = max_length
        self.samples = []

        for src, tgt in zip(src_lines, tgt_lines):

            src = src.strip()
            tgt = tgt.strip()

            src_ids = src_tokenizer.encode(
                src,
                add_bos=False,
                add_eos=False
            )

            tgt_ids = tgt_tokenizer.encode(
                tgt,
                add_bos=False,
                add_eos=True
            )

            decoder_input_ids = [self.bos_id] + tgt_ids[:-1]

            src_ids = src_ids[:max_length]
            decoder_input_ids = decoder_input_ids[:max_length]
            tgt_ids = tgt_ids[:max_length]

            self.samples.append({
                "input_ids": src_ids,
                "decoder_input_ids": decoder_input_ids,
                "labels": tgt_ids
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]

        return {
            "input_ids": torch.tensor(sample["input_ids"], dtype=torch.long),
            "decoder_input_ids": torch.tensor(sample["decoder_input_ids"], dtype=torch.long),
            "labels": torch.tensor(sample["labels"], dtype=torch.long),
        }

def collate_fn(batch, pad_id):
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]

    max_len = max(x.size(0) for x in input_ids)

    padded_inputs = []
    padded_labels = []

    for inp, lab in zip(input_ids, labels):
        pad_length = max_len - inp.size(0)

        padded_inputs.append(
            torch.cat([inp, torch.full((pad_length,), pad_id, dtype=torch.long)])
        )

        padded_labels.append(
            torch.cat([lab, torch.full((pad_length,), -100, dtype=torch.long)])
        )

    return {
        "input_ids": torch.stack(padded_inputs),
        "labels": torch.stack(padded_labels),
    }

def seq2seq_collate_fn(batch, pad_id):

    input_ids = [x["input_ids"] for x in batch]
    decoder_input_ids = [x["decoder_input_ids"] for x in batch]
    labels = [x["labels"] for x in batch]

    max_src_len = max(x.size(0) for x in input_ids)
    max_tgt_len = max(x.size(0) for x in decoder_input_ids)

    padded_src = []
    padded_dec = []
    padded_labels = []

    for src, dec, lab in zip(input_ids, decoder_input_ids, labels):

        src_pad = max_src_len - src.size(0)
        tgt_pad = max_tgt_len - dec.size(0)

        padded_src.append(
            torch.cat([src, torch.full((src_pad,), pad_id, dtype=torch.long)])
        )

        padded_dec.append(
            torch.cat([dec, torch.full((tgt_pad,), pad_id, dtype=torch.long)])
        )

        padded_labels.append(
            torch.cat([lab, torch.full((tgt_pad,), -100, dtype=torch.long)])
        )

    return {
        "input_ids": torch.stack(padded_src),
        "decoder_input_ids": torch.stack(padded_dec),
        "labels": torch.stack(padded_labels)
    }

def create_dataloader(dataset, batch_size=32, shuffle=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_fn(batch, dataset.pad_id),
    )

def create_seq2seq_dataloader(dataset, batch_size=32, shuffle=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: seq2seq_collate_fn(batch, dataset.pad_id),
    )