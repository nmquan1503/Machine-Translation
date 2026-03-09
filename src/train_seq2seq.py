import torch
import torch.optim as optim
from tqdm import tqdm

from src.dataset import (
    Tokenizer,
    MTSeq2SeqDataset,
    create_seq2seq_dataloader
)

from src.trainer import Seq2SeqTrainer
from ssm_mamba import Seq2SeqModel, Seq2SeqModelConfig


def main():

    train_src = "resources/train.en.txt"
    train_tgt = "resources/train.vi.txt"
    dev_src = "resources/dev.en.txt"
    dev_tgt = "resources/dev.vi.txt"

    src_vocab_size = 18000
    tgt_vocab_size = 12000

    batch_size = 32
    max_length = 256
    epochs = 5
    lr = 3e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ======================
    # Tokenizers
    # ======================

    src_tokenizer = Tokenizer(
        model_path=None,
        paths=[train_src],
        vocab_size=src_vocab_size,
    )

    tgt_tokenizer = Tokenizer(
        model_path=None,
        paths=[train_tgt],
        vocab_size=tgt_vocab_size,
    )

    # ======================
    # Dataset
    # ======================

    train_dataset = MTSeq2SeqDataset(
        train_src,
        train_tgt,
        src_tokenizer,
        tgt_tokenizer,
        max_length=max_length,
    )

    dev_dataset = MTSeq2SeqDataset(
        dev_src,
        dev_tgt,
        src_tokenizer,
        tgt_tokenizer,
        max_length=max_length,
    )

    # ======================
    # Dataloader
    # ======================

    train_loader = create_seq2seq_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    dev_loader = create_seq2seq_dataloader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    # ======================
    # Model
    # ======================

    config = Seq2SeqModelConfig(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        pad_token_id=src_tokenizer.pad_id,
        bos_token_id=tgt_tokenizer.bos_id,
        eos_token_id=tgt_tokenizer.eos_id,
    )

    model = Seq2SeqModel(config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    # ======================
    # Optimizer
    # ======================

    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # ======================
    # Trainer
    # ======================

    trainer = Seq2SeqTrainer(
        model=model,
        train_dataloader=train_loader,
        dev_dataloader=dev_loader,
        optimizer=optimizer,
        epochs=epochs,
        device=device,
    )

    trainer.train()

    # ======================
    # Load best model
    # ======================

    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    model.to(device)
    model.eval()

    # ======================
    # Generation
    # ======================

    output_path = "dev_generate.txt"

    with open(output_path, "w", encoding="utf-8") as f:

        with torch.no_grad():

            for batch in tqdm(dev_loader, desc="Generating"):

                input_ids = batch["input_ids"].to(device)

                output_ids = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=100
                ).cpu()

                for seq in output_ids:

                    tokens = seq.tolist()

                    if tgt_tokenizer.eos_id in tokens:
                        tokens = tokens[:tokens.index(tgt_tokenizer.eos_id)]

                    text = tgt_tokenizer.decode(tokens)
                    f.write(text + "\n")

    print(f"Saved generation to {output_path}")


if __name__ == "__main__":
    main()