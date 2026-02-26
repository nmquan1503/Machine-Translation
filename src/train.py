import torch
import torch.optim as optim

from src.dataset import Tokenizer, MTDataset, create_dataloader
from src.trainer import Trainer
from Mamba import CausalLM, CausalLMConfig

def main():

    train_src = "resources/train.en.txt"
    train_tgt = "resources/train.vi.txt"
    dev_src = "resources/dev.en.txt"
    dev_tgt = "resources/dev.vi.txt"

    vocab_size = 30000
    batch_size = 32
    max_length = 256
    epochs = 10
    lr = 3e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = Tokenizer(
        model_path=None,
        paths=[train_src, train_tgt],
        vocab_size=vocab_size,
    )

    train_dataset = MTDataset(
        train_src,
        train_tgt,
        tokenizer,
        max_length=max_length,
    )

    dev_dataset = MTDataset(
        dev_src,
        dev_tgt,
        tokenizer,
        max_length=max_length,
    )

    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    dev_loader = create_dataloader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    config = CausalLMConfig(
        vocab_size=vocab_size,
        model_dim=512,
        state_dim=16,
        conv_kernel=4,
        expansion_factor=2,
        dropout_rate=0.2,
        num_layers=5,
        tie_embeddings=True,
    )

    model = CausalLM(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=lr)

    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        dev_dataloader=dev_loader,
        optimizer=optimizer,
        epochs=epochs,
        device=device,
    )

    trainer.train()


if __name__ == "__main__":
    main()