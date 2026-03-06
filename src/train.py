import torch
import torch.optim as optim
from tqdm import tqdm

from src.dataset import Tokenizer, MTDataset, create_dataloader
from src.trainer import Trainer
from ssm_mamba import CausalLM, CausalLMConfig

def main():

    train_src = "resources/train.en.txt"
    train_tgt = "resources/train.vi.txt"
    dev_src = "resources/dev.en.txt"
    dev_tgt = "resources/dev.vi.txt"

    vocab_size = 32000
    batch_size = 32
    max_length = 256
    epochs = 5
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

    model = CausalLM().to(device)
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

    model.eval()

    output_path = "dev_generate.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        with torch.no_grad():
            for batch in tqdm(dev_loader, desc="Generating"):

                input_ids = batch["input_ids"].to(device)

                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=100
                )

                output_ids = output_ids.cpu()

                for seq in output_ids:
                    text = tokenizer.decode(seq.tolist())
                    f.write(text + "\n")

    print(f"Saved generation to {output_path}")
if __name__ == "__main__":
    main()