import torch
import torch.nn as nn
from tqdm import tqdm

class Trainer:
    def __init__(
        self,
        model,
        train_dataloader,
        dev_dataloader,
        optimizer,
        epochs=10,
        device="cuda",
        save_path="best_model.pt",
        max_grad_norm=1.0
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
        self.optimizer = optimizer
        self.epochs = epochs
        self.max_grad_norm = max_grad_norm
        self.save_path = save_path

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

        self.best_dev_loss = float("inf")
    
    def train(self):
        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_one_epoch()
            dev_loss = self._evaluate()

            print(f"\nEpoch {epoch}/{self.epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Dev   Loss: {dev_loss:.4f}")

            if dev_loss < self.best_dev_loss:
                self.best_dev_loss = dev_loss
                torch.save(self.model.state_dict(), self.save_path)

    def _train_one_epoch(self):
        self.model.train()
        total_loss = 0.0

        for batch in tqdm(self.train_dataloader):
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            logits = self.model(input_ids)

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = self.criterion(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

            self.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm,
            )

            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_dataloader)

    @torch.no_grad()
    def _evaluate(self):
        self.model.eval()
        total_loss = 0.0

        for batch in tqdm(self.dev_dataloader):
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            logits = self.model(input_ids)

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = self.criterion(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

            total_loss += loss.item()

        return total_loss / len(self.dev_dataloader)