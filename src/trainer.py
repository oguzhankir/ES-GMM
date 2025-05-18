# src/trainer.py

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import numpy as np
from src.esgmm_filter import EsgmmFilter
import torch.nn.functional as F
from torch.utils.data import DataLoader


class Trainer:
    """
    Handles the model training process, including warm-up and ES-GMM filtering.
    """

    def __init__(self, model, loss_fn, train_loader, device, total_epochs, warmup_epochs, num_classes):
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.train_loader = train_loader
        self.device = device
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.num_classes = num_classes

        # Optimizer and Scheduler setup from the paper
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=0.2, momentum=0.9, weight_decay=1e-4, nesterov=True
        )

        warmup_scheduler = LinearLR(self.optimizer, start_factor=0.01, total_iters=self.warmup_epochs)
        main_scheduler = CosineAnnealingLR(self.optimizer, T_max=self.total_epochs - self.warmup_epochs, eta_min=1e-4)
        self.scheduler = SequentialLR(self.optimizer, schedulers=[warmup_scheduler, main_scheduler],
                                      milestones=[self.warmup_epochs])

        self.esgmm_filter = EsgmmFilter(device)

    def _get_all_data_for_filtering(self):
        """
        Performs a full forward pass on the dataset without gradients to get
        embeddings and predictions needed for filtering.
        """
        self.model.eval()
        all_embeddings = []
        all_labels = []
        all_outputs = []

        print("Gathering embeddings and predictions for ES-GMM filtering...")
        with torch.no_grad():
            for inputs, labels in tqdm(self.train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                embeddings = self.model(inputs)
                # We need raw outputs for AAMSoftmax logic to get predictions
                outputs = self.loss_fn.weight.matmul(F.normalize(embeddings).T).T * self.loss_fn.s

                all_embeddings.append(embeddings)
                all_labels.append(labels)
                all_outputs.append(outputs)

        return torch.cat(all_embeddings), torch.cat(all_labels), torch.cat(all_outputs)

    def train(self):
        """
        Main training loop.
        """
        for epoch in range(self.total_epochs):
            self.model.train()
            total_loss = 0

            print(f"\n--- Epoch {epoch + 1}/{self.total_epochs} ---")

            # --- Sample Selection Stage (after warm-up) ---
            if epoch >= self.warmup_epochs:
                embeddings, labels, outputs = self._get_all_data_for_filtering()
                clean_mask = self.esgmm_filter.filter_labels(embeddings, labels, outputs, self.num_classes)

                # Create a temporary dataset/loader for this epoch with clean samples
                clean_indices = clean_mask.nonzero(as_tuple=True)[0].cpu().numpy()
                clean_sampler = torch.utils.data.SubsetRandomSampler(clean_indices)
                epoch_train_loader = DataLoader(
                    self.train_loader.dataset,
                    batch_size=self.train_loader.batch_size,
                    sampler=clean_sampler,
                    num_workers=self.train_loader.num_workers,
                    drop_last=self.train_loader.drop_last
                )
            else:
                # --- Warm Up Stage ---
                print(f"Warm-up Stage: Training on all samples.")
                epoch_train_loader = self.train_loader

            # --- Training on selected samples ---
            self.model.train()
            progress_bar = tqdm(epoch_train_loader, desc="Training")
            for i, (inputs, labels) in enumerate(progress_bar):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                embeddings = self.model(inputs)
                loss = self.loss_fn(embeddings, labels)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                progress_bar.set_postfix({'Loss': total_loss / (i + 1)})

            self.scheduler.step()
            avg_loss = total_loss / len(epoch_train_loader)
            print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}, LR: {self.optimizer.param_groups[0]['lr']:.6f}")