"""
Elastic Weight Consolidation (EWC)

Implements continual learning regularization to prevent catastrophic forgetting
by penalizing changes to parameters important for previous tasks.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class EWC:
    """
    Elastic Weight Consolidation (EWC) regularizer.

    Stores a snapshot of model parameters and estimates their importance
    using the Fisher Information Matrix.
    """

    def __init__(self, model, dataset, device="cpu", fisher_sample_size=None):
        """
        Args:
            model (nn.Module): trained model after previous task
            dataset (torch.utils.data.Dataset): dataset from previous task
            device (str): computation device
            fisher_sample_size (int, optional): number of samples to estimate Fisher
        """
        self.model = model
        self.dataset = dataset
        self.device = device

        # Store reference parameters (θ*)
        self.params = {
            name: param.clone().detach().to(self.device)
            for name, param in self.model.named_parameters()
        }

        # Estimate parameter importance (Fisher Information)
        self.precision_matrices = self._compute_fisher(
            sample_size=fisher_sample_size
        )

    def _compute_fisher(self, sample_size=None):
        """
        Estimates the diagonal Fisher Information Matrix.

        Args:
            sample_size (int, optional): limit number of samples for efficiency
        """
        precision_matrices = {
            name: torch.zeros_like(param).to(self.device)
            for name, param in self.model.named_parameters()
        }

        self.model.to(self.device)
        self.model.eval()

        loader = DataLoader(self.dataset, batch_size=1, shuffle=True)
        criterion = nn.CrossEntropyLoss()

        count = 0
        for data, target in loader:
            data = data.to(self.device)
            target = target.to(self.device)

            self.model.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    precision_matrices[name] += param.grad.data.pow(2)

            count += 1
            if sample_size is not None and count >= sample_size:
                break

        # Normalize
        if count == 0:
            count = 1

        for name in precision_matrices:
            precision_matrices[name] /= float(count)

        return precision_matrices

    def penalty(self, model):
        """
        Computes the EWC penalty term.

        Args:
            model (nn.Module): current model
        Returns:
            torch.Tensor: scalar EWC penalty
        """
        loss = 0.0
        for name, param in model.named_parameters():
            loss += (
                self.precision_matrices[name]
                * (param.to(self.device) - self.params[name]).pow(2)
            ).sum()
        return loss

