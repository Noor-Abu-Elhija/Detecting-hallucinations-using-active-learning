# src/entailment.py
# This module provides a lightweight wrapper around DeBERTa for natural language inference (NLI).
# It computes entailment probabilities between pairs of sentences, used in semantic clustering.

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class EntailmentDeberta:
    """
    Wrapper for DeBERTa-XLarge-MNLI model to compute entailment confidence between text pairs.
    """

    def __init__(self):
        """Initialize tokenizer and model for entailment checking."""
        self.device = "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-xlarge-mnli")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-xlarge-mnli"
        ).to(self.device)
        self.model.eval()

    def check_implication(self, a: str, b: str) -> float:
        """
        Compute entailment score for hypothesis b given premise a.

        Args:
            a (str): premise text
            b (str): hypothesis text

        Returns:
            float: probability that a entails b (index 2 in MNLI label space)
        """
        inputs = self.tokenizer(a, b, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.softmax(logits, dim=-1)
        return probs[0][2].item()  # Entailment class = index 2
