# src/nli.py
# This module wraps a Natural Language Inference (NLI) model (BART-Large-MNLI)
# for entailment prediction between a premise (retrieved text) and hypothesis (generated answer).

from __future__ import annotations
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class NLI:
    """
    Natural Language Inference (NLI) utility.
    Uses 'facebook/bart-large-mnli' by default to classify a text pair as:
      - entailment
      - neutral
      - contradiction
    """

    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        """Initialize tokenizer and model for MNLI classification."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

        self.device = "cpu"
        self.model.to(self.device)
        self.labels = ["contradiction", "neutral", "entailment"]

    def predict(self, premise: str, hypothesis: str):
        """
        Predicts the NLI relationship between premise and hypothesis.

        Args:
            premise (str): Supporting or retrieved sentence.
            hypothesis (str): Candidate statement to verify.

        Returns:
            tuple:
                (label: str, confidence: float, scores: dict)
        """
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            padding=True
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        idx = int(probs.argmax())
        label = self.labels[idx]

        return label, float(probs[idx]), dict(zip(self.labels, probs.tolist()))
