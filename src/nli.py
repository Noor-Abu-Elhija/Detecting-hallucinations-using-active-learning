# src/nli.py
from __future__ import annotations
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class NLI:
    """
    Natural Language Inference (entailment/neutral/contradiction).
    Default model: 'facebook/bart-large-mnli' (robust baseline).
    """
    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        device = "cpu"
        self.model.to(device)
        self.device = device
        # Common MNLI label order for this model:
        self.labels = ["contradiction", "neutral", "entailment"]

    def predict(self, premise: str, hypothesis: str):
        """
        premise: retrieved/supporting passage
        hypothesis: model's answer/claim
        returns: (label:str, conf:float, scores:dict)
        """
        inputs = self.tokenizer(premise, hypothesis, return_tensors="pt",
                                truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        idx = int(probs.argmax())
        label = self.labels[idx]
        return label, float(probs[idx]), dict(zip(self.labels, probs.tolist()))
