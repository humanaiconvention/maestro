"""Linear probes for SGT protocol-state classification from hidden states.

Tests whether the model internally represents SGT protocol state (turn type,
pivot signal, compression due) before emitting surface markers.

Usage:
    probe = SGTProtocolProbe(layer_idx=12)
    probe.fit(hf_model, tokenizer, turn_prompts, labels)
    accuracy = probe.score(hf_model, tokenizer, test_prompts, test_labels)
    report = probe.diagnosis_report(hf_model, tokenizer, test_prompts, test_labels)

Labels should be drawn from the turn types relevant to your SGT protocol.
Common examples: "init", "continuation", "pivot", "compression".

Requirements:
    pip install scikit-learn
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SGTProtocolProbe:
    """Linear probe for SGT turn-type / protocol-state classification.

    Extracts hidden states from a specified transformer layer, pools them
    over the sequence dimension, and fits a logistic regression classifier.

    The probe is designed to answer: "Does the model internally represent
    the upcoming protocol turn type before it emits the surface token?"

    A high probe accuracy (>> chance) on a layer indicates that layer carries
    meaningful protocol state. This localises *where* in the network the SGT
    protocol is represented, which guides activation-patching experiments.

    Args:
        layer_idx: Layer to extract hidden states from. -1 = last layer.
        pooling:   Sequence pooling strategy: "mean" | "last" | "cls".
        C:         Logistic regression regularisation (smaller = stronger).
        max_iter:  Maximum solver iterations.
    """

    def __init__(
        self,
        layer_idx: int = -1,
        pooling: str = "mean",
        C: float = 1.0,
        max_iter: int = 1000,
    ) -> None:
        self.layer_idx = layer_idx
        self.pooling = pooling
        self.C = C
        self.max_iter = max_iter
        self._classifier = None
        self._label_encoder = None

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract(self, model: Any, tokenizer: Any, prompts: List[str]) -> "np.ndarray":
        """Return a (n_prompts, hidden_dim) feature matrix."""
        import numpy as np

        try:
            import torch
        except ImportError:
            raise RuntimeError("torch is required for SGTProtocolProbe")

        device = next(model.parameters()).device
        features = []

        for prompt in prompts:
            try:
                enc = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                )
                enc = {k: v.to(device) for k, v in enc.items()}

                with torch.no_grad():
                    out = model(**enc, output_hidden_states=True)

                hidden = out.hidden_states[self.layer_idx][0]  # (seq, dim)

                if self.pooling == "last":
                    vec = hidden[-1, :].float()
                elif self.pooling == "cls":
                    vec = hidden[0, :].float()
                else:  # mean
                    vec = hidden.float().mean(dim=0)

                features.append(vec.cpu().numpy())

            except Exception as exc:
                logger.warning("SGTProtocolProbe: skipped prompt — %s", exc)
                # Placeholder so indices align; will cause sklearn to warn but not crash
                features.append(None)

        # Replace None entries with zero vectors matching the first valid shape
        valid = next((f for f in features if f is not None), None)
        if valid is None:
            raise RuntimeError("SGTProtocolProbe: no hidden states could be extracted.")

        features = [f if f is not None else np.zeros_like(valid) for f in features]
        return np.array(features)

    # ------------------------------------------------------------------
    # Fit / predict / score
    # ------------------------------------------------------------------

    def fit(
        self,
        model: Any,
        tokenizer: Any,
        prompts: List[str],
        labels: List[str],
    ) -> "SGTProtocolProbe":
        """Fit the linear probe on labelled SGT turn-boundary examples.

        Args:
            model:     HF model with output_hidden_states support.
            tokenizer: HF tokenizer.
            prompts:   Turn-boundary prompt strings.
            labels:    Turn-type labels (e.g. "pivot", "compression").

        Returns:
            self, for chaining.
        """
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import LabelEncoder, StandardScaler
        except ImportError:
            raise RuntimeError("scikit-learn is required: pip install scikit-learn")

        X = self._extract(model, tokenizer, prompts)

        self._label_encoder = LabelEncoder()
        y = self._label_encoder.fit_transform(labels)

        self._classifier = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=self.C, max_iter=self.max_iter, random_state=42
            )),
        ])
        self._classifier.fit(X, y)

        logger.info(
            "SGTProtocolProbe fitted: %d samples, %d classes, layer=%d",
            len(prompts), len(self._label_encoder.classes_), self.layer_idx,
        )
        return self

    def predict(self, model: Any, tokenizer: Any, prompts: List[str]) -> List[str]:
        """Return predicted turn-type labels for a list of prompts."""
        if self._classifier is None or self._label_encoder is None:
            raise RuntimeError("Probe not fitted — call fit() first.")
        X = self._extract(model, tokenizer, prompts)
        y_pred = self._classifier.predict(X)
        return list(self._label_encoder.inverse_transform(y_pred))

    def predict_proba(
        self, model: Any, tokenizer: Any, prompts: List[str]
    ) -> "np.ndarray":
        """Return class probabilities for each prompt. Shape: (n, n_classes)."""
        if self._classifier is None:
            raise RuntimeError("Probe not fitted — call fit() first.")
        X = self._extract(model, tokenizer, prompts)
        return self._classifier.predict_proba(X)

    def score(
        self,
        model: Any,
        tokenizer: Any,
        prompts: List[str],
        labels: List[str],
    ) -> float:
        """Return accuracy on a labelled test set."""
        predicted = self.predict(model, tokenizer, prompts)
        return sum(p == t for p, t in zip(predicted, labels)) / len(labels) if labels else 0.0

    def diagnosis_report(
        self,
        model: Any,
        tokenizer: Any,
        prompts: List[str],
        labels: List[str],
    ) -> Dict[str, Any]:
        """Return a structured report for protocol adherence diagnosis.

        Returns:
            dict with:
                accuracy          — overall test accuracy
                per_class_accuracy — per-turn-type accuracy
                n_samples         — number of test examples
                layer_idx         — which layer was probed
                classes           — list of turn-type labels seen in training
        """
        if self._classifier is None or self._label_encoder is None:
            raise RuntimeError("Probe not fitted — call fit() first.")

        predicted = self.predict(model, tokenizer, prompts)
        acc = self.score(model, tokenizer, prompts, labels)

        per_class: Dict[str, float] = {}
        for cls in self._label_encoder.classes_:
            idx = [i for i, l in enumerate(labels) if l == cls]
            if idx:
                per_class[cls] = sum(
                    1 for i in idx if predicted[i] == labels[i]
                ) / len(idx)

        return {
            "accuracy": acc,
            "per_class_accuracy": per_class,
            "n_samples": len(labels),
            "layer_idx": self.layer_idx,
            "pooling": self.pooling,
            "classes": list(self._label_encoder.classes_),
        }
