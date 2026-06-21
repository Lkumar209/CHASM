"""
White-box activation probe monitor.
Trains linear and MLP probes on agent activations to predict covert-action-active.
Requires the local HF backend with hook-based activation extraction.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


class LinearProbe:
    """Linear probe on mean-pooled activations from a single layer."""

    def __init__(self, layer: int = -1) -> None:
        self.layer = layer
        self._scaler = StandardScaler()
        self._clf = LogisticRegression(C=1.0, max_iter=300, solver="lbfgs")
        self._fitted = False

    def fit(self, activations: list[np.ndarray], labels: list[int]) -> None:
        x = np.stack([a.mean(axis=0) for a in activations])
        y = np.array(labels)
        x_scaled = self._scaler.fit_transform(x)
        self._clf.fit(x_scaled, y)
        self._fitted = True

    def predict_proba(self, activations: np.ndarray) -> float:
        if not self._fitted:
            return 0.5
        x = activations.mean(axis=0, keepdims=True)
        x_scaled = self._scaler.transform(x)
        return float(self._clf.predict_proba(x_scaled)[0, 1])


class MLPProbe:
    """Two-layer MLP probe on activations."""

    def __init__(self, layer: int = -1, hidden_size: int = 64) -> None:
        self.layer = layer
        self._scaler = StandardScaler()
        self._clf = MLPClassifier(
            hidden_layer_sizes=(hidden_size,),
            max_iter=300,
            random_state=0,
        )
        self._fitted = False

    def fit(self, activations: list[np.ndarray], labels: list[int]) -> None:
        x = np.stack([a.mean(axis=0) for a in activations])
        y = np.array(labels)
        x_scaled = self._scaler.fit_transform(x)
        self._clf.fit(x_scaled, y)
        self._fitted = True

    def predict_proba(self, activations: np.ndarray) -> float:
        if not self._fitted:
            return 0.5
        x = activations.mean(axis=0, keepdims=True)
        x_scaled = self._scaler.transform(x)
        return float(self._clf.predict_proba(x_scaled)[0, 1])


def extract_activations(
    model: Any,
    tokenizer: Any,
    text: str,
    layer: int = -1,
    device: str = "cpu",
) -> np.ndarray:
    """
    Extract activations from a specified layer of a HF transformer.
    Returns (seq_len, hidden_size) array.
    Only available when torch + transformers are installed.
    """
    import torch

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[layer]  # (1, seq_len, hidden)
    return hidden[0].cpu().numpy()  # type: ignore[no-any-return]


def layer_sweep(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    labels: list[int],
    n_layers: int,
    device: str = "cpu",
) -> dict[int, float]:
    """
    Fit a linear probe at each layer and return {layer: dev_auroc} dict.
    Used for the layer-sweep ablation (M6).
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_predict

    results = {}
    for layer in range(n_layers):
        acts = [extract_activations(model, tokenizer, t, layer=layer, device=device) for t in texts]
        x = np.stack([a.mean(axis=0) for a in acts])
        y = np.array(labels)
        if len(set(y)) < 2:
            continue
        clf = LogisticRegression(C=1.0, max_iter=200)
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        probs = cross_val_predict(clf, x_scaled, y, cv=3, method="predict_proba")[:, 1]
        with np.errstate(invalid="ignore"):
            results[layer] = float(roc_auc_score(y, probs))
    return results
