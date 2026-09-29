"""Testes rápidos das dimensões e do fluxo diferenciável."""

import numpy as np
import torch

from xtcn.features import extract_features
from xtcn.fingerprint import FeatureStandardizer, aggregate_heatmap_by_cell, build_fingerprint_map
from xtcn.gradcam import GradCAM
from xtcn.model import XTCN
from xtcn.preprocessing import relative_conductance
from xtcn.synthetic import generate_synthetic_enose


def test_preprocessing_features_and_fingerprint() -> None:
    signals, _ = generate_synthetic_enose(samples_per_class=2, time_steps=64, seed=1)
    relative, g0 = relative_conductance(signals, baseline_samples=4)
    features = extract_features(relative, sampling_rate=10.0)
    scaled = FeatureStandardizer().fit_transform(features)
    fingerprints = build_fingerprint_map(scaled)
    assert g0.shape == (12, 6)
    assert features.shape == (12, 6, 8)
    assert fingerprints.shape == (12, 3, 224, 224)
    assert np.isfinite(fingerprints).all()


def test_model_and_gradcam() -> None:
    model = XTCN()
    inputs = torch.rand(2, 3, 224, 224)
    with GradCAM(model, model.gradcam_target_layer) as gradcam:
        heatmaps, logits = gradcam(inputs)
    assert logits.shape == (2, 6)
    assert heatmaps.shape == (2, 1, 224, 224)
    assert aggregate_heatmap_by_cell(heatmaps[0].numpy()).shape == (6, 8)

