"""Tests for synthetic training data generator."""

from __future__ import annotations

import numpy as np
import pytest

from aigrandprix.perception.data_generator import GateDataGenerator, SyntheticSample


class TestGateDataGenerator:
    def test_sample_returns_image(self):
        """Sample should return a BGR image of correct shape and dtype."""
        gen = GateDataGenerator()
        sample = gen.generate_sample(rng=np.random.default_rng(42))
        assert isinstance(sample, SyntheticSample)
        assert sample.image.shape == (480, 640, 3)
        assert sample.image.dtype == np.uint8

    def test_sample_has_annotations(self):
        """Gate images should have non-empty annotations."""
        gen = GateDataGenerator()
        # Generate multiple to find at least one with gates
        rng = np.random.default_rng(0)
        found_gate = False
        for _ in range(20):
            sample = gen.generate_sample(rng=rng)
            if sample.annotations:
                found_gate = True
                break
        assert found_gate, "Should find at least one sample with gate annotations"

    def test_annotation_corners_shape(self):
        """Gate annotation corners should be 4x2."""
        gen = GateDataGenerator()
        rng = np.random.default_rng(0)
        for _ in range(20):
            sample = gen.generate_sample(rng=rng)
            for ann in sample.annotations:
                assert ann.corners.shape == (4, 2)

    def test_bbox_within_image(self):
        """Bounding box should be within image bounds."""
        gen = GateDataGenerator()
        rng = np.random.default_rng(1)
        for _ in range(20):
            sample = gen.generate_sample(rng=rng)
            for ann in sample.annotations:
                x, y, w, h = ann.bbox
                assert x >= 0
                assert y >= 0
                assert x + w <= gen.width
                assert y + h <= gen.height

    def test_reproducible_with_seed(self):
        """Same seed should produce identical samples."""
        gen = GateDataGenerator()
        s1 = gen.generate_sample(rng=np.random.default_rng(99))
        s2 = gen.generate_sample(rng=np.random.default_rng(99))
        np.testing.assert_array_equal(s1.image, s2.image)

    def test_coco_format_keys(self, tmp_path):
        """Dataset output should have COCO format keys."""
        gen = GateDataGenerator()
        coco = gen.generate_dataset(5, str(tmp_path / "dataset"), seed=42)
        assert "images" in coco
        assert "annotations" in coco
        assert "categories" in coco
        assert len(coco["images"]) == 5
        assert coco["categories"][0]["name"] == "gate"
