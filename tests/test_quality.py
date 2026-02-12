"""Tests for src.data.quality — blur and exposure scoring."""

import numpy as np
import pytest

from src.data.quality import compute_blur_score, compute_exposure_score


class TestComputeBlurScore:
    def test_sharp_checkerboard(self) -> None:
        """Sharp checkerboard pattern has high blur score."""
        # Create a sharp checkerboard
        checker = np.zeros((256, 256), dtype=np.uint8)
        checker[::2, ::2] = 255
        checker[1::2, 1::2] = 255
        score = compute_blur_score(checker)
        # Sharp patterns have very high Laplacian variance
        assert score > 1000

    def test_blurred_image(self) -> None:
        """Uniform gray image has near-zero blur score."""
        uniform = np.full((256, 256), 128, dtype=np.uint8)
        score = compute_blur_score(uniform)
        assert score < 1.0

    def test_sharp_higher_than_blurred(self) -> None:
        """Sharp image scores higher than blurred version."""
        import cv2

        checker = np.zeros((256, 256), dtype=np.uint8)
        checker[::2, ::2] = 255
        checker[1::2, 1::2] = 255

        blurred = cv2.GaussianBlur(checker, (31, 31), 10)

        sharp_score = compute_blur_score(checker)
        blurred_score = compute_blur_score(blurred)
        assert sharp_score > blurred_score


class TestComputeExposureScore:
    def test_all_white(self) -> None:
        """All-white image is fully clipped."""
        white = np.full((100, 100), 255, dtype=np.uint8)
        score = compute_exposure_score(white)
        assert score == pytest.approx(1.0)

    def test_all_black(self) -> None:
        """All-black image is fully clipped."""
        black = np.zeros((100, 100), dtype=np.uint8)
        score = compute_exposure_score(black)
        assert score == pytest.approx(1.0)

    def test_mid_gray(self) -> None:
        """Mid-gray image has zero clipping."""
        gray = np.full((100, 100), 128, dtype=np.uint8)
        score = compute_exposure_score(gray)
        assert score == pytest.approx(0.0)

    def test_partial_clipping(self) -> None:
        """Half-white, half-gray has 50% clipping."""
        img = np.full((100, 100), 128, dtype=np.uint8)
        img[:50, :] = 255
        score = compute_exposure_score(img)
        assert score == pytest.approx(0.5)
