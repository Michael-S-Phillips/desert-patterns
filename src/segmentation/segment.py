"""Pattern segmentation: DINOv3 patch attention maps → SAM instance masks."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from PIL import Image

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SegmentationConfig:
    """Configuration for DINOv3-attention + SAM segmentation."""

    sam_checkpoint: str = "/Volumes/Rohan/Software/sam/sam_vit_b_01ec64.pth"
    output_dir: str = "outputs/segmentations"
    n_foreground_prompts: int = 5
    n_background_prompts: int = 5
    threshold_method: str = "otsu"           # "otsu" | "percentile"
    attention_percentile: float = 0.70
    min_mask_area_fraction: float = 0.005
    iou_dedup_threshold: float = 0.5
    attention_mode: str = "classifier"       # "classifier" | "self_attention"
    prompt_strategy: str = "topk"            # "topk" | "fps"
    sam_version: str = "sam1"               # "sam1" | "sam2"
    sam2_checkpoint: str = ""
    sam2_model_cfg: str = "sam2_hiera_b+.yaml"


def load_segmentation_config(config_dict: dict) -> SegmentationConfig:
    """Load SegmentationConfig from the full parsed YAML config dict.

    Reads the ``segmentation:`` sub-dict internally, same pattern as
    ``load_classifier_config()``.

    Args:
        config_dict: Full parsed YAML dictionary.

    Returns:
        Populated SegmentationConfig.
    """
    seg = config_dict.get("segmentation", {})
    return SegmentationConfig(
        sam_checkpoint=seg.get("sam_checkpoint", SegmentationConfig.sam_checkpoint),
        output_dir=seg.get("output_dir", SegmentationConfig.output_dir),
        n_foreground_prompts=seg.get("n_foreground_prompts", SegmentationConfig.n_foreground_prompts),
        n_background_prompts=seg.get("n_background_prompts", SegmentationConfig.n_background_prompts),
        threshold_method=seg.get("threshold_method", SegmentationConfig.threshold_method),
        attention_percentile=seg.get("attention_percentile", SegmentationConfig.attention_percentile),
        min_mask_area_fraction=seg.get("min_mask_area_fraction", SegmentationConfig.min_mask_area_fraction),
        iou_dedup_threshold=seg.get("iou_dedup_threshold", SegmentationConfig.iou_dedup_threshold),
        attention_mode=seg.get("attention_mode", SegmentationConfig.attention_mode),
        prompt_strategy=seg.get("prompt_strategy", SegmentationConfig.prompt_strategy),
        sam_version=seg.get("sam_version", SegmentationConfig.sam_version),
        sam2_checkpoint=seg.get("sam2_checkpoint", SegmentationConfig.sam2_checkpoint),
        sam2_model_cfg=seg.get("sam2_model_cfg", SegmentationConfig.sam2_model_cfg),
    )


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class SegmentationResult:
    """Output of segmenting a single image."""

    attention_map: np.ndarray        # float32, shape (H, W), values in [0, 1]
    binary_mask: np.ndarray          # uint8, shape (H, W), values 0 or 255
    instance_masks: list[np.ndarray] # each bool array shape (H, W)
    iou_scores: list[float]          # one per instance mask (SAM predicted IoU)
    image_path: Path
    class_name: str


# ---------------------------------------------------------------------------
# Segmenter
# ---------------------------------------------------------------------------


class PatternSegmenter:
    """Segment desert pattern images using DINOv3 attention maps as SAM prompts.

    DINOv3 and SAM are both lazy-loaded on first use.

    Args:
        classifier_model: Fitted ``sklearn.linear_model.LogisticRegression``.
        seg_config: Segmentation configuration.
        dino_config: DINOv3 configuration (from ``load_dino_config()``).
    """

    def __init__(
        self,
        classifier_model: Any | None,
        seg_config: SegmentationConfig,
        dino_config: Any,
    ) -> None:
        self._model = classifier_model
        self._seg_config = seg_config
        self._dino_config = dino_config
        self._extractor: Any = None
        self._sam_predictor: Any = None

    # ------------------------------------------------------------------
    # Lazy loaders
    # ------------------------------------------------------------------

    def _get_extractor(self) -> Any:
        if self._extractor is None:
            from src.features.dino_embeddings import DinoFeatureExtractor
            self._extractor = DinoFeatureExtractor(self._dino_config)
        return self._extractor

    def _get_sam_predictor(self) -> Any:
        if self._sam_predictor is None:
            import torch
            from segment_anything import SamPredictor, sam_model_registry

            device = "mps" if torch.backends.mps.is_available() else "cpu"
            sam = sam_model_registry["vit_b"](checkpoint=self._seg_config.sam_checkpoint)
            sam.to(device)
            self._sam_predictor = SamPredictor(sam)
            logger.info("SAM loaded on device=%s", device)
        return self._sam_predictor

    # ------------------------------------------------------------------
    # Attention map
    # ------------------------------------------------------------------

    def _compute_attention(
        self,
        patch_tokens: np.ndarray,
        cls_coef: np.ndarray,
        image_size: tuple[int, int],
    ) -> np.ndarray:
        """Compute spatial attention map from patch tokens × LR coefficient.

        Args:
            patch_tokens: shape (n_patches, 768)
            cls_coef: shape (768,) — LR weight vector for the image's class
            image_size: (width, height) in PIL convention

        Returns:
            float32 attention map of shape (height, width), values in [0, 1]
        """
        from scipy.ndimage import gaussian_filter

        n_patches = patch_tokens.shape[0]
        grid_size = int(round(np.sqrt(n_patches)))

        activations = (patch_tokens @ cls_coef)[: grid_size * grid_size]
        spatial = activations.reshape(grid_size, grid_size).astype(np.float32)

        # Normalize grid to [0, 1]
        vmin, vmax = spatial.min(), spatial.max()
        spatial_norm = (
            (spatial - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(spatial)
        )

        # Upsample to native image size: PIL.Image.resize takes (width, height)
        width, height = image_size
        heat_pil = Image.fromarray((spatial_norm * 255).astype(np.uint8), mode="L")
        heat_up = (
            np.asarray(heat_pil.resize((width, height), Image.BILINEAR), dtype=np.float32)
            / 255.0
        )

        # Gaussian smoothing then re-normalize
        heat_smooth = gaussian_filter(heat_up, sigma=2).astype(np.float32)
        smin, smax = heat_smooth.min(), heat_smooth.max()
        return (
            (heat_smooth - smin) / (smax - smin) if smax > smin else heat_smooth
        )

    # ------------------------------------------------------------------
    # Self-attention map (alternative to classifier-based attention)
    # ------------------------------------------------------------------

    def _select_attention_head(self, attn_maps: np.ndarray) -> int:
        """Return the index of the head with the highest spatial entropy.

        Higher entropy = attention is spread over more patches = more
        spatially informative for segmentation prompts.

        Args:
            attn_maps: float32, shape (n_heads, n_patches), each row sums to 1.

        Returns:
            Index of the highest-entropy head.
        """
        eps = 1e-10
        entropy = -(attn_maps * np.log(attn_maps + eps)).sum(axis=-1)  # (n_heads,)
        return int(np.argmax(entropy))

    def _compute_self_attention(
        self,
        pil_img: Image,
        image_size: tuple[int, int],
    ) -> tuple[np.ndarray, int]:
        """Compute attention map from the max-entropy DINOv3 self-attention head.

        Extracts per-head CLS→patch attention, selects the head with the
        highest spatial entropy, then upsamples and smooths to native resolution
        (same post-processing as ``_compute_attention``).

        Args:
            pil_img: PIL Image at native resolution.
            image_size: (width, height) in PIL convention.

        Returns:
            (attention_map, n_patches) where attention_map is float32 (H, W)
            in [0, 1] and n_patches is read from the attention tensor shape.
        """
        from scipy.ndimage import gaussian_filter

        attn_maps = self._get_extractor().extract_attention_maps(pil_img)
        n_patches = attn_maps.shape[1]

        best_head = self._select_attention_head(attn_maps)
        selected = attn_maps[best_head]  # (n_patches,)

        grid_size = int(round(np.sqrt(n_patches)))
        width, height = image_size

        spatial = selected.reshape(grid_size, grid_size).astype(np.float32)
        heat_pil = Image.fromarray((spatial * 255).astype(np.uint8), mode="L")
        heat_up = (
            np.asarray(heat_pil.resize((width, height), Image.BILINEAR), dtype=np.float32)
            / 255.0
        )

        heat_smooth = gaussian_filter(heat_up, sigma=2).astype(np.float32)
        smin, smax = heat_smooth.min(), heat_smooth.max()
        attn_out = (
            (heat_smooth - smin) / (smax - smin) if smax > smin else heat_smooth
        )
        return attn_out, n_patches

    # ------------------------------------------------------------------
    # Thresholding
    # ------------------------------------------------------------------

    def _threshold(self, attention_map: np.ndarray) -> np.ndarray:
        """Convert soft attention map to binary mask (0 / 255 uint8).

        Uses Otsu or fixed percentile depending on ``seg_config.threshold_method``.
        """
        uint8_map = (attention_map * 255).astype(np.uint8)

        if self._seg_config.threshold_method == "otsu":
            _, binary = cv2.threshold(
                uint8_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        else:  # "percentile"
            thresh_val = float(
                np.quantile(attention_map, self._seg_config.attention_percentile)
            )
            binary = ((attention_map >= thresh_val) * 255).astype(np.uint8)

        return binary

    # ------------------------------------------------------------------
    # Prompt generation
    # ------------------------------------------------------------------

    def _farthest_point_sample(
        self,
        candidates: np.ndarray,
        grid_size: int,
        n_points: int,
    ) -> np.ndarray:
        """Select spatially diverse patch indices via farthest-point sampling.

        Seeds from the first candidate (caller should pre-sort descending by
        attention so the highest-attention patch is the seed).

        Args:
            candidates: 1-D int array of flat patch indices.
            grid_size: side length of the patch grid (e.g. 14 for 196 patches).
            n_points: number of points to select.

        Returns:
            Selected flat patch indices, shape (min(n_points, len(candidates)),).
        """
        if len(candidates) <= n_points:
            return candidates

        rows = candidates // grid_size
        cols = candidates % grid_size
        coords = np.stack([rows, cols], axis=1).astype(np.float32)  # (N, 2)

        selected_local = [0]  # seed: first candidate (highest attention)
        min_dists = np.full(len(candidates), np.inf)

        for _ in range(n_points - 1):
            last = coords[selected_local[-1]]
            dists = np.linalg.norm(coords - last, axis=1)
            min_dists = np.minimum(min_dists, dists)
            min_dists[selected_local] = -np.inf  # exclude already selected
            selected_local.append(int(np.argmax(min_dists)))

        return candidates[np.array(selected_local)]

    def _patch_centroids(
        self,
        attention_map: np.ndarray,
        n_patches: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map top/bottom attention patches to pixel coordinates for SAM prompts.

        Args:
            attention_map: float32, shape (H, W) — native-resolution attention map
            n_patches: total number of patches (derives grid_size dynamically)

        Returns:
            (fg_points, bg_points) each shape (k, 2) as [[x, y], ...] float32
        """
        grid_size = int(round(np.sqrt(n_patches)))
        height, width = attention_map.shape

        # Score each grid cell by its mean attention in the upsampled map
        patch_activations = np.array(
            [
                attention_map[
                    int(row / grid_size * height) : int((row + 1) / grid_size * height),
                    int(col / grid_size * width) : int((col + 1) / grid_size * width),
                ].mean()
                for row in range(grid_size)
                for col in range(grid_size)
            ],
            dtype=np.float32,
        )

        n_fg = min(self._seg_config.n_foreground_prompts, n_patches)
        n_bg = min(self._seg_config.n_background_prompts, n_patches)

        sorted_idx = np.argsort(patch_activations)

        if self._seg_config.prompt_strategy == "fps":
            median_val = float(np.median(patch_activations))
            # Foreground: at or above median, sorted descending (seed = highest attn patch)
            fg_mask = patch_activations >= median_val
            fg_candidates = np.where(fg_mask)[0]
            fg_candidates = fg_candidates[np.argsort(patch_activations[fg_candidates])[::-1]]
            # Background: below median, sorted ascending (seed = lowest attn patch)
            bg_candidates = np.where(~fg_mask)[0]
            bg_candidates = bg_candidates[np.argsort(patch_activations[bg_candidates])]
            fg_indices = self._farthest_point_sample(fg_candidates, grid_size, n_fg)
            bg_indices = self._farthest_point_sample(bg_candidates, grid_size, n_bg)
        else:  # "topk"
            fg_indices = sorted_idx[-n_fg:][::-1]  # highest activation first
            bg_indices = sorted_idx[:n_bg]          # lowest activation first

        def to_pixel(patch_idx: int) -> list[float]:
            row, col = divmod(int(patch_idx), grid_size)
            cx = (col + 0.5) / grid_size * width
            cy = (row + 0.5) / grid_size * height
            return [cx, cy]

        fg_points = np.array([to_pixel(i) for i in fg_indices], dtype=np.float32)
        bg_points = np.array([to_pixel(i) for i in bg_indices], dtype=np.float32)
        return fg_points, bg_points

    # ------------------------------------------------------------------
    # IoU and deduplication
    # ------------------------------------------------------------------

    def _iou(self, mask_a: np.ndarray, mask_b: np.ndarray) -> float:
        """Intersection-over-Union between two boolean masks."""
        intersection = int((mask_a & mask_b).sum())
        union = int((mask_a | mask_b).sum())
        return float(intersection) / float(union) if union > 0 else 0.0

    def _deduplicate_masks(
        self,
        masks: list[np.ndarray],
        scores: list[float],
    ) -> tuple[list[np.ndarray], list[float]]:
        """Remove masks that overlap too much with a higher-scoring mask.

        Processes masks in descending score order; discards any mask whose IoU
        with an already-accepted mask exceeds ``iou_dedup_threshold``.
        """
        if not masks:
            return [], []

        order = np.argsort(scores)[::-1]
        kept_masks: list[np.ndarray] = []
        kept_scores: list[float] = []

        for idx in order:
            mask = masks[idx]
            if all(
                self._iou(mask, kept) < self._seg_config.iou_dedup_threshold
                for kept in kept_masks
            ):
                kept_masks.append(mask)
                kept_scores.append(scores[idx])

        return kept_masks, kept_scores

    # ------------------------------------------------------------------
    # SAM
    # ------------------------------------------------------------------

    def _run_sam(
        self,
        image_np: np.ndarray,
        attention_map: np.ndarray,
        n_patches: int,
    ) -> tuple[list[np.ndarray], list[float]]:
        """Run SAM with attention-derived prompts; return deduped masks + IoU scores.

        For each foreground prompt point, runs ``SamPredictor.predict()``
        with that point plus all background points.  Selects the candidate
        mask with the highest predicted IoU, filters by minimum area, then
        deduplicates across prompts.

        Args:
            image_np: uint8 RGB array of shape (H, W, 3) at native resolution.
            attention_map: float32, shape (H, W) — already at native resolution.
            n_patches: number of patch tokens (used to derive grid_size).

        Returns:
            (instance_masks, iou_scores) — parallel lists after deduplication.
        """
        predictor = self._get_sam_predictor()
        predictor.set_image(image_np)

        fg_points, bg_points = self._patch_centroids(attention_map, n_patches)
        h, w = image_np.shape[:2]
        min_area = self._seg_config.min_mask_area_fraction * h * w

        all_masks: list[np.ndarray] = []
        all_scores: list[float] = []

        for fg_pt in fg_points:
            # Combine this fg point with all bg points
            point_coords = np.vstack([fg_pt[np.newaxis, :], bg_points])  # (1+n_bg, 2)
            point_labels = np.array([1] + [0] * len(bg_points), dtype=np.int32)

            masks, iou_scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )
            # Select the mask with the highest predicted IoU
            best_idx = int(np.argmax(iou_scores))
            mask = masks[best_idx].astype(bool)

            if mask.sum() >= min_area:
                all_masks.append(mask)
                all_scores.append(float(iou_scores[best_idx]))

        if not all_masks:
            return [], []

        return self._deduplicate_masks(all_masks, all_scores)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def segment(self, image_path: Path, class_name: str) -> SegmentationResult:
        """Run full segmentation pipeline for a single image.

        Loads the image at native resolution, extracts DINOv3 patch tokens,
        computes the attention map, thresholds it, and runs SAM with the
        attention-derived prompt points.

        Args:
            image_path: Path to the image file (JPEG or PNG).
            class_name: The image's true class label (e.g. "mudcrack").
                        Must be present in ``classifier_model.classes_``.

        Returns:
            SegmentationResult with attention map, binary mask, and instance masks.
        """
        if self._seg_config.attention_mode == "classifier":
            if self._model is None:
                raise ValueError(
                    "classifier_model is required when attention_mode='classifier'. "
                    "Pass a fitted LogisticRegression or set attention_mode='self_attention'."
                )
            class_idx = list(self._model.classes_).index(class_name)
            cls_coef = self._model.coef_[class_idx]  # (768,)

        pil_img = Image.open(image_path).convert("RGB")
        image_size = pil_img.size  # (width, height) in PIL convention
        image_np = np.asarray(pil_img)  # (H, W, 3) uint8 for SAM

        patch_tokens = self._get_extractor().extract_patch_tokens(pil_img)
        n_patches = patch_tokens.shape[0]

        attention_map = self._compute_attention(patch_tokens, cls_coef, image_size)
        binary_mask = self._threshold(attention_map)
        instance_masks, iou_scores = self._run_sam(image_np, attention_map, n_patches)

        return SegmentationResult(
            attention_map=attention_map,
            binary_mask=binary_mask,
            instance_masks=instance_masks,
            iou_scores=iou_scores,
            image_path=image_path,
            class_name=class_name,
        )
