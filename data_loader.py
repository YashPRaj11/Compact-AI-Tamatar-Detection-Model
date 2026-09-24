import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf

from model_architecture import SCALE_NAMES, _normalize_anchor_dict


class DataLoader:
    """Load LabelMe/rectangle/polygon JSON annotations for the detector."""

    def __init__(
        self,
        image_dir: str,
        json_dir: str,
        class_names: Optional[List[str]] = None,
        image_size: Tuple[int, int] = (416, 416),
        image_sie: Optional[Tuple[int, int]] = None,
    ):
        self.image_dir = Path(image_dir)
        self.json_dir = Path(json_dir)
        self.image_size = image_sie if image_sie is not None else image_size
        self.class_names = list(class_names) if class_names is not None else []
        self.class_to_id = {}

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.json_dir.exists():
            raise FileNotFoundError(f"JSON directory not found: {self.json_dir}")

        self.image_files = sorted(
            [
                f
                for f in os.listdir(self.image_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".JPG".lower()))
            ]
        )
        print(f"Found {len(self.image_files)} images:")

        if class_names is None:
            self._detect_classes()
        else:
            self.class_to_id = {name: idx for idx, name in enumerate(self.class_names)}

        print(f"Classes: {self.class_names}")

    def _detect_classes(self):
        classes = set()
        for image_file in self.image_files:
            json_file = self.json_dir / (Path(image_file).stem + ".json")
            if not json_file.exists():
                continue
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                objects = data.get("shapes", data.get("objects", []))
                for obj in objects:
                    name = obj.get("class") or obj.get("class_name") or obj.get("label")
                    if name:
                        classes.add(name)
            except (OSError, json.JSONDecodeError):
                continue
        self.class_names = sorted(classes)
        self.class_to_id = {name: idx for idx, name in enumerate(self.class_names)}
        print(f"Auto-detected {len(self.class_names)} classes")

    def load_original_image(self, image_file: str) -> Tuple[np.ndarray, Tuple[int, int]]:
        image_path = self.image_dir / image_file
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, image.shape[:2]

    def letterbox_image(self, image: np.ndarray, pad_value: int = 114):
        target_h, target_w = self.image_size
        orig_h, orig_w = image.shape[:2]
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = max(1, int(round(orig_w * scale)))
        new_h = max(1, int(round(orig_h * scale)))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((target_h, target_w, 3), pad_value, dtype=np.uint8)
        pad_x = (target_w - new_w) / 2.0
        pad_y = (target_h - new_h) / 2.0
        left = int(round(pad_x))
        top = int(round(pad_y))
        canvas[top : top + new_h, left : left + new_w] = resized
        meta = {
            "orig_w": orig_w,
            "orig_h": orig_h,
            "scale": scale,
            "pad_x": pad_x,
            "pad_y": pad_y,
            "target_w": target_w,
            "target_h": target_h,
        }
        return canvas, meta

    def load_image(self, image_file: str):
        image, original_shape = self.load_original_image(image_file)
        letterboxed, meta = self.letterbox_image(image)
        return letterboxed, original_shape, meta

    def load_annotations(self, image_file: str) -> List[Dict]:
        json_file = self.json_dir / (Path(image_file).stem + ".json")
        if not json_file.exists():
            print(f"Warning: JSON not found for {image_file}")
            return []

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        image, original_shape = self.load_original_image(image_file)
        img_h, img_w = original_shape
        annotations = []
        objects = data.get("shapes", data.get("objects", []))

        for obj in objects:
            class_name = obj.get("class") or obj.get("class_name") or obj.get("label") or "unknown"
            if class_name not in self.class_to_id:
                continue
            bbox = self._extract_bbox(obj, img_h, img_w)
            if bbox is not None:
                annotations.append(
                    {
                        "bbox": bbox,
                        "class_id": self.class_to_id[class_name],
                        "class_name": class_name,
                    }
                )
        return annotations

    @staticmethod
    def _extract_bbox(obj: Dict, img_height: int, img_width: int) -> Optional[Tuple[float, float, float, float]]:
        if "bbox" in obj:
            bbox = obj["bbox"]
            if isinstance(bbox, dict):
                x = float(bbox.get("x", 0))
                y = float(bbox.get("y", 0))
                w = float(bbox.get("width", bbox.get("w", 0)))
                h = float(bbox.get("height", bbox.get("h", 0)))
            else:
                x, y, w, h = map(float, bbox[:4])
        elif "points" in obj and len(obj["points"]) >= 2:
            points = np.asarray(obj["points"], dtype=np.float32)
            x_min, y_min = points.min(axis=0)
            x_max, y_max = points.max(axis=0)
            x, y, w, h = float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)
        elif "polygon" in obj and len(obj["polygon"]) > 0:
            points = np.asarray(obj["polygon"], dtype=np.float32)
            x_min, y_min = points.min(axis=0)
            x_max, y_max = points.max(axis=0)
            x, y, w, h = float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)
        elif "keypoints" in obj and len(obj["keypoints"]) > 0:
            points = np.asarray(obj["keypoints"], dtype=np.float32)
            x_min, y_min = points.min(axis=0)
            x_max, y_max = points.max(axis=0)
            x, y, w, h = float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)
        else:
            return None

        if w <= 0 or h <= 0:
            return None

        x_center = np.clip((x + w / 2.0) / img_width, 0.0, 1.0)
        y_center = np.clip((y + h / 2.0) / img_height, 0.0, 1.0)
        w_norm = np.clip(w / img_width, 1e-6, 1.0)
        h_norm = np.clip(h / img_height, 1e-6, 1.0)
        return float(x_center), float(y_center), float(w_norm), float(h_norm)

    def _to_letterbox_boxes(self, boxes, meta):
        out = []
        tw, th = meta["target_w"], meta["target_h"]
        for xc, yc, w, h in boxes:
            px = xc * meta["orig_w"]
            py = yc * meta["orig_h"]
            pw = w * meta["orig_w"]
            ph = h * meta["orig_h"]

            px = px * meta["scale"] + meta["pad_x"]
            py = py * meta["scale"] + meta["pad_y"]
            pw = pw * meta["scale"]
            ph = ph * meta["scale"]

            out.append((px / tw, py / th, pw / tw, ph / th))
        return out

    @staticmethod
    def _wh_iou(gt_wh, anchors):
        gt = np.asarray(gt_wh, dtype=np.float32)[None, :]
        anc = np.asarray(anchors, dtype=np.float32)
        inter = np.minimum(gt[:, 0], anc[:, 0]) * np.minimum(gt[:, 1], anc[:, 1])
        union = gt[:, 0] * gt[:, 1] + anc[:, 0] * anc[:, 1] - inter + 1e-9
        return inter / union

    def estimate_anchors(self, num_anchors_per_scale=1, seed=42, iterations=80):
        """Run k-means on normalized GT box widths/heights and return 3 scales."""
        wh = []
        for image_file in self.image_files:
            for ann in self.load_annotations(image_file):
                _, _, w, h = ann["bbox"]
                wh.append([w, h])
        if not wh:
            raise ValueError("No bounding boxes found; cannot estimate anchors.")
        wh = np.asarray(wh, dtype=np.float32)

        k = int(num_anchors_per_scale) * 3
        rng = np.random.default_rng(seed)
        if len(wh) < k:
            raise ValueError(f"Need at least {k} boxes to estimate {k} anchors; found {len(wh)}.")

        centers = wh[rng.choice(len(wh), size=k, replace=False)].copy()
        for _ in range(iterations):
            ious = np.stack([self._wh_iou(x, centers) for x in wh], axis=0)
            assignments = np.argmax(ious, axis=1)
            new_centers = centers.copy()
            for j in range(k):
                members = wh[assignments == j]
                if len(members):
                    new_centers[j] = np.median(members, axis=0)
            if np.allclose(new_centers, centers, atol=1e-5):
                break
            centers = new_centers

        areas = centers[:, 0] * centers[:, 1]
        centers = centers[np.argsort(areas)]
        return {
            "fine": centers[:num_anchors_per_scale].astype(np.float32),
            "medium": centers[num_anchors_per_scale : 2 * num_anchors_per_scale].astype(np.float32),
            "coarse": centers[2 * num_anchors_per_scale :].astype(np.float32),
        }

    def create_target_tensor(self, boxes, class_ids, grid_sizes, anchors):
        """Create one raw-parameter target tensor per detection scale."""
        anchors = _normalize_anchor_dict(anchors)
        num_scales = len(grid_sizes)
        num_anchors = len(anchors["fine"])
        num_classes = len(self.class_names)

        targets = [
            np.zeros((g, g, num_anchors, 5 + num_classes), dtype=np.float32)
            for g in grid_sizes
        ]

        # Flatten anchors so every box can select its best shape prior.
        flat_anchors = []
        flat_map = []
        for s_idx, name in enumerate(SCALE_NAMES[:num_scales]):
            for a_idx, anchor in enumerate(anchors[name]):
                flat_anchors.append(anchor)
                flat_map.append((s_idx, a_idx))
        flat_anchors = np.asarray(flat_anchors, dtype=np.float32)

        occupied = set()
        for box, class_id in zip(boxes, class_ids):
            xc, yc, w, h = map(float, box)
            shape_iou = self._wh_iou([w, h], flat_anchors)[0]
            order = np.argsort(-shape_iou)

            selected = None
            for idx in order:
                s_idx, a_idx = flat_map[int(idx)]
                g = int(grid_sizes[s_idx])
                gx_f = xc * g
                gy_f = yc * g
                gx = min(max(int(np.floor(gx_f)), 0), g - 1)
                gy = min(max(int(np.floor(gy_f)), 0), g - 1)
                key = (s_idx, gy, gx, a_idx)
                if key not in occupied:
                    selected = (s_idx, a_idx, gx, gy, gx_f - gx, gy_f - gy, key)
                    break

            if selected is None:
                # Extremely dense fallback: overwrite the lowest-IoU assignment.
                idx = int(order[0])
                s_idx, a_idx = flat_map[idx]
                g = int(grid_sizes[s_idx])
                gx_f = xc * g
                gy_f = yc * g
                gx = min(max(int(np.floor(gx_f)), 0), g - 1)
                gy = min(max(int(np.floor(gy_f)), 0), g - 1)
                key = (s_idx, gy, gx, a_idx)
                selected = (s_idx, a_idx, gx, gy, gx_f - gx, gy_f - gy, key)

            s_idx, a_idx, gx, gy, tx, ty, key = selected
            anchor_w, anchor_h = anchors[SCALE_NAMES[s_idx]][a_idx]
            tw = np.log(max(w, 1e-8) / max(float(anchor_w), 1e-8))
            th = np.log(max(h, 1e-8) / max(float(anchor_h), 1e-8))

            target = targets[s_idx]
            target[gy, gx, a_idx, 0:4] = [tx, ty, tw, th]
            target[gy, gx, a_idx, 4] = 1.0
            target[gy, gx, a_idx, 5 + int(class_id)] = 1.0
            occupied.add(key)

        return targets

    def get_dataset(self, model=None, anchors=None):
        from model_architecture import create_model

        if model is None:
            if anchors is None:
                anchors = self.estimate_anchors()
            model = create_model(num_classes=len(self.class_names), anchors=anchors)
        if anchors is None:
            anchors = model.anchors_per_scale

        grid_sizes = model.grid_sizes
        images, targets = [], []
        for i, image_file in enumerate(self.image_files):
            if (i + 1) % 10 == 0:
                print(f"Loading {i + 1}/{len(self.image_files)}")
            image, _, meta = self.load_image(image_file)
            image = image.astype(np.float32) / 255.0
            annotations = self.load_annotations(image_file)
            boxes = [ann["bbox"] for ann in annotations]
            class_ids = [ann["class_id"] for ann in annotations]
            boxes_letterbox = self._to_letterbox_boxes(boxes, meta)
            target = self.create_target_tensor(boxes_letterbox, class_ids, grid_sizes, anchors)
            images.append(image)
            targets.append(target)
        return images, targets

    def visualize_sample(self, sample_idx=0):
        if sample_idx >= len(self.image_files):
            raise IndexError(f"Invalid sample index {sample_idx}")
        image_file = self.image_files[sample_idx]
        image, _, meta = self.load_image(image_file)
        annotations = self.load_annotations(image_file)
        boxes = self._to_letterbox_boxes([a["bbox"] for a in annotations], meta)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image)
        H, W = image.shape[:2]
        for ann, box in zip(annotations, boxes):
            xc, yc, w, h = box
            x = (xc - w / 2) * W
            y = (yc - h / 2) * H
            rect = patches.Rectangle((x, y), w * W, h * H, linewidth=2, edgecolor="red", facecolor="none")
            ax.add_patch(rect)
            ax.text(x, max(0, y - 5), ann["class_name"], color="red", fontsize=11,
                    bbox=dict(facecolor="white", alpha=0.7))
        ax.set_title(f"GT: {image_file}")
        ax.axis("off")
        plt.tight_layout()
        plt.show()

    def visualize_batch(self, num_samples=5):
        count = min(num_samples, len(self.image_files))
        for idx in np.random.default_rng(42).choice(len(self.image_files), count, replace=False):
            self.visualize_sample(int(idx))


class DataGenerator(tf.keras.utils.Sequence):
    """Batch generator with label-consistent geometric and photometric augmentation."""

    def __init__(self, images, targets, batch_size=8, augment=True, shuffle=True):
        super().__init__()
        self.images = images
        self.targets = targets  # sample-major: N x [fine, medium, coarse]
        self.batch_size = int(batch_size)
        self.augment = bool(augment)
        self.shuffle_enabled = bool(shuffle)
        self.indices = np.arange(len(images))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.images) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle_enabled:
            np.random.shuffle(self.indices)

    def __getitem__(self, batch_idx):
        idxs = self.indices[batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size]
        batch_images = np.stack([self.images[i].copy() for i in idxs], axis=0)
        batch_sample_targets = [[t.copy() for t in self.targets[i]] for i in idxs]

        if self.augment:
            batch_images, batch_sample_targets = self._augment_batch(batch_images, batch_sample_targets)

        batch_targets = [
            np.stack([sample_targets[s] for sample_targets in batch_sample_targets], axis=0)
            for s in range(len(SCALE_NAMES))
        ]
        return batch_images, batch_targets

    @staticmethod
    def _augment_batch(images, targets):
        out_images, out_targets = [], []
        for image, per_scale_targets in zip(images, targets):
            per_scale_targets = [t.copy() for t in per_scale_targets]

            if np.random.rand() < 0.5:
                image = np.ascontiguousarray(np.fliplr(image))
                for target in per_scale_targets:
                    # Mirror grid columns first, then mirror the encoded
                    # intra-cell x offset. This preserves the exact global
                    # center coordinate used by the decoder.
                    old = target.copy()
                    target[:] = old[:, ::-1, :, :]
                    ys, xs, aas = np.where(target[..., 4] > 0.5)
                    for y, x, a in zip(ys, xs, aas):
                        tx = float(target[y, x, a, 0])
                        new_tx = 1.0 - tx
                        target[y, x, a, 0] = np.clip(new_tx, 1e-5, 0.999999)

            if np.random.rand() < 0.35:
                brightness = np.random.uniform(0.9, 1.1)
                image = np.clip(image * brightness, 0.0, 1.0)

            if np.random.rand() < 0.35:
                contrast = np.random.uniform(0.9, 1.1)
                image = np.clip((image - 0.5) * contrast + 0.5, 0.0, 1.0)

            out_images.append(image)
            out_targets.append(per_scale_targets)

        return np.stack(out_images, axis=0), out_targets


def stack_targets_sample_major(targets):
    if not targets:
        return []
    return [
        np.stack([sample[s] for sample in targets], axis=0)
        for s in range(len(targets[0]))
    ]
