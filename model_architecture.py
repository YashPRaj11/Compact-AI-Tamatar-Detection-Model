import math
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model


SCALE_NAMES = ["fine", "medium", "coarse"]
DEFAULT_INPUT_SIZE = (416, 416)

DEFAULT_ANCHORS = {
    "fine": np.array([[0.10, 0.13]], dtype=np.float32),
    "medium": np.array([[0.28, 0.35]], dtype=np.float32),
    "coarse": np.array([[0.58, 0.64]], dtype=np.float32),
}

def _normalize_anchor_dict(anchors: Optional[Dict[str, np.ndarray]]):
    if anchors is None:
        anchors = DEFAULT_ANCHORS
    out = {}
    for name in SCALE_NAMES:
        arr = np.asarray(anchors[name], dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"Anchors for {name} must have shape [N,2], got {arr.shape}")
        out[name] = arr
    n = {len(v) for v in out.values()}
    if len(n) != 1:
        raise ValueError("Every detection scale must use the same number of anchors.")
    return out


@keras.utils.register_keras_serializable(package="Tomato")
class Lightweight(Model):

    def __init__(
        self,
        num_classes: int = 2,
        num_anchors: int = 1,
        input_shape: Tuple[int, int, int] = (416, 416, 3),
        anchors: Optional[Dict[str, np.ndarray]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_classes = int(num_classes)
        self.num_anchors = int(num_anchors)
        self.input_shape_val = tuple(input_shape)
        self.anchors_per_scale = _normalize_anchor_dict(anchors)

        if any(len(self.anchors_per_scale[s]) != self.num_anchors for s in SCALE_NAMES):
            raise ValueError("num_anchors does not match anchors_per_scale.")

        self.backbone = self._build_backbone(self.input_shape_val)
        self.grid_sizes = [int(out.shape[1]) for out in self.backbone.outputs]

        self.head_fine = self._build_head(self.grid_sizes[0], 48, "head_fine")
        self.head_medium = self._build_head(self.grid_sizes[1], 64, "head_medium")
        self.head_coarse = self._build_head(self.grid_sizes[2], 96, "head_coarse")
        self.detection_heads = [self.head_fine, self.head_medium, self.head_coarse]

    @staticmethod
    def _conv_bn_act(x, filters, kernel=3, stride=1, name=None):
        x = layers.Conv2D(filters, kernel, strides=stride, padding="same", use_bias=False, name=None if name is None else name + "_conv")(x)
        x = layers.BatchNormalization(name=None if name is None else name + "_bn")(x)
        return layers.Activation("swish", name=None if name is None else name + "_act")(x)

    @staticmethod
    def _dw_block(x, filters, stride=1, name=None):
        # A real 3x3 depthwise convolution preserves local spatial reasoning.
        x = layers.DepthwiseConv2D(3, strides=stride, padding="same", use_bias=False, name=None if name is None else name + "_dw")(x)
        x = layers.BatchNormalization(name=None if name is None else name + "_dw_bn")(x)
        x = layers.Activation("swish", name=None if name is None else name + "_dw_act")(x)
        x = layers.Conv2D(filters, 1, padding="same", use_bias=False, name=None if name is None else name + "_pw")(x)
        x = layers.BatchNormalization(name=None if name is None else name + "_pw_bn")(x)
        return layers.Activation("swish", name=None if name is None else name + "_pw_act")(x)

    def _build_backbone(self, input_shape):
        inputs = layers.Input(shape=input_shape, name="image")

        # 416 -> 208 -> 104
        x = self._conv_bn_act(inputs, 24, 3, 2, "stem1")
        x = self._conv_bn_act(x, 32, 3, 2, "stem2")

        # 104 -> 52 : fine features
        c3 = self._dw_block(x, 64, 2, "stage3_down")
        c3 = self._dw_block(c3, 64, 1, "stage3_refine")

        # 52 -> 26 : medium features
        c4 = self._dw_block(c3, 96, 2, "stage4_down")
        c4 = self._dw_block(c4, 96, 1, "stage4_refine")

        # 26 -> 13 : coarse features
        c5 = self._dw_block(c4, 128, 2, "stage5_down")
        c5 = self._dw_block(c5, 128, 1, "stage5_refine")

        # Lightweight top-down feature pyramid.
        p5 = layers.Conv2D(96, 1, padding="same", use_bias=False, name="fpn_p5_proj")(c5)
        p5 = layers.BatchNormalization(name="fpn_p5_bn")(p5)
        p5 = layers.Activation("swish", name="fpn_p5_act")(p5)

        up4 = layers.UpSampling2D(size=2, interpolation="nearest", name="fpn_up4")(p5)
        p4_lat = layers.Conv2D(64, 1, padding="same", use_bias=False, name="fpn_p4_lat")(c4)
        p4_lat = layers.BatchNormalization(name="fpn_p4_lat_bn")(p4_lat)
        p4 = layers.Concatenate(name="fpn_p4_concat")([p4_lat, up4])
        p4 = self._conv_bn_act(p4, 64, 3, 1, "fpn_p4_refine")

        up3 = layers.UpSampling2D(size=2, interpolation="nearest", name="fpn_up3")(p4)
        p3_lat = layers.Conv2D(48, 1, padding="same", use_bias=False, name="fpn_p3_lat")(c3)
        p3_lat = layers.BatchNormalization(name="fpn_p3_lat_bn")(p3_lat)
        p3 = layers.Concatenate(name="fpn_p3_concat")([p3_lat, up3])
        p3 = self._conv_bn_act(p3, 48, 3, 1, "fpn_p3_refine")

        return Model(inputs=inputs, outputs=[p3, p4, p5], name="backbone_fpn")

    def _build_head(self, grid_size, channels, name):
        inputs = layers.Input(shape=(grid_size, grid_size, None if False else channels), name=name + "_input")
        x = self._conv_bn_act(inputs, channels, 3, 1, name + "_stem")
        out_channels = self.num_anchors * (5 + self.num_classes)
        x = layers.Conv2D(out_channels, 1, padding="same", name=name + "_pred")(x)
        return Model(inputs=inputs, outputs=x, name=name)

    def call(self, inputs, training=None):
        p3, p4, p5 = self.backbone(inputs, training=training)
        raws = [
            self.head_fine(p3, training=training),
            self.head_medium(p4, training=training),
            self.head_coarse(p5, training=training),
        ]
        return [
            tf.reshape(r, [tf.shape(r)[0], r.shape[1], r.shape[2], self.num_anchors, 5 + self.num_classes])
            for r in raws
        ]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "num_anchors": self.num_anchors,
                "input_shape": list(self.input_shape_val),
                "anchors": {k: v.tolist() for k, v in self.anchors_per_scale.items()},
            }
        )
        return config


def decode_predictions(raw, grid, anchors):
    """Decode raw predictions into normalized [xc,yc,w,h,obj_logit,class_logits]."""
    anchors_tf = tf.constant(np.asarray(anchors, dtype=np.float32), dtype=tf.float32)
    gy, gx = tf.meshgrid(
        tf.range(grid, dtype=tf.float32),
        tf.range(grid, dtype=tf.float32),
        indexing="ij",
    )
    gx = tf.reshape(gx, (1, grid, grid, 1))
    gy = tf.reshape(gy, (1, grid, grid, 1))

    tx, ty = raw[..., 0], raw[..., 1]
    tw, th = raw[..., 2], raw[..., 3]
    obj_logit = raw[..., 4:5]
    class_logits = raw[..., 5:]

    bx = (tf.sigmoid(tx) + gx) / float(grid)
    by = (tf.sigmoid(ty) + gy) / float(grid)

    aw = tf.reshape(anchors_tf[:, 0], (1, 1, 1, -1))
    ah = tf.reshape(anchors_tf[:, 1], (1, 1, 1, -1))
    bw = aw * tf.exp(tf.clip_by_value(tw, -6.0, 6.0))
    bh = ah * tf.exp(tf.clip_by_value(th, -6.0, 6.0))

    return tf.concat([tf.stack([bx, by, bw, bh], axis=-1), obj_logit, class_logits], axis=-1)


def _ciou(pred, target, eps=1e-7):
    pred_cx, pred_cy, pred_w, pred_h = tf.unstack(pred, axis=-1)
    tgt_cx, tgt_cy, tgt_w, tgt_h = tf.unstack(target, axis=-1)

    pred_w = tf.maximum(pred_w, eps)
    pred_h = tf.maximum(pred_h, eps)
    tgt_w = tf.maximum(tgt_w, eps)
    tgt_h = tf.maximum(tgt_h, eps)

    px1, py1 = pred_cx - pred_w / 2.0, pred_cy - pred_h / 2.0
    px2, py2 = pred_cx + pred_w / 2.0, pred_cy + pred_h / 2.0
    tx1, ty1 = tgt_cx - tgt_w / 2.0, tgt_cy - tgt_h / 2.0
    tx2, ty2 = tgt_cx + tgt_w / 2.0, tgt_cy + tgt_h / 2.0

    ix1, iy1 = tf.maximum(px1, tx1), tf.maximum(py1, ty1)
    ix2, iy2 = tf.minimum(px2, tx2), tf.minimum(py2, ty2)
    inter = tf.maximum(ix2 - ix1, 0.0) * tf.maximum(iy2 - iy1, 0.0)

    union = pred_w * pred_h + tgt_w * tgt_h - inter + eps
    iou = tf.clip_by_value(inter / union, 0.0, 1.0)

    ex1, ey1 = tf.minimum(px1, tx1), tf.minimum(py1, ty1)
    ex2, ey2 = tf.maximum(px2, tx2), tf.maximum(py2, ty2)
    c2 = tf.square(ex2 - ex1) + tf.square(ey2 - ey1) + eps
    rho2 = tf.square(pred_cx - tgt_cx) + tf.square(pred_cy - tgt_cy)

    v = (4.0 / (math.pi ** 2)) * tf.square(
        tf.atan(tgt_w / tgt_h) - tf.atan(pred_w / pred_h)
    )
    alpha = tf.stop_gradient(v / (1.0 - iou + v + eps))
    ciou = 1.0 - iou + rho2 / c2 + alpha * v
    return ciou, iou


@keras.utils.register_keras_serializable(package="Tomato")
class ScaleLoss(keras.losses.Loss):
    """Serializable loss object for a single detection scale."""

    def __init__(self, grid_size, anchors, num_classes, name=None):
        super().__init__(name=name or f"loss_{grid_size}", reduction=keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        self.grid_size = int(grid_size)
        self.anchors = np.asarray(anchors, dtype=np.float32)
        self.num_classes = int(num_classes)

    def call(self, y_true, y_pred):
        anchors_np = self.anchors
        g = self.grid_size

        tx_t, ty_t, tw_t, th_t = [y_true[..., i] for i in range(4)]
        pos = tf.cast(y_true[..., 4] > 0.5, tf.float32)

        gy, gx = tf.meshgrid(
            tf.range(g, dtype=tf.float32),
            tf.range(g, dtype=tf.float32),
            indexing="ij",
        )
        gx = tf.reshape(gx, (1, g, g, 1))
        gy = tf.reshape(gy, (1, g, g, 1))

        # Decode prediction.
        pred_boxes = decode_predictions(y_pred, g, anchors_np)[..., :4]

        # Decode the encoded GT target using the same anchor prior.
        tx_abs = (tx_t + gx) / float(g)
        ty_abs = (ty_t + gy) / float(g)
        aw = tf.reshape(tf.constant(anchors_np[:, 0], tf.float32), (1, 1, 1, -1))
        ah = tf.reshape(tf.constant(anchors_np[:, 1], tf.float32), (1, 1, 1, -1))
        bw_true = aw * tf.exp(tf.clip_by_value(tw_t, -8.0, 8.0))
        bh_true = ah * tf.exp(tf.clip_by_value(th_t, -8.0, 8.0))
        target_boxes = tf.stack([tx_abs, ty_abs, bw_true, bh_true], axis=-1)

        num_pos = tf.reduce_sum(pos, axis=[1, 2, 3])
        normalizer = tf.maximum(num_pos, 1.0)

        ciou, iou = _ciou(pred_boxes, target_boxes)
        box_loss = tf.reduce_sum(ciou * pos, axis=[1, 2, 3]) / normalizer

        # IoU-aware positive objectness, bounded away from zero so that early
        # poor localization does not immediately suppress a real tomato.
        obj_target = tf.where(
            pos > 0.5,
            0.5 + 0.5 * tf.stop_gradient(iou),
            tf.zeros_like(iou),
        )
        obj_bce = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=obj_target,
            logits=y_pred[..., 4],
        )
        pos_obj = tf.reduce_sum(obj_bce * pos, axis=[1, 2, 3]) / normalizer

        neg = 1.0 - pos
        num_neg = tf.reduce_sum(neg, axis=[1, 2, 3])
        neg_normalizer = tf.maximum(num_neg, 1.0)
        neg_obj = tf.reduce_sum(obj_bce * neg, axis=[1, 2, 3]) / neg_normalizer

        if self.num_classes > 1:
            class_ce = tf.nn.softmax_cross_entropy_with_logits(
                labels=y_true[..., 5:],
                logits=y_pred[..., 5:],
            )
        else:
            class_ce = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=y_true[..., 5],
                logits=y_pred[..., 5],
            )
        class_loss = tf.reduce_sum(class_ce * pos, axis=[1, 2, 3]) / normalizer

        lambda_box = 10.0
        lambda_pos_obj = 5.0
        lambda_neg_obj = 0.40
        lambda_class = 1.0
        total = lambda_box * box_loss + lambda_pos_obj * pos_obj + lambda_neg_obj * neg_obj + lambda_class * class_loss
        return tf.reduce_mean(total)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "grid_size": self.grid_size,
                "anchors": self.anchors.tolist(),
                "num_classes": self.num_classes,
            }
        )
        return config


def loss(y_true, y_pred):
    """Backward-compatible fallback loss; create_model() uses ScaleLoss."""
    return tf.reduce_mean(tf.square(y_pred[..., :4] - y_true[..., :4]))



def create_model(
    num_classes=2,
    anchors: Optional[Dict[str, np.ndarray]] = None,
    learning_rate=3e-4,
    input_shape=(416, 416, 3),
):
    anchors = _normalize_anchor_dict(anchors)
    num_anchors = len(anchors["fine"])
    model = Lightweight(
        num_classes=num_classes,
        num_anchors=num_anchors,
        input_shape=input_shape,
        anchors=anchors,
    )

    dummy = tf.zeros((1,) + tuple(input_shape), dtype=tf.float32)
    _ = model(dummy, training=False)

    losses_per_scale = [
        ScaleLoss(
            grid_size=model.grid_sizes[i],
            anchors=anchors[SCALE_NAMES[i]],
            num_classes=num_classes,
            name=f"loss_{SCALE_NAMES[i]}",
        )
        for i in range(3)
    ]

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=float(learning_rate), clipnorm=5.0),
        loss=losses_per_scale,
    )
    return model


def _letterbox_rgb(image, target_size=(416, 416), pad_value=114):
    target_h, target_w = target_size
    orig_h, orig_w = image.shape[:2]
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((target_h, target_w, 3), pad_value, dtype=np.uint8)
    pad_x = (target_w - new_w) / 2.0
    pad_y = (target_h - new_h) / 2.0
    left, top = int(round(pad_x)), int(round(pad_y))
    canvas[top:top + new_h, left:left + new_w] = resized
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


def _map_box_from_letterbox_to_original(xc, yc, w, h, meta):
    # Box is normalized in letterboxed coordinates.
    px = xc * meta["target_w"]
    py = yc * meta["target_h"]
    pw = w * meta["target_w"]
    ph = h * meta["target_h"]

    px = (px - meta["pad_x"]) / meta["scale"]
    py = (py - meta["pad_y"]) / meta["scale"]
    pw = pw / meta["scale"]
    ph = ph / meta["scale"]

    xcn = px / meta["orig_w"]
    ycn = py / meta["orig_h"]
    wn = pw / meta["orig_w"]
    hn = ph / meta["orig_h"]
    return xcn, ycn, wn, hn


def _nms_class_aware(detections, iou_threshold=0.5, max_detections=100):
    if not detections:
        return []

    kept = []
    class_ids = sorted({int(d["class_id"]) for d in detections})
    for class_id in class_ids:
        group = [d for d in detections if int(d["class_id"]) == class_id]
        group.sort(key=lambda d: d["score"], reverse=True)
        while group:
            best = group.pop(0)
            kept.append(best)
            survivors = []
            bx = best["x_center"]
            by = best["y_center"]
            bw = best["width"]
            bh = best["height"]
            bx1, by1 = bx - bw / 2.0, by - bh / 2.0
            bx2, by2 = bx + bw / 2.0, by + bh / 2.0

            for d in group:
                x = d["x_center"]
                y = d["y_center"]
                w = d["width"]
                h = d["height"]
                x1 = max(bx1, x - w / 2.0)
                y1 = max(by1, y - h / 2.0)
                x2 = min(bx2, x + w / 2.0)
                y2 = min(by2, y + h / 2.0)
                inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                area_a = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
                area_b = max(0.0, (x + w / 2.0) - (x - w / 2.0)) * max(0.0, (y + h / 2.0) - (y - h / 2.0))
                iou = inter / (area_a + area_b - inter + 1e-7)
                if iou <= iou_threshold:
                    survivors.append(d)
            group = survivors

    kept.sort(key=lambda d: d["score"], reverse=True)
    return kept[:max_detections]


def predict_objects(
    model,
    image_path,
    confidence_threshold=0.25,
    iou_threshold=0.5,
    max_detections=100,
):
    """Run inference, decode all scales, apply class-aware NMS, and return original-image normalized boxes."""
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    lb, meta = _letterbox_rgb(image_rgb, DEFAULT_INPUT_SIZE)
    x = lb.astype(np.float32) / 255.0
    x = np.expand_dims(x, 0)

    raw_outputs = model.predict(x, verbose=0)
    candidates = []

    for scale_idx, scale_name in enumerate(SCALE_NAMES):
        raw = raw_outputs[scale_idx]
        grid = model.grid_sizes[scale_idx]
        anchors = model.anchors_per_scale[scale_name]
        decoded = decode_predictions(tf.convert_to_tensor(raw), grid, anchors).numpy()[0]

        obj_prob = 1.0 / (1.0 + np.exp(-raw[0, ..., 4]))
        class_logits = raw[0, ..., 5:]
        if class_logits.shape[-1] == 1:
            class_probs = 1.0 / (1.0 + np.exp(-class_logits[..., 0]))
            class_ids = np.zeros_like(class_probs, dtype=np.int32)
            class_conf = class_probs
        else:
            class_probs = tf.nn.softmax(class_logits, axis=-1).numpy()
            class_ids = np.argmax(class_probs, axis=-1)
            class_conf = np.max(class_probs, axis=-1)

        scores = obj_prob * class_conf
        yy, xx, aa = np.where(scores >= confidence_threshold)
        for y, xidx, a in zip(yy.tolist(), xx.tolist(), aa.tolist()):
            box = decoded[y, xidx, a, :4]
            xc, yc, w, h = [float(v) for v in box]
            xc, yc, w, h = _map_box_from_letterbox_to_original(xc, yc, w, h, meta)

            candidates.append(
                {
                    "scale": scale_name,
                    "x_center": float(xc),
                    "y_center": float(yc),
                    "width": float(w),
                    "height": float(h),
                    "objectness": float(obj_prob[y, xidx, a]),
                    "class_id": int(class_ids[y, xidx, a]),
                    "class_confidence": float(class_conf[y, xidx, a]),
                    "score": float(scores[y, xidx, a]),
                }
            )

    return _nms_class_aware(candidates, iou_threshold=iou_threshold, max_detections=max_detections)


def decode_all_scales(model, raw_outputs):
    """Convenience function for debugging/visualization."""
    decoded = []
    for i, name in enumerate(SCALE_NAMES):
        decoded.append(decode_predictions(raw_outputs[i], model.grid_sizes[i], model.anchors_per_scale[name]))
    return decoded


if __name__ == "__main__":
    model = create_model(num_classes=2)
    model.summary()
    print("Grid sizes [fine, medium, coarse]:", model.grid_sizes)
    print("Anchors:", model.anchors_per_scale)
