"""Synthetic test scenes with known ground truth (no real photos needed)."""
import cv2
import numpy as np


def etched_scene(h=1000, w=750, seed=0, with_person=True, with_ornament=True):
    """Frosted grey glass, a dark door frame on the right, a purple 'person' on the
    left, and a thin-line etched ornament in the middle-right.

    Returns (rgb, ornament_mask, person_mask)."""
    rng = np.random.default_rng(seed)
    base = np.full((h, w), 150, np.float32)
    base += cv2.GaussianBlur(rng.normal(0, 14, (h, w)).astype(np.float32), (0, 0), 1.2)  # frost grain
    base += np.linspace(-10, 10, w)[None, :]  # soft lighting gradient
    img = np.dstack([base, base, base + 4]).clip(0, 255).astype(np.uint8)

    person = np.zeros((h, w), np.uint8)
    if with_person:
        cv2.ellipse(person, (int(w * 0.17), int(h * 0.55)), (int(w * 0.17), int(h * 0.42)), 0, 0, 360, 255, -1)
        cv2.circle(person, (int(w * 0.17), int(h * 0.1)), int(w * 0.07), 255, -1)
        shirt = cv2.GaussianBlur(rng.normal(0, 4, (h, w)).astype(np.float32), (0, 0), 3)
        for c, v in enumerate((95, 40, 110)):
            ch = img[..., c].astype(np.float32)
            ch[person > 0] = (v + shirt[person > 0])
            img[..., c] = ch.clip(0, 255).astype(np.uint8)
    cv2.rectangle(img, (int(w * 0.93), 0), (w, h), (35, 35, 38), -1)  # dark door frame

    orn = np.zeros((h, w), np.uint8)
    if with_ornament:
        cx = int(w * 0.62)
        pts = []
        for y in np.linspace(0.02, 0.88, 40):
            pts.append((int(cx + 0.06 * w * np.sin(y * 18)), int(y * h)))
        cv2.polylines(orn, [np.array(pts, np.int32)], False, 255, 6)
        for cy, r in ((0.18, 0.11), (0.42, 0.13), (0.68, 0.10)):
            cv2.ellipse(orn, (cx, int(cy * h)), (int(r * w), int(r * h * 0.55)), 0, 0, 360, 255, 5)
            cv2.ellipse(orn, (cx, int(cy * h)), (int(r * w * 0.55), int(r * h * 0.3)), 30, 0, 360, 255, 4)
            cv2.line(orn, (cx - int(r * w), int(cy * h)), (cx + int(r * w), int(cy * h)), 255, 4)
        for dy in (0.05, 0.3, 0.55, 0.8):
            cv2.rectangle(orn, (cx - int(0.1 * w), int(dy * h)), (cx + int(0.1 * w), int((dy + 0.08) * h)), 255, 4)
        orn = cv2.GaussianBlur(orn, (0, 0), 1.0)
        # etched lines look like lighter, slightly blurred strokes on frosted glass
        a = (orn.astype(np.float32) / 255)[..., None]
        img = (img * (1 - a * 0.85) + 238 * a * 0.85).clip(0, 255).astype(np.uint8)
    region = cv2.morphologyEx((orn > 20).astype(np.uint8) * 255, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61)))
    return img, region, person
