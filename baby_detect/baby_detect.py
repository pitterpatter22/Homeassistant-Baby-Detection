#!/usr/bin/env python3
"""Poll a UniFi Protect camera snapshot, detect a baby (person) in it, and
report the result to a Home Assistant webhook.

Detection uses a YOLOv8n ONNX model on CPU via onnxruntime. COCO has no
"baby" class, so this looks for the "person" class, optionally limited to a
region of interest (e.g. the crib) so adults walking past are ignored.

Configuration comes from environment variables (see .env.example). When run
as a Home Assistant app, the app's options in /data/options.json are loaded
into the environment first, keyed by the upper-cased option name.
"""

import json

import io
import logging
import os
import signal
import sys
import time

import numpy as np
import onnxruntime as ort
import requests
import urllib3
from PIL import Image

log = logging.getLogger("baby_detect")

PERSON_CLASS = 0  # COCO class index for "person"
OPTIONS_FILE = "/data/options.json"  # written by the Home Assistant Supervisor
DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov8n.onnx")


def load_ha_options(path=OPTIONS_FILE):
    """Export Home Assistant app options as environment variables, if present."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        options = json.load(f)
    for key, value in options.items():
        if value is None:
            continue
        if isinstance(value, bool):
            value = "true" if value else "false"
        os.environ[key.upper()] = str(value)


def env(name, default=None, cast=str):
    value = os.environ.get(name, default)
    if value is None or value == "":
        sys.exit(f"Missing required setting: {name}")
    return cast(value)


def env_bool(name, default):
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


class Config:
    def __init__(self):
        self.snapshot_url = env("SNAPSHOT_URL")
        self.unifi_api_key = os.environ.get("UNIFI_API_KEY")  # optional, for the Protect integration API
        self.verify_ssl = env_bool("VERIFY_SSL", False)  # UniFi consoles use self-signed certs
        self.webhook_url = env("HA_WEBHOOK_URL")
        self.ha_verify_ssl = env_bool("HA_VERIFY_SSL", False)  # e.g. self-signed or private-CA cert
        self.model_path = env("MODEL_PATH", DEFAULT_MODEL)
        self.interval = env("INTERVAL_SECONDS", "10", float)
        self.threshold = env("CONFIDENCE", "0.40", float)
        # Consecutive matching frames needed before the reported state flips.
        self.on_frames = env("FRAMES_TO_DETECT", "2", int)
        self.off_frames = env("FRAMES_TO_CLEAR", "3", int)
        # Re-send the current state this often even if unchanged (covers HA restarts).
        self.heartbeat = env("HEARTBEAT_SECONDS", "300", float)
        # Optional region of interest as fractions of the image: "x1,y1,x2,y2".
        roi = os.environ.get("ROI")
        self.roi = tuple(float(v) for v in roi.split(",")) if roi else None
        if self.roi and len(self.roi) != 4:
            sys.exit("ROI must be four comma-separated fractions: x1,y1,x2,y2")


class PersonDetector:
    def __init__(self, model_path, threshold, roi=None):
        opts = ort.SessionOptions()
        # Keep CPU usage low: a single thread is plenty for one frame every few seconds.
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.size = inp.shape[2] if isinstance(inp.shape[2], int) else 640
        self.threshold = threshold
        self.roi = roi

    def _letterbox(self, img):
        """Resize keeping aspect ratio and pad to a square, as YOLO expects."""
        w, h = img.size
        scale = self.size / max(w, h)
        nw, nh = round(w * scale), round(h * scale)
        canvas = Image.new("RGB", (self.size, self.size), (114, 114, 114))
        pad_x, pad_y = (self.size - nw) // 2, (self.size - nh) // 2
        canvas.paste(img.resize((nw, nh), Image.BILINEAR), (pad_x, pad_y))
        arr = np.asarray(canvas, dtype=np.float32) / 255.0
        return arr.transpose(2, 0, 1)[None], scale, pad_x, pad_y

    def detect(self, img):
        """Return the best person confidence found (inside the ROI, if set), or 0.0."""
        img = img.convert("RGB")
        w, h = img.size
        tensor, scale, pad_x, pad_y = self._letterbox(img)
        # YOLOv8 output: (1, 84, N) -> rows are cx, cy, w, h, then 80 class scores.
        out = self.session.run(None, {self.input_name: tensor})[0][0]
        scores = out[4 + PERSON_CLASS]
        keep = scores >= self.threshold
        if not keep.any():
            return 0.0
        scores = scores[keep]
        if self.roi:
            # Map box centres back to original-image fractions and test against the ROI.
            cx = (out[0][keep] - pad_x) / scale / w
            cy = (out[1][keep] - pad_y) / scale / h
            x1, y1, x2, y2 = self.roi
            inside = (cx >= x1) & (cx <= x2) & (cy >= y1) & (cy <= y2)
            scores = scores[inside]
        return float(scores.max()) if scores.size else 0.0


def fetch_snapshot(session, cfg):
    headers = {"X-API-KEY": cfg.unifi_api_key} if cfg.unifi_api_key else {}
    resp = session.get(cfg.snapshot_url, headers=headers, timeout=10, verify=cfg.verify_ssl)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content))


def send_state(session, cfg, detected, confidence):
    resp = session.post(
        cfg.webhook_url,
        json={"detected": detected, "confidence": round(confidence, 3)},
        timeout=10,
        verify=cfg.ha_verify_ssl,
    )
    resp.raise_for_status()


def main():
    load_ha_options()
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    cfg = Config()
    if not (cfg.verify_ssl and cfg.ha_verify_ssl):
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    detector = PersonDetector(cfg.model_path, cfg.threshold, cfg.roi)
    http = requests.Session()

    running = True

    def stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    reported = None  # last state successfully sent to HA
    last_sent = 0.0
    streak_state, streak = None, 0
    confidence = 0.0

    log.info("Polling %s every %.0fs", cfg.snapshot_url, cfg.interval)
    while running:
        started = time.monotonic()
        try:
            confidence = detector.detect(fetch_snapshot(http, cfg))
            seen = confidence > 0
            streak = streak + 1 if seen == streak_state else 1
            streak_state = seen
            log.debug("person=%s confidence=%.2f streak=%d", seen, confidence, streak)

            needed = cfg.on_frames if seen else cfg.off_frames
            target = seen if streak >= needed else reported
            if reported is None and target is None:
                target = seen  # first run: report immediately rather than waiting

            due = time.monotonic() - last_sent >= cfg.heartbeat
            if target != reported or due:
                send_state(http, cfg, target, confidence)
                if target != reported:
                    log.info("Baby detected: %s (confidence %.2f)", target, confidence)
                reported, last_sent = target, time.monotonic()
        except Exception as exc:  # keep polling through camera/HA hiccups
            log.warning("Cycle failed: %s", exc)

        # Sleep in small steps so SIGTERM is handled promptly.
        remaining = cfg.interval - (time.monotonic() - started)
        while running and remaining > 0:
            time.sleep(min(remaining, 1.0))
            remaining -= 1.0

    log.info("Stopped")


if __name__ == "__main__":
    main()
