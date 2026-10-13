"""OCR pipeline reused from ocr_app.py: same preprocessing (grayscale,
polarity fix, 2x upscale, CLAHE, blur) and the same line-grouping heuristic
that stitches RapidOCR's word boxes back into reading-order rows.
"""

import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR


class OCREngine:
    def __init__(self):
        self.ocr = None

    @property
    def is_loaded(self):
        return self.ocr is not None

    def load(self):
        if self.ocr is None:
            self.ocr = RapidOCR()

    def release(self):
        self.ocr = None

    @staticmethod
    def preprocess(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if np.mean(gray) < 127:
            gray = cv2.bitwise_not(gray)

        h, w = gray.shape[:2]
        scaled = cv2.resize(gray, (int(w * 2.0), int(h * 2.0)), interpolation=cv2.INTER_CUBIC)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(scaled)

        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        return cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)

    def extract_text(self, frame):
        if self.ocr is None:
            raise RuntimeError("OCREngine.load() must be called before extract_text()")

        processed = self.preprocess(frame)
        result, _ = self.ocr(processed)
        if not result:
            return ""

        items = []
        for item in result:
            box = item[0]
            text = item[1].strip()
            if not text:
                continue

            xs = [pt[0] for pt in box]
            ys = [pt[1] for pt in box]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            items.append({
                "text": text,
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
                "y_center": (y_min + y_max) / 2.0,
                "height": max(1.0, y_max - y_min),
            })

        items.sort(key=lambda it: it["y_center"])

        lines = []
        for item in items:
            placed = False
            for line in lines:
                line_y_min = min(m["y_min"] for m in line)
                line_y_max = max(m["y_max"] for m in line)
                line_y_center = (line_y_min + line_y_max) / 2.0
                line_h = max(1.0, line_y_max - line_y_min)

                overlap = min(item["y_max"], line_y_max) - max(item["y_min"], line_y_min)
                min_h = min(item["height"], line_h)
                overlap_ratio = (overlap / min_h) if min_h > 0 else 0

                center_diff = abs(item["y_center"] - line_y_center)
                avg_h = (item["height"] + line_h) / 2.0

                if overlap_ratio >= 0.50 and center_diff <= (0.60 * avg_h):
                    line.append(item)
                    placed = True
                    break

            if not placed:
                lines.append([item])

        lines.sort(key=lambda line: min(m["y_center"] for m in line))

        output_rows = []
        for line in lines:
            line.sort(key=lambda it: it["x_min"])
            output_rows.append("   ".join(it["text"] for it in line))

        return "\n".join(output_rows)
