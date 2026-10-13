"""Face detection/verification via InsightFace buffalo_s (ArcFace 512D) with
a passive CNN liveness check (insightface's built-in `liveness` addon).

buffalo_s only returns 5 coarse keypoints (eye centers, nose, mouth corners),
not the eye-contour points a real Eye-Aspect-Ratio blink check needs, so
spoof resistance here comes from the liveness addon's single-frame
anti-spoofing classifier instead of blink detection. It reuses the same 5
keypoints buffalo_s already produces, so no extra model/dependency is needed
beyond one small (~1.4MB) verified download.

Verification is matched against a core.profile_store.ProfileStore, which can
hold more than one enrolled person's embedding (see core/profile_store.py
and the in-app "Enroll New Person" flow in main.py).

Heavy model (FaceAnalysis) is lazily loaded via load() and must be released()
before the OCR pipeline loads, so only one heavy model is resident at a time.
"""

import gc

import numpy as np


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class AuthResult:
    def __init__(self, face_found, is_live=None, live_score=None,
                 similarity=None, matched=False, matched_name=None, bbox=None):
        self.face_found = face_found
        self.is_live = is_live
        self.live_score = live_score
        self.similarity = similarity
        self.matched = matched
        self.matched_name = matched_name
        self.bbox = bbox


class AuthEngine:
    def __init__(self, profile_store, sim_threshold=0.50,
                 liveness_threshold=0.8, det_size=(320, 320)):
        self.profile_store = profile_store
        self.sim_threshold = sim_threshold
        self.liveness_threshold = liveness_threshold
        self.det_size = det_size
        self.app = None

    @property
    def has_profile(self):
        return not self.profile_store.is_empty

    @property
    def is_loaded(self):
        return self.app is not None

    def load(self):
        if self.app is not None:
            return
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(
            name="buffalo_s",
            providers=["CPUExecutionProvider"],
            addons=["liveness"],
            liveness_threshold=self.liveness_threshold,
        )
        app.prepare(ctx_id=0, det_size=self.det_size)
        self.app = app

    def release(self):
        self.app = None
        gc.collect()

    def detect(self, frame):
        """Returns the largest detected Face (bbox/kps/embedding/liveness),
        or None. Used directly by enrollment, which doesn't compare against
        the profile store."""
        if self.app is None:
            raise RuntimeError("AuthEngine.load() must be called before detect()")
        faces = self.app.get(frame)
        if not faces:
            return None
        return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    def verify(self, frame):
        face = self.detect(frame)
        if face is None:
            return AuthResult(face_found=False)

        liveness = face.liveness
        is_live = liveness["is_live"] if liveness else None
        live_score = liveness["live_score"] if liveness else None

        embedding = face.embedding
        similarity = None
        matched = False
        matched_name = None
        if embedding is not None and not self.profile_store.is_empty:
            matched_name, similarity = self.profile_store.best_match(embedding)
            matched = bool(is_live) and similarity >= self.sim_threshold

        return AuthResult(
            face_found=True,
            is_live=is_live,
            live_score=live_score,
            similarity=similarity,
            matched=matched,
            matched_name=matched_name if matched else None,
            bbox=face.bbox,
        )
