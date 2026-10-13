"""Enforces sequential loading of heavy models: at most one of
AuthEngine (InsightFace) / OCREngine (RapidOCR) is resident at a time,
matching the Raspberry Pi memory budget.
"""


class ModelManager:
    def __init__(self, auth_engine, ocr_engine):
        self.auth_engine = auth_engine
        self.ocr_engine = ocr_engine
        self._active = None

    def get_auth(self):
        if self._active == "ocr":
            self.ocr_engine.release()
        if self._active != "auth":
            self.auth_engine.load()
            self._active = "auth"
        return self.auth_engine

    def get_ocr(self):
        if self._active == "auth":
            self.auth_engine.release()
        if self._active != "ocr":
            self.ocr_engine.load()
            self._active = "ocr"
        return self.ocr_engine

    def release_all(self):
        self.auth_engine.release()
        self.ocr_engine.release()
        self._active = None
