"""Named face-embedding registry: more than one person can be enrolled,
each identified by name, so the Gatekeeper can unlock for any of them.
Stored as one .npz file (one 512D array per person).
"""

import os

import numpy as np

from . import PROJECT_ROOT
from .auth_engine import cosine_similarity

DEFAULT_PATH = str(PROJECT_ROOT / "data" / "manager_profiles.npz")


class ProfileStore:
    def __init__(self, path=DEFAULT_PATH):
        self.path = path
        self._profiles = {}
        self.reload()

    def reload(self):
        if os.path.exists(self.path):
            with np.load(self.path) as data:
                self._profiles = {name: data[name] for name in data.files}
        else:
            self._profiles = {}

    def _save(self):
        directory = os.path.dirname(self.path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        np.savez(self.path, **self._profiles)

    @property
    def names(self):
        return list(self._profiles.keys())

    @property
    def is_empty(self):
        return len(self._profiles) == 0

    def add(self, name, embedding):
        name = (name or "").strip()
        if not name:
            raise ValueError("Name is required")
        self._profiles[name] = np.asarray(embedding, dtype=np.float32)
        self._save()

    def remove(self, name):
        if name in self._profiles:
            del self._profiles[name]
            self._save()

    def best_match(self, embedding):
        """Returns (name, similarity) of the closest enrolled profile, or
        (None, 0.0) if the store is empty."""
        best_name, best_sim = None, -1.0
        for name, ref in self._profiles.items():
            sim = cosine_similarity(embedding, ref)
            if sim > best_sim:
                best_sim, best_name = sim, name
        if best_name is None:
            return None, 0.0
        return best_name, best_sim
