### feedback_manager.py

import json
import os
import numpy as np
from scipy.spatial.distance import cosine

class FeedbackManager:
    def __init__(self, memory_file):
        self.memory_file = memory_file
        self.memory = {"normal": [], "deviation": []}
        self.similarity_threshold = 0.25  # relaxed threshold
        self.fixed_length = 30             # normalize to this length
        self.load_memory()

    def load_memory(self):
        if os.path.exists(self.memory_file):
            with open(self.memory_file, "r") as f:
                self.memory = json.load(f)

    def save_memory(self):
        with open(self.memory_file, "w") as f:
            json.dump(self.memory, f)

    def normalize(self, angles):
        angles = np.array(angles)
        if len(angles) == 0:
            return np.zeros(self.fixed_length).tolist()
        if len(angles) == self.fixed_length:
            return angles.tolist()
        return np.interp(np.linspace(0, len(angles)-1, self.fixed_length), np.arange(len(angles)), angles).tolist()

    def is_known_normal(self, angles):
        norm_angles = self.normalize(angles)
        for normal in self.memory["normal"]:
            distance = cosine(norm_angles, normal)
            print(f"[DEBUG] Normal match distance: {distance:.4f}")
            if distance < self.similarity_threshold:
                return True
        return False

    def is_known_deviation(self, angles):
        norm_angles = self.normalize(angles)
        for deviation in self.memory["deviation"]:
            distance = cosine(norm_angles, deviation)
            print(f"[DEBUG] Deviation match distance: {distance:.4f}")
            if distance < self.similarity_threshold:
                return True
        return False

    def add_normal(self, angles):
        norm = self.normalize(angles)
        self.memory["normal"].append(norm)

    def add_deviation(self, angles):
        norm = self.normalize(angles)
        self.memory["deviation"].append(norm)

    def should_log_cycle(self, status):
        return status == "Deviation"
