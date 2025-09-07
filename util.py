import json
import numpy as np


def compute_direction(start, end):
    start = np.array(start, dtype=float)
    end = np.array(end, dtype=float)
    vec = end - start
    norm = np.linalg.norm(vec)
    return vec / norm


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f)


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)
