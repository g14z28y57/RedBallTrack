import json
import numpy as np
import pickle


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


def save_pickle(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def read_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)
