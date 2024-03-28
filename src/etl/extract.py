import pickle
from typing import Dict, Tuple

import numpy as np


def load_data(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def extract_data(
    data: Dict[str, Dict[str, Dict[str, np.ndarray]]]
) -> Tuple[np.ndarray, np.ndarray]:
    embeddings = []
    syndrome_ids = []

    for syndrome_id, syndrome_data in data.items():
        for subject_data in syndrome_data.values():
            for encoding in subject_data.values():
                embeddings.append(encoding)
                syndrome_ids.append(syndrome_id)

    embeddings = np.array(embeddings)
    syndrome_ids = np.array(syndrome_ids)

    return syndrome_ids, embeddings
