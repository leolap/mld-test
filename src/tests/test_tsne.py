import pickle

import numpy as np

from src.etl.extract import extract_data
from src.etl.tsne import perform_tsne


def test_data_extraction():
    with open("src/data/mini_gm_public_v0.1.p", "rb") as f:
        data = pickle.load(f)
    embeddings, syndrome_ids = extract_data(data)
    assert len(embeddings) == len(syndrome_ids)
    assert isinstance(embeddings, np.ndarray)
    assert isinstance(syndrome_ids, np.ndarray)


def test_perform_tsne():
    fake_data = {
        "1": {"1": {"1": np.random.rand(320)}},
        "2": {"1": {"1": np.random.rand(320)}},
        "3": {"1": {"1": np.random.rand(320)}},
        "4": {"1": {"1": np.random.rand(320)}},
        "5": {"1": {"1": np.random.rand(320)}},
        "6": {"1": {"1": np.random.rand(320)}},
    }

    with open("fake_data.p", "wb") as f:
        pickle.dump(fake_data, f)

    with open("fake_data.p", "rb") as f:
        data = pickle.load(f)

    _, embeddings = extract_data(data)
    result = perform_tsne(embeddings, perplexity=5)

    assert len(result) == len(embeddings)
    assert all(isinstance(point, np.ndarray) and len(point) == 2 for point in result)
