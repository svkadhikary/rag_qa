import os
import faiss
import numpy as np
from logger import logging

logger = logging.getLogger("Vectorstore_logs")

class VectorStore:
    def __init__(self, dimension: int, index_path: str):
        self.dimension = dimension
        self.index_path = index_path
    
    def create_index(self, vectors: np.ndarray):

        try:
            index = faiss.IndexFlatL2(self.dimension)
            faiss.normalize_L2(vectors)
            index.add(vectors)

            logger.info("FAISS index created")
        except Exception as e:
            logger.exception("Vector indices cannot be created")
            return ""

        return index

def search(query_vector: np.ndarray, index: faiss.IndexFlatL2, top_k: int = 5):

    query_vector = query_vector.reshape(1, -1).astype('float32')
    faiss.normalize_L2(query_vector)
    distances, indices = index.search(query_vector, top_k)

    logger.info(f"Found {top_k} matches: {[(idx, dist) for idx, dist in zip(distances, indices)]}")

    return distances, indices
