import pandas as pd
import faiss
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from sentence_transformers import SentenceTransformer

df = pd.read_csv("data/historical_tickets.csv")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
X = embedder.encode(df["ticket_text"].tolist())

index = faiss.IndexFlatL2(X[0].shape[0])
index.add(np.array(X))
faiss.write_index(index, "retrieval/faiss_index.bin")
