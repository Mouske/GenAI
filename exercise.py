import numpy as np
from gensim.models import KeyedVectors

# charge ton modèle (adapte si besoin)
model = KeyedVectors.load("data/glove-wiki-gigaword-300.kv")

sentence = "I love learning"
words = sentence.lower().split()

for word in words:
    if word in model:
        vector = model[word]
        print(f"Word: {word}")
        print(f"Vector size: {vector.shape[0]}")
        print(f"Magnitude: {np.linalg.norm(vector):.4f}")
        print(f"First 10 values: {vector[:10]}")
        print("-" * 40)
    else:
        print(f"{word} not in vocabulary")
