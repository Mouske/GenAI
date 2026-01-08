import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from gensim.models import KeyedVectors

# charge ton modèle (adapte si besoin)
model = KeyedVectors.load("data/glove-wiki-gigaword-300.kv")

words = ["computer", "keyboard", "water", "ocean"]

# Récupération des vecteurs
vectors = {w: model[w] for w in words}

# Fonctions
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def euclidean_distance(v1, v2):
    return np.linalg.norm(v1 - v2)

# Matrices
cosine_matrix = np.zeros((len(words), len(words)))
euclidean_matrix = np.zeros((len(words), len(words)))

for i, w1 in enumerate(words):
    for j, w2 in enumerate(words):
        cosine_matrix[i, j] = cosine_similarity(vectors[w1], vectors[w2])
        euclidean_matrix[i, j] = euclidean_distance(vectors[w1], vectors[w2])

plt.figure()
plt.imshow(cosine_matrix)
plt.colorbar()
plt.xticks(range(len(words)), words)
plt.yticks(range(len(words)), words)
plt.title("Cosine Similarity")
plt.show()


plt.figure()
plt.imshow(euclidean_matrix)
plt.colorbar()
plt.xticks(range(len(words)), words)
plt.yticks(range(len(words)), words)
plt.title("Euclidean Distance")
plt.show()
