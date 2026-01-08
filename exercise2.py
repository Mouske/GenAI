import numpy as np
from gensim.models import KeyedVectors

# charge ton modèle (adapte si besoin)
model = KeyedVectors.load("data/glove-wiki-gigaword-300.kv")


text = "I want to be a famous data scientist"

# 1) Tokenisation simple
words = text.lower().split()

# 2) Récupérer les vecteurs connus par le modèle
vectors = []
for word in words:
    if word in model:
        vectors.append(model[word])

# Sécurité : vérifier qu'on a bien des vecteurs
vectors = np.array(vectors)

print(f"Number of word vectors: {len(vectors)}")

# 3) Calcul du vecteur moyen
avg_vector = np.mean(vectors, axis=0)

print(f"Average vector size: {avg_vector.shape}")

# 4) Trouver les mots les plus proches du vecteur moyen
most_similar = model.most_similar(avg_vector, topn=10)

print("\nMost similar words to the average vector:")
for word, score in most_similar:
    print(f"{word:15s} {score:.4f}")
