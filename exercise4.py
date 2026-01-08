import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from gensim.models import KeyedVectors

# charge ton modèle (adapte si besoin)
model = KeyedVectors.load("data/glove-wiki-gigaword-300.kv")

examples = [
    (["man", "hair"], []),
    (["hair", "man"], ["woman"]),
    (["mice", "city"], ["home"]),
    (["children", "goose"], ["child"]),
    (["paris", "belgium"], ["france"]),
    (["triceratops", "wolf"], ["deer"]),
]

for positive, negative in examples:
    try:
        result = model.most_similar(
            positive=positive,
            negative=negative,
            topn=1
        )[0][0]

        expr = " + ".join(positive)
        if negative:
            expr += " - " + " - ".join(negative)

        print(f"{expr} = {result}")
    except KeyError as e:
        print(f"Missing word in vocabulary: {e}")
