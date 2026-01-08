import os
import gensim.downloader as api
from gensim.models import KeyedVectors
import math
import numpy as np

# Path where you want to store/load the model
model_path = "glove-wiki-gigaword-300.kv"

# Load model from disk if exists, else download and save it
if os.path.exists(model_path):
    print("Loading model from local file...")
    model = KeyedVectors.load("data/"+model_path)
else:
    print("Downloading model...")
    model = api.load(model_path[:-3])
    model.save("data/"+model_path)
    print("Model downloaded and saved.")