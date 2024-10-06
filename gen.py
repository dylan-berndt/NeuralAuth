
import cohere

import os
from dotenv import load_dotenv

import requests

import random

from contextlib import contextmanager

import numpy as np

import h5py


def project(x, y):
    return y * np.dot(x, y.T) / np.dot(y, y.T)


class Cohere:
    def __init__(self):
        load_dotenv()
        self.client = cohere.ClientV2(os.environ.get("COHERE_KEY"))

    def embed(self, text, query=False):
        response = self.client.embed(
            texts=text, model="embed-english-v3.0", input_type="search_" + ("query" if query else "document"),
            embedding_types=["float"]
        )

        return np.asarray(response.embeddings.float_)


class Generator:
    word_site = "https://www.mit.edu/~ecprice/wordlist.10000"

    response = requests.get(word_site)
    allWords = response.content.splitlines()
    words = [word.decode("utf-8") for word in allWords if len(word) > 3]

    client = Cohere()

    def __init__(self, amount):
        self.words = random.choices(Generator.words, k=amount)
        wordEmbeddings = Generator.client.embed([' '.join(self.words)])
        self.optionsEmbeddings = wordEmbeddings / np.linalg.norm(wordEmbeddings)

        self.projections = []

    def choose(self, choice):
        if choice in self.words:
            self.makeChoice(self.words.index(choice))
            return True
        elif choice.isnumeric():
            if -1 < int(choice) < len(self.words):
                self.makeChoice(int(choice))
                return True
        return False

    def makeChoice(self, choice):
        choiceEmbedding = Generator.client.embed([self.words[choice]], True)
        choiceEmbedding = choiceEmbedding / np.linalg.norm(choiceEmbedding)

        projection = project(choiceEmbedding, self.optionsEmbeddings)

        self.projections.append(projection[0])

        self.words = random.choices(Generator.words, k=len(self.words))

    def save(self, saveName=""):
        if saveName != "":
            with h5py.File('data.h5', 'a') as file:
                file.create_dataset(saveName, data=np.asarray(self.projections), compression='gzip')


if __name__ == '__main__':
    with Generator(9) as context:
        context.makeChoice(0)
        print(context.projections)



