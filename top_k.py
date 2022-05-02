# Avital Rose 318413408
import numpy as np
import torch
import torch.nn.functional as F

def most_similar(word, k):
    """
    Function that returns k most similar words (by cosine similarity) to given word
    :param word: (str) word to find similarity to
    :param k: (int) number of similar words to return
    :return:
    """
    # translate word to index
    word_index = word_to_ix[word]

    # get vector at index
    word_vector = vecs[word_index]
    dist = F.cosine_similarity(word_vector, vecs)
    index_sorted = torch.argsort(dist, descending=True)
    top_k = index_sorted[1:k + 1] # don't want to return actual word, which obviously be most close to self
    distances = list(torch.topk(dist, k+1).values.numpy())[1:]

    # index from vector
    similar_words = []
    for similar_word_vector in top_k:
        key = [k for k, v in word_to_ix.items() if v == similar_word_vector]
        similar_words.extend(key)

    return similar_words, distances


if __name__ == "__main__":
    # get wordVectors
    vecs = torch.from_numpy(np.loadtxt("wordVectors.txt"))

    # read all possible words
    with open("vocab.txt", "r") as f:
        lines = f.readlines()
    lines = [i.split("\n")[0] for i in lines]

    # create word to index dictionary
    word_to_ix = {word: i for i, word in enumerate(lines, start=1)}

    words = ["dog", "england", "john", "explode", "office"]
    for word in words:
        similar, sim_dist = most_similar(word, 5)
        print("most similar to: {}\nwords are: {}\ndistances are {}\n".format(word, similar, sim_dist))

"""
To do:
what about cosine similarity to word not in embedded matrix? 
can we change the signature of the function most_similar? 
what to do about words not in embedded matrix? (not in word vector)
make nicer sorted dist
including same word or not (for most similar)? 
"""