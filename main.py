# Avital Rose 318413408
import numpy as np
import matplotlib.pyplot as plt
from torch import nn


class Network(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(input_size, 40)
        # Output layer, 36 units - one for each POS
        self.output = nn.Linear(40, 36)

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden(x)
        # Define tanh activation
        x = nn.Tanh(x)
        x = self.output(x)
        # and softmax output
        x = nn.Softmax(x)
        return x


def part_one():
    learning_rate = 0.01

    pass


def read_file_to_list(train_file):
    """
    This function converts a file to a parsed list of (word, POS) as well as adding special symbols.
    :return:
    """
    # need to remove some lines, by what criteria though?
    with open(train_file, 'r') as f:
        lines = f.readlines()

    # split into sentences
    sentences = [[]]
    j = 0
    for i in lines:
        if len(i) > 1:
            sentences[j].append(i)
        else:
            j+=1
            sentences.append()
    # split into tokens
    lines = [(i.split(' ')[0], i.split(' ')[1][:-1]) for i in lines if len(i) > 1]
    # make list of words
    vocab = set([i[0] for i in lines]) # ends up being 7896 words
    minus_one_word = "?%?"
    minus_two_word = "?%?%"
    plus_one_word = "%?%"
    plus_two_word = "%?%?"
    lines = [minus_two_word, minus_one_word] + lines + [plus_one_word, plus_two_word]
    vocab = [minus_two_word, minus_one_word] + vocab + [plus_one_word, plus_two_word]


def read_tags():
    """
    Function to create tags from pos_tags file
    :return:
    """
    with open("pos_tags.txt", 'r') as f:
        lines = f.readlines()

    tags = [i.split('.\t')[1].split('\t')[0] for i in lines]
    return tags


if __name__ == '__main__':
    read_file_to_list('pos/train')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
# import numpy as np
# def get_embedding_matrix(self):
#     """
#     Returns Embedding matrix
#     """
#     embedding_matrix = np.random.random(5 + 1, 10)
#     absent_words = 0
#     for word, i in zip(['cow','ball','duck','me','you'], range(5)).items():
#         embedding_vector = self.embedding_index.get(word)
#         if embedding_vector is not None:
#             # words not found in embedding index will be all-zeros.
#             embedding_matrix[i] = embedding_vector
#         else:
#             absent_words += 1
#     return embedding_matrix



"""
Questions to self:
    1. how to handle empty lines? 
    2. The words are one hot encoded or bi-grammed?
    3. Straight after the tanh there is softmax? or first there is another layer?
    4. how to handle ., etc?.	punctuation marks	. , ; ! - just dont tag them, use them only as words before or after
    5. How to handle sentence after sentence- sentence tokenization? will split by " " (space)
    
"""