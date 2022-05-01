import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

CONTEXT_SIZE = 5
EMBEDDING_DIM = 50


class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


def train(vocab, ngrmas, word_to_ix):

    losses = []
    loss_function = nn.NLLLoss()
    model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    for epoch in range(10):
        total_loss = 0
        for context, target in ngrams:

            # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
            # into integer indices and wrap them in tensors)
            context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)

            # Step 2. Recall that torch *accumulates* gradients. Before passing in a
            # new instance, you need to zero out the gradients from the old
            # instance
            model.zero_grad()

            # Step 3. Run the forward pass, getting log probabilities over next
            # words
            log_probs = model(context_idxs)

            # Step 4. Compute your loss function. (Again, Torch wants the target
            # word wrapped in a tensor)
            loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))

            # Step 5. Do the backward pass and update the gradient
            loss.backward()
            optimizer.step()

            # Get the Python number from a 1-element Tensor by calling tensor.item()
            total_loss += loss.item()
        losses.append(total_loss)
    print(losses)  # The loss decreased every iteration over the training data!


def transform_data_to_ngrams(file_name):
    """
    this function transforms the data to five grams and target
    :param file_name:
    :return:
    """
    # words to add beginning and end
    minus_one_word = "h%h"
    minus_two_word = "h%h%"
    plus_one_word = "%h"
    plus_two_word = "%h%h"

    # read file
    with open(file_name, "r") as f:
        lines = f.readlines()

    # make up vocabulary as well as possible tags
    lines_split = [(i.split(' ')[0], i.split(' ')[1][:-1]) for i in lines if len(i) > 1]  # length of empty lines is 0
    vocab_with_special_characters = set([j[0] for j in lines_split])  # ends up being 7896 words or 43127
    vocab_with_special_characters = [minus_two_word, minus_one_word] + list(vocab_with_special_characters) + \
                                    [plus_one_word, plus_two_word]
    pos_vocab = set([j[1] for j in lines_split])

    # create sentences- by removing empty spaces
    sentences = [[]]
    labels = []
    j = 0
    for i in lines:
        if len(i) > 1:
            sentences[j].append(i.split(' ')[0])
            labels.append(i.split(' ')[1][:-1])
        else:
            j += 1
            sentences.append([])
    # remove the end, empty sentences:
    sentences = sentences[:-1]

    print("length of sentences 1: ", len(sentences), sentences[0])
    print("length of labels 1:", len(labels))
    # extract word without tag, add to sentences header and end
    sentences_with_header_and_end = []
    for sentence in sentences:
        sentences_with_header_and_end.append([minus_two_word, minus_one_word] + sentence + [plus_one_word, plus_two_word])

    print("length of sentences 2: ", len(sentences_with_header_and_end), sentences_with_header_and_end[0])

    # ngramed
    ngramed_sentences = []
    for sentence in sentences_with_header_and_end:
        ngramed_sentences.append([[sentence[i - 2], sentence[i - 1], sentence[i], sentence[i + 1],
                         sentence[i + 2]] for i, word in enumerate(sentence[2:-2], start=2)])
    print("length of sentences 3: ", len(ngramed_sentences), ngramed_sentences[0])


    # back to one long list now
    run_on_sentence = []
    for sentence in ngramed_sentences:
        run_on_sentence.extend(sentence)

    print("length of words: ", len(run_on_sentence), run_on_sentence[0])

    # put back with label
    zipped_features_and_labels = list(zip(run_on_sentence, labels))
    print("length of zipped: ", len(zipped_features_and_labels), zipped_features_and_labels[1])


if __name__ == "__main__":
    five_grams = transform_data_to_ngrams("pos/train")
    # To get the embedding of a particular word, e.g. "beauty"

    #
    #
    # # We will use Shakespeare Sonnet 2
    #
    # # we should tokenize the input, but we will ignore that for now
    # # build a list of tuples.
    # # Each tuple is ([ word_i-CONTEXT_SIZE, ..., word_i-1 ], target word)
    # ngrams = [
    #     (
    #         [test_sentence[i - j - 1] for j in range(CONTEXT_SIZE)],
    #         test_sentence[i]
    #     )
    #     for i in range(CONTEXT_SIZE, len(test_sentence))
    # ]
    # # Print the first 3, just so you can see what they look like.
    # print(ngrams[:3])
    #
    # vocab = set(test_sentence)
    # word_to_ix = {word: i for i, word in enumerate(vocab)}
