# Avital Rose 318413408
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

CONTEXT_SIZE = 5
EMBEDDING_DIM = 50
BATCH_SIZE = 512
PREFIX_SIZE = 3
SUFFIX_SIZE = 3

class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size, tags_size, prefix_size, suffix_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size + 1, embedding_dim)
        self.prefix_embedding = nn.Embedding(prefix_size + 1, embedding_dim)
        self.suffix_embedding = nn.Embedding(suffix_size + 1, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, tags_size)

    def forward(self, inputs, p_inputs, s_inputs):
        # m = nn.Dropout(p=0.3)
        embeds = self.embeddings(inputs.t())
        embeds = embeds.reshape(embeds.shape[0], CONTEXT_SIZE * EMBEDDING_DIM)
        embeds_p = self.prefix_embedding(p_inputs.t())
        embeds_p = embeds_p.reshape(embeds_p.shape[0], CONTEXT_SIZE * EMBEDDING_DIM)
        embeds_s = self.suffix_embedding(s_inputs.t())
        embeds_s = embeds_s.reshape(embeds_s.shape[0], CONTEXT_SIZE * EMBEDDING_DIM)
        embeds = embeds_p + embeds + embeds_s
        out = F.relu(self.linear1(embeds))
        # out = m(out)
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


def train(_model, _optimizer, _loss_function, vocab, ngrams, word_to_index, label_to_index, input_len, p_t_i, s_t_i):
    total_loss = 0
    train_correct = 0
    for idx, (context, target) in enumerate(ngrams):
        # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
        # into integer indices and wrap them in tensors)
        context_idxs = []
        for tup in context:
            context_idxs.append(tuple((word_to_index.get(w, 0) for w in tup)))
        context_idxs = torch.tensor(context_idxs, dtype=torch.long)

        # adding in prefix and suffix
        prefix_idxs = []
        for tup in context:
            prefix_idxs.append(tuple((p_t_i.get(make_prefix(w, PREFIX_SIZE), 0) for w in tup)))
        prefix_idxs = torch.tensor(prefix_idxs, dtype=torch.long)

        suffix_idxs = []
        for tup in context:
            suffix_idxs.append(tuple((s_t_i.get(make_suffix(w, SUFFIX_SIZE), 0) for w in tup)))
        suffix_idxs = torch.tensor(suffix_idxs, dtype=torch.long)

        # Step 2. Recall that torch *accumulates* gradients. Before passing in a
        # new instance, you need to zero out the gradients from the old
        # instance
        _model.zero_grad()

        # Step 3. Run the forward pass, getting log probabilities over next
        # words
        log_probs = _model(context_idxs, prefix_idxs, suffix_idxs)

        # Step 4. Compute your loss function. (Again, Torch wants the target
        # word wrapped in a tensor)
        t = tuple((label_to_index[t] for t in target))
        loss = _loss_function(log_probs, torch.tensor(t, dtype=torch.long))

        # Step 5. Do the backward pass and update the gradient
        loss.backward()
        _optimizer.step()
        # Get the Python number from a 1-element Tensor by calling tensor.item()
        total_loss += _loss_function(log_probs, torch.tensor(t, dtype=torch.long)).item()
        pred = log_probs.max(1, keepdim=True)[1]  # get the index of the max log-probability
        train_correct += pred.eq(torch.tensor(t, dtype=torch.long).view_as(pred)).cpu().sum().item()

    return total_loss/ input_len, (100 * train_correct) / input_len


def validate(_model, _loss_function, valid_ngrams, word_to_index, label_to_index, input_len, p_t_i, s_t_i):
    """

    :return:
    """
    valid_loss = 0
    valid_correct = 0
    for context, target in valid_ngrams:
        # log_probs = _model(context_idxs)
        # loss = _loss_function(log_probs, torch.tensor([label_to_index[target]], dtype=torch.long))
        # valid_loss += loss.item()
        # pred = log_probs.max(1, keepdim=True)[1]  # get the index of the max log-probability
        # if pred.item() == label_to_index[target]:
        #     valid_correct += 1

        context_idxs = []
        for tup in context:
            context_idxs.append(tuple((word_to_index.get(w, 0) for w in tup)))
        context_idxs = torch.tensor(context_idxs, dtype=torch.long)
        # adding in prefix and suffix
        prefix_idxs = []
        for tup in context:
            prefix_idxs.append(tuple((p_t_i.get(make_prefix(w, PREFIX_SIZE), 0) for w in tup)))
        prefix_idxs = torch.tensor(prefix_idxs, dtype=torch.long)

        suffix_idxs = []
        for tup in context:
            suffix_idxs.append(tuple((s_t_i.get(make_suffix(w, SUFFIX_SIZE), 0) for w in tup)))
        suffix_idxs = torch.tensor(suffix_idxs, dtype=torch.long)

        log_probs = _model(context_idxs, prefix_idxs, suffix_idxs)

        t = tuple((label_to_index[t] for t in target))
        loss = _loss_function(log_probs, torch.tensor(t, dtype=torch.long))
        # Get the Python number from a 1-element Tensor by calling tensor.item()
        valid_loss += _loss_function(log_probs, torch.tensor(t, dtype=torch.long)).item()
        pred = log_probs.max(1, keepdim=True)[1]  # get the index of the max log-probability
        valid_correct += pred.eq(torch.tensor(t, dtype=torch.long).view_as(pred)).cpu().sum().item()

    return valid_loss / input_len, (100 * valid_correct) / input_len


def transform_data_to_ngrams(file_name):
    """
    this function transforms the data to five grams and target, as well as finds the possible vocabulary and tags
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
    lines_split = [(i.split()[0], i.split()[1].split("\n")[0]) for i in lines if len(i) > 1]  # length of empty lines is 0
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
            sentences[j].append(i.split()[0])
            labels.append(i.split()[1].split("\n")[0])
        else:
            j += 1
            sentences.append([])
    # remove the end, empty sentences:
    sentences = sentences[:-1]

    # extract word without tag, add to sentence header and end
    sentences_with_header_and_end = []
    for sentence in sentences:
        sentences_with_header_and_end.append([minus_two_word, minus_one_word] + sentence + [plus_one_word, plus_two_word])


    # n gramed
    n_gramed_sentences = []
    for sentence in sentences_with_header_and_end:
        n_gramed_sentences.append([[sentence[i - 2], sentence[i - 1], sentence[i], sentence[i + 1],
                                   sentence[i + 2]] for i, word in enumerate(sentence[2:-2], start=2)])

    # back to one long list now
    run_on_sentence = []
    for sentence in n_gramed_sentences:
        run_on_sentence.extend(sentence)

    # put back with label
    zipped_features_and_labels = list(zip(run_on_sentence, labels))
    return zipped_features_and_labels, vocab_with_special_characters, pos_vocab


def tagger(task, train_file, valid_file, test_file):
    # prepare data
    five_grams, vocab, pos_tags = transform_data_to_ngrams(train_file)
    five_grams_valid, vocab_valid, pos_tags_valid = transform_data_to_ngrams(valid_file)


    # make word to index dictionary
    word_to_ix = {word: i for i, word in enumerate(vocab, start=1)}

    # make prefix and suffix to index dictionary
    prefix_to_ix, suffix_to_ix = make_suffix_and_prefix(vocab[2:-2])

    # # make label to dictionary loss
    label_to_ix = {label: i for i, label in enumerate(pos_tags)}

    # # load data
    train_loader = DataLoader(five_grams, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(five_grams_valid, batch_size=BATCH_SIZE, shuffle=True)


    # train
    losses = []
    accuracies = []
    losses_valid = []
    accuracies_valid = []
    lr = 0.2
    loss_function = nn.CrossEntropyLoss()
    model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE, len(label_to_ix.keys()),
                                 len(prefix_to_ix.keys()), len(suffix_to_ix.keys()))
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(5):
        print("epoch is: {}".format(epoch))
        loss_t, accuracy_t = train(model, optimizer, loss_function, vocab, train_loader, word_to_ix, label_to_ix,
                                   len(five_grams), prefix_to_ix, suffix_to_ix)
        losses.append(loss_t)
        accuracies.append(accuracy_t)
        loss_v, accuracy_v = validate(model, loss_function, valid_loader, word_to_ix, label_to_ix,
                                      len(five_grams_valid), prefix_to_ix, suffix_to_ix)
        losses_valid.append(loss_v)
        accuracies_valid.append(accuracy_v)

    print("accuracies are:", accuracies, "validation accuracies: ", accuracies_valid)

    plt.plot(losses, label="train")
    plt.plot(losses_valid, label="validation")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title("{}\nLoss according to epochs, learning rate = {}".format(task, lr))
    plt.legend()
    plt.show()

    plt.plot(accuracies, label="train")
    plt.plot(accuracies_valid, label="validation")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.title("{}\nAccuracy according to epochs, learning rate = {}".format(task,lr))
    plt.legend()
    plt.show()


def make_suffix_and_prefix(vocabulary):
    """
    Function to take the vocabulary and engineer suffix's and prefixes
    :param vocabulary:
    :return:
    """
    suffixes_list = []
    prefixes_list = []
    N = 3
    for word in vocabulary:
        prefixes_list.append(make_prefix(word, N))
        suffixes_list.append(make_suffix(word, N))

    prefixes_list = list(set(prefixes_list))
    suffixes_list = list(set(suffixes_list))

    prefix_to_index = {pre: i for i, pre in enumerate(prefixes_list, start=1)}
    suffix_to_index = {suf: i for i, suf in enumerate(suffixes_list, start=1)}

    return prefix_to_index, suffix_to_index


def make_suffix(w, n):
    """
    Function to return word suffix, if word smaller than 3, returns smaller suffix
    :param n:
    :param w:
    :return:
    """
    return w[-n:]


def make_prefix(w, n):
    """
    Function to return word prefix, if word smaller than 3, returns smaller prefix
    :param n:
    :param w:
    :return:
    """
    return w[:n]


if __name__ == "__main__":
    tagger("POS tagger", train_file="pos/train", valid_file="pos/dev", test_file="pos/test")
    tagger("NER tagger", train_file="ner/train", valid_file="ner/dev", test_file="ner/test")

"""
To do:
* batches and tensors instead of vectors, drop out, experiment with optimizers and hyper parameters
* dev data and test data
* check on NER- add proper accuracy check for it
* what to do with words not in train? right now it is getting the first row always. is dict.get(k) slower than dict[k]? 
* add test and write to file
* answer questions for part 1- first thing tomorrow morning
* add sentence spaces in correct place
* add documentation 
* should be random vector if no match, not same one at 0
"""