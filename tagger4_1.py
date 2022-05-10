import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

CONTEXT_SIZE = 5
EMBEDDING_DIM = 50
CHAR_EMBEDDING_DIM = 30
CHAR_VOCAB_SIZE = 68
BATCH_SIZE = 1024
WORD_FIXED_LENGTH = 20


class NGramLanguageModeler(nn.Module):

    def __init__(self, embedding_matrix, tags_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding.from_pretrained(embedding_matrix.float())
        self.char_embedding = nn.Embedding(CHAR_VOCAB_SIZE + 1, CHAR_EMBEDDING_DIM)
        self.layer1 = torch.nn.Conv1d(1, 30, kernel_size=3, padding=1)
        self.max_pool = torch.nn.MaxPool1d(kernel_size=600, stride=1)
        self.linear1 = nn.Linear((CHAR_EMBEDDING_DIM + EMBEDDING_DIM) * CONTEXT_SIZE, 128)
        self.linear2 = nn.Linear(128, tags_size)

    def forward(self, inputs):
        # word embedding
        inputs_words = torch.stack(inputs[0]).t()
        word_embeds = self.embeddings(inputs_words)
        # word_embeds = word_embeds.reshape(word_embeds.shape[0], CONTEXT_SIZE * EMBEDDING_DIM)

        # char embedding preparation
        word_one_char_embedding = self.char_embedding(inputs[1][0])
        word_one_char = torch.flatten(word_one_char_embedding, start_dim=1).unsqueeze(1)
        word_two_char_embedding = self.char_embedding(inputs[1][1])
        word_two_char = torch.flatten(word_two_char_embedding, start_dim=1).unsqueeze(1)
        word_three_char_embedding = self.char_embedding(inputs[1][2])
        word_three_char = torch.flatten(word_three_char_embedding, start_dim=1).unsqueeze(1)
        word_four_char_embedding = self.char_embedding(inputs[1][3])
        word_four_char = torch.flatten(word_four_char_embedding, start_dim=1).unsqueeze(1)
        word_five_char_embedding = self.char_embedding(inputs[1][4])
        word_five_char = torch.flatten(word_five_char_embedding, start_dim=1).unsqueeze(1)

        # convolution for char
        word_one_convolution = torch.flatten(self.max_pool(self.layer1(word_one_char)), start_dim=1)
        word_two_convolution = torch.flatten(self.max_pool(self.layer1(word_two_char)), start_dim=1)
        word_three_convolution = torch.flatten(self.max_pool(self.layer1(word_three_char)), start_dim=1)
        word_four_convolution = torch.flatten(self.max_pool(self.layer1(word_four_char)), start_dim=1)
        word_five_convolution = torch.flatten(self.max_pool(self.layer1(word_five_char)), start_dim=1)

        # contact words and char
        words_combined_convolution = torch.stack((word_one_convolution, word_two_convolution, word_three_convolution,
                                                  word_four_convolution, word_five_convolution)).permute(2, 1, 0)
        word_embeds = word_embeds.permute(2, 0, 1)
        embeds = torch.cat((word_embeds, words_combined_convolution)).permute(1, 0, 2)
        embeds = embeds.reshape(embeds.shape[0], CONTEXT_SIZE * (EMBEDDING_DIM + CHAR_EMBEDDING_DIM))

        # next layer
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


class NGramsDataset(Dataset):
    def __init__(self, vocabulary, word_vectors, task="POS", stage="train", word_to_ix=None, label_to_ix=None,
                 ix_to_label=None, char_to_ix=None):
        self.stage = stage
        if task == "POS":
            if self.stage == "train":
                file = "pos/train"
            elif self.stage == "valid":
                file = "pos/dev"
            else:
                file = "pos/test"
        elif task == "NER":
            if self.stage == "train":
                file = "ner/train"
            elif self.stage == "valid":
                file = "ner/dev"
            else:
                file = "ner/dev"

        # get data from files
        if self.stage == "train" or self.stage == "valid":
            self.five_grams, self.pos_tags = transform_data_to_ngrams(file, False)
        elif self.stage == "test":
            self.five_grams, self.empty_lines, self.lines = transform_data_to_ngrams(file, True)

        # make dictionaries
        if self.stage == "train":
            # make word to index dictionary
            self.word_to_ix = {word: i for i, word in enumerate(vocabulary, start=1)}

            # make label to dictionary loss
            self.label_to_ix = {label: i for i, label in enumerate(self.pos_tags)}
            self.ix_to_label = {i: label for i, label in enumerate(self.pos_tags)}

            # get char data and make vocabulary of characters
            char_length, list_of_characters = get_char_data()
            self.char_to_ix = {char: i for i, char in enumerate(list_of_characters, start=1)}

        else:
            self.word_to_ix = word_to_ix
            self.label_to_ix = label_to_ix
            self.ix_to_label = ix_to_label
            self.char_to_ix = char_to_ix

        if task == "NER":
            self.o_label = self.label_to_ix["O"]
        else:
            self.o_label = None

        self.len_of_data = len(self.five_grams)

    def __getitem__(self, item):
        if self.stage == "train" or self.stage == "valid":
            x1 = tuple(self.word_to_ix.get(w, 0) for w in self.five_grams[item][0])
            x2 = tuple(char_to_padded(w, self.char_to_ix) for w in self.five_grams[item][0])
            y = torch.tensor(self.label_to_ix.get(self.five_grams[item][1]), dtype=torch.long)
            return (x1, x2), y
        elif self.stage == "test":
            x1 = tuple(self.word_to_ix.get(w, 0) for w in self.five_grams[item])
            x2 = tuple(char_to_padded(w, self.char_to_ix) for w in self.five_grams[item])
            return x1, x2

    def __len__(self):
        return self.len_of_data

    def word_to_index(self, word):
        return self.word_to_ix.get(word, 0)

    def character_to_index(self, char):
        return self.char_to_ix.get(char, 0)


def char_to_padded(char_seq, char_to_index):
    assert len(char_seq) <= WORD_FIXED_LENGTH, f"got {char_seq=} too big"

    char_seq_idx = [char_to_index.get(c, 0) for c in char_seq]
    char_idx = np.zeros(WORD_FIXED_LENGTH, dtype=np.int64)
    start_id = (WORD_FIXED_LENGTH - len(char_seq)) // 2
    char_idx[start_id:(start_id + len(char_seq))] = char_seq_idx
    return char_idx


def pre_process():
    minus_one_word = "h%h"
    minus_two_word = "h%h%"
    plus_one_word = "%h"
    plus_two_word = "%h%h"
    vecs = torch.from_numpy(np.loadtxt("wordVectors.txt"))
    T1 = torch.rand(1, vecs.shape[1])
    T2 = torch.rand(1, vecs.shape[1])
    T3 = torch.rand(1, vecs.shape[1])
    T4 = torch.rand(1, vecs.shape[1])
    T5 = torch.rand(1, vecs.shape[1])
    vecs = torch.cat((T5, T1, T2, vecs, T3, T4))

    # vocabulary
    with open("vocab.txt", "r") as f:
        lines = f.readlines()
    lines = [i.split("\n")[0] for i in lines]
    vocab = [minus_two_word, minus_one_word] + list(set(lines)) + [plus_one_word, plus_two_word]
    return vocab, vecs


def get_char_data():
    characters = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'/\|_@#$%^&*~`+-=<>()[]{}" + '"'

    def Convert(string):
        list1 = []
        list1[:0] = string
        return list1

    list_of_chars = Convert(characters)
    return len(characters), list_of_chars


def transform_data_to_ngrams(file_name, test=False):
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
    if not test:
        lines_split = [(i.split()[0], i.split()[1].split("\n")[0]) for i in lines if len(i) > 1]
    else:
        lines_split = [(i.split()[0]) for i in lines if len(i) > 1]

    if not test:
        pos_vocab = set([j[1] for j in lines_split])

    # find empty spaces if test
    if test:
        empty_spaces = [index for index in range(len(lines)) if len(lines[index]) <= 1]

    # create sentences- by removing empty spaces
    sentences = [[]]
    labels = []
    j = 0
    for i in lines:
        if len(i) > 1:
            sentences[j].append(i.split()[0][:WORD_FIXED_LENGTH].lower())
            if not test:
                labels.append(i.split()[1].split("\n")[0])
        else:
            j += 1
            sentences.append([])
    # remove the end, empty sentences:
    sentences = sentences[:-1]

    # extract word without tag, add to sentence header and end
    sentences_with_header_and_end = []
    for sentence in sentences:
        sentences_with_header_and_end.append(
            [minus_two_word, minus_one_word] + sentence + [plus_one_word, plus_two_word])

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
    if not test:
        zipped_features_and_labels = list(zip(run_on_sentence, labels))
        return zipped_features_and_labels, pos_vocab

    if test:
        return run_on_sentence, empty_spaces, lines_split


def train(_model, _optimizer, _loss_function, dataloader, input_len, task, o_tag=None):
    total_loss = 0
    correct = 0
    total = 0
    for idx, (features, labels) in enumerate(dataloader):
        _model.zero_grad()
        log_probs = _model(features)
        loss = _loss_function(log_probs, labels)
        loss.backward()
        _optimizer.step()
        total_loss += _loss_function(log_probs, labels).item()
        _, predicted = torch.max(log_probs.data, 1)
        if task == "NER":
            not_o = (labels != o_tag)
            total += not_o.sum().item()
            correct += ((predicted == labels) * not_o).sum().item()
        else:
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return total_loss / input_len, (100 * correct) / total


def validate(_model, _optimizer, _loss_function, dataloader, input_len, task, o_tag=None):
    total_loss = 0
    correct = 0
    total = 0
    _model.eval()
    for idx, (features, labels) in enumerate(dataloader):
        _model.zero_grad()
        log_probs = _model(features)
        total_loss += _loss_function(log_probs, labels).item()
        _, predicted = torch.max(log_probs.data, 1)
        if task == "NER":
            not_o = (labels != o_tag)
            total += not_o.sum().item()
            correct += ((predicted == labels) * not_o).sum().item()
        else:
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return total_loss / input_len, (100 * correct) / total


def predict(_model, _loss_function, dataloader):
    results = []
    _model.eval()  # switch layers which act different in train and test, like dropout etc
    for idx, (features) in enumerate(dataloader):
        _model.zero_grad()
        log_probs = _model(features)
        _, predicted = torch.max(log_probs.data, 1)
        pred = log_probs.max(1, keepdim=True)[1]
        results.extend(pred.tolist())
    return results


if __name__ == "__main__":
    wanted_task = "POS"
    vocabulary, word_vectors = pre_process()
    dataset = NGramsDataset(vocabulary, word_vectors, wanted_task, "train")

    # obtain all dictionaries
    w_t_i = dataset.word_to_ix
    l_t_i = dataset.label_to_ix
    i_t_l = dataset.ix_to_label
    c_t_i = dataset.char_to_ix

    valid_set = NGramsDataset(vocabulary, word_vectors, wanted_task, "valid", w_t_i, l_t_i, i_t_l, c_t_i)
    test_set = NGramsDataset(vocabulary, word_vectors, wanted_task, "test", w_t_i, l_t_i, i_t_l, c_t_i)

    # load data
    train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(dataset=valid_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=True)
    model = NGramLanguageModeler(embedding_matrix=word_vectors, tags_size=len(dataset.label_to_ix.keys()))
    lr = 0.001
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.CrossEntropyLoss()
    accuracies = []
    losses = []
    valid_accuracies = []
    valid_losses = []
    for epoch in range(5):
        print("epoch is: {}".format(epoch))
        l_train, a_train = train(model, optimizer, loss_function, train_loader, len(train_loader), wanted_task,
                                 dataset.o_label)
        accuracies.append(a_train)
        losses.append(l_train)
        l_valid, a_valid = train(model, optimizer, loss_function, train_loader, len(train_loader), wanted_task,
                                 dataset.o_label)
        valid_accuracies.append(a_valid)
        valid_losses.append(l_valid)
        print("l is {} , a is {}: ".format(l_train, a_train))

    predictions = predict(_model=model, _loss_function=loss_function, dataloader=test_loader)

    if wanted_task == "POS":
        results_file_name = "pos_results.txt"
    else:
        results_file_name = "ner_results.txt"
    with open(results_file_name, "w") as f:
        cnt = 0
        j = 0
        for i in range(len(predictions) + len(test_set.empty_lines)):
            if cnt in test_set.empty_lines:  # to save format
                f.write("\n")
            else:
                f.write(str(test_set.lines[j]) + "\t" + str(dataset.ix_to_label[predictions[j][0]]) + "\n")
                j += 1
            cnt += 1

    print("accuracies are:", accuracies, "validation accuracies: ", valid_accuracies)
    print("loss are: ", losses, "valid losses", valid_losses)

    plt.plot(losses, label="train")
    plt.plot(valid_losses, label="validation")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title("{}\nLoss according to epochs, learning rate = {}".format(wanted_task, lr))
    plt.legend()
    plt.show()

    plt.plot(accuracies, label="train")
    plt.plot(valid_accuracies, label="validation")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.title("{}\nAccuracy according to epochs, learning rate = {}".format(wanted_task, lr))
    plt.legend()
    plt.show()
