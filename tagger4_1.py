import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim

CONTEXT_SIZE = 5
EMBEDDING_DIM = 50
CHAR_EMBEDDING_DIM = 30
CHAR_VOCAB_SIZE = 68
BATCH_SIZE = 512
WORD_FIXED_LENGTH = 20


class NGramLanguageModeler(nn.Module):

    def __init__(self, embedding_matrix, tags_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding.from_pretrained(embedding_matrix.float())
        self.char_embedding = nn.Embedding(CHAR_VOCAB_SIZE + 1, CHAR_EMBEDDING_DIM)
        # self.layer1 = torch.nn.Sequential(
        #     torch.nn.Conv2d(1, 30, kernel_size=(3,20), stride=1, padding=(2,0)),
        #     torch.nn.MaxPool2d(kernel_size=(3,3), stride=1))
        # self.layer1 = torch.nn.Sequential(
        #     torch.nn.Conv2d(1, 30, kernel_size=(3,3), stride=1),
        #     torch.nn.MaxPool2d(kernel_size=(3,3), stride=1))
        self.layer1 = torch.nn.Conv1d(1, 30, kernel_size=3, padding=1)
        self.max_pool = torch.nn.MaxPool1d(kernel_size=600, stride=1)
        self.linear1 = nn.Linear(EMBEDDING_DIM * CONTEXT_SIZE, 128)
        self.linear2 = nn.Linear(128, tags_size)

    def forward(self, inputs):
        # word embedding
        inputs_words = torch.stack(inputs[0]).t()
        embeds = self.embeddings(inputs_words)
        embeds = embeds.reshape(embeds.shape[0], CONTEXT_SIZE * EMBEDDING_DIM)

        # char embedding
        word_one_char = self.char_embedding(inputs[1][0])
        word_one_char = torch.flatten(word_one_char, start_dim=1).unsqueeze(1)
        word_two_char = self.embeddings(inputs[1][1])
        word_three_char = self.embeddings(inputs[1][2])
        word_four_char = self.embeddings(inputs[1][3])
        word_five_char = self.embeddings(inputs[1][4])
        print("word one char dims is: ", word_one_char.shape)
        word = self.layer1(word_one_char)
        print("word one conv is: ", word.shape)
        word = torch.flatten(self.max_pool(word), start_dim=1)
        print("word one conv is: ", word.shape)
        return
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


class NGramsDataset(Dataset):
    def __init__(self, vocabulary, word_vectors, task="POS", train=True, valid=False, test=False):
        if task == "POS":
            train_file = "pos/train"
            valid_file = "pos/dev"
            test_file = "pos/test"
        elif task == "NER":
            train_file = "ner/train"
            valid_file = "ner/dev"
            test_file = "ner/test"

        # get data from files
        self.five_grams, self.pos_tags = transform_data_to_ngrams(train_file, False)
        self.five_grams_valid, self.pos_tags_valid = transform_data_to_ngrams(valid_file, False)
        self.five_grams_test, self.empty_lines = transform_data_to_ngrams(test_file, True)

        # make word to index dictionary
        self.word_to_ix = {word: i for i, word in enumerate(vocabulary, start=1)}

        # make label to dictionary loss
        self.label_to_ix = {label: i for i, label in enumerate(self.pos_tags)}
        self.ix_to_label = {i: label for i, label in enumerate(self.pos_tags)}

        # get char data and make vocabulary of characters
        char_length, list_of_characters = get_char_data()
        self.char_to_ix = {char: i for i, char in enumerate(list_of_characters, start=1)}

        self.len_of_data = len(self.five_grams)

        if task == "NER":
            self.o_label = self.label_to_ix["O"]
        else:
            self.o_label = None

    def __getitem__(self, item):
        x1 = tuple(self.word_to_ix.get(w, 0) for w in self.five_grams[item][0])
        x2 = tuple(char_to_padded(w, self.char_to_ix) for w in self.five_grams[item][0])
        y = torch.tensor(self.label_to_ix.get(self.five_grams[item][1]), dtype=torch.long)
        return (x1, x2), y

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
    char_idx[start_id:(start_id+len(char_seq))] = char_seq_idx
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
        return run_on_sentence, empty_spaces


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
    print("total is: ", total, "correct is: ", correct)
    return total_loss / input_len, (100 * correct) / total


if __name__ == "__main__":
    wanted_task = "POS"
    vocabulary, word_vectors = pre_process()
    dataset = NGramsDataset(vocabulary, word_vectors, wanted_task)
    train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = NGramLanguageModeler(embedding_matrix=word_vectors, tags_size=len(dataset.label_to_ix.keys()))
    lr = 0.5
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.CrossEntropyLoss()
    for epoch in range(10):
        print("epoch is: {}".format(epoch))
        l_train, a_train = train(model, optimizer, loss_function, train_loader, len(train_loader), wanted_task, dataset.o_label)
        print("l is {} , a is {}: ".format(l_train, a_train))


