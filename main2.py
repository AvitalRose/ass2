# Avital Rose 318413408
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import torch
from torch import nn, optim
# from torch.autograd.grad_mode import F
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


class Network(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(input_size, 40)
        # Output layer, 36 units - one for each POS
        self.output = nn.Linear(40, 45)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden(x)
        # Define tanh activation
        x = self.tanh(x)
        x = self.output(x)
        # and softmax output
        x = self.softmax(x)
        return x


def train(_train_loader, model, optimizer):
    model.train()
    train_loss = 0
    train_correct = 0
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (_data, _labels) in enumerate(_train_loader):
        optimizer.zero_grad()
        output = model(_data)
        loss = criterion(output, _labels)
        train_loss += criterion(output, _labels).item()
        # train_loss += F.nll_loss(output, _labels, size_average=False).item()
        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        train_correct += pred.eq(_labels.view_as(pred)).cpu().sum().item()
        loss.backward()
        optimizer.step()
    train_loss /= len(_train_loader.dataset)
    train_correct = 100. * train_correct / len(_train_loader.dataset)
    return train_loss, train_correct

def part_one(train_x, train_y):
    """

    :return:
    """
    # step 1: prepare data

    train_x = torch.from_numpy(np.array(train_x)).float()
    train_y = torch.from_numpy(np.array(train_y)).long()
    print("shape of data train_x: ", train_x.shape, "shape of data train_y : ", train_y.shape)
    dataset = TensorDataset(train_x, train_y)
    # train_set, validate_set = torch.utils.data.random_split(dataset, [44000, 11000])
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)

    # step 2: prepare model
    model = Network(input_size=250)
    lr = 0.2
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # step 3: train
    loss_train_list = []
    accuracy_train_list = []
    for e in range(1, 10 + 1):
        print("epoch number {} ".format(e))
        l_t, a_t = train(train_loader, model, optimizer)
        print("accuracy is: ", a_t, "loss is: ", l_t)
        loss_train_list.append(l_t)
        accuracy_train_list.append(a_t)


    # step 4: plot findings
    plt.plot(loss_train_list, label="train")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title("loss according to epochs, learning rate = {}".format(lr))
    plt.legend()
    plt.show()

    plt.plot(accuracy_train_list, label="train")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.title("accuracy according to epochs, learning rate = {}".format(lr))
    plt.legend()
    plt.show()

def read_file_to_list(train_file):
    """
    This function converts a file to a parsed list of (word, POS) as well as adding special symbols.
    :return:
    """
    # need to remove some lines, by what criteria though?
    with open(train_file, 'r') as f:
        lines = f.readlines()
    lines_split = [(i.split(' ')[0], i.split(' ')[1][:-1]) for i in lines if len(i) > 1]  # length of empty lines is 0
    vocab_with_special_characters = set([j[0] for j in lines_split])  # ends up being 7896 words or 43127
    pos_vocab = set([j[1] for j in lines_split])
    # development
    # cnt = 0
    # for i in lines:
    #     if len(i) <= 1:
    #         cnt += 1
    # print("cnt is: ", cnt)
    #  split into sentences
    sentences = [[]]
    j = 0
    for i in lines:
        if len(i) > 1:
            sentences[j].append(i)
        else:
            j += 1
            sentences.append([])
    # remove the end, empty sentences:
    sentences = sentences[:-1]
    # # split into tokens
    sentences_no_tag = []
    sentence_split = []
    sentences_label = []
    for sentence in sentences:
        sentences_no_tag.append([i.split(' ')[0] for i in sentence if len(i) > 1])
        sentence_split.append([(i.split(' ')[0], i.split(' ')[1][:-1]) for i in sentence if len(i) > 1])
        sentences_label.append([i.split(' ')[1][:-1] for i in sentence if len(i) > 1])

    #  insert in the beginning of every sentence a special two words and in the end
    minus_one_word = "h%h"
    minus_two_word = "h%h%"
    plus_one_word = "%h"
    plus_two_word = "%h%h"
    sentence_split_with_header_and_finish = []
    for sentence in sentences_no_tag:
        sentence_split_with_header_and_finish.append([minus_two_word, minus_one_word] + sentence +
                                                     [plus_one_word, plus_two_word])
    vocab_with_special_characters = [minus_two_word, minus_one_word] + list(vocab_with_special_characters) + \
                                    [plus_one_word, plus_two_word]

    return vocab_with_special_characters, pos_vocab, sentence_split_with_header_and_finish, sentences_label, sentences_no_tag


def get_embedding_matrix_part_one(embedded_matrix, vocab_index, sentence):
    """
    Returns embedded sentence in 5-gram fashion
    """
    sentence_vectors = []
    for word in sentence:
        sentence_vectors.append(embedded_matrix[vocab_index[word]])
    # used the following line for development
    # ngrams_words = [[sentence[i - 2], sentence[i - 1], sentence[i], sentence[i + 1],
    #                  sentence[i + 2]] for i, word in enumerate(sentence[2:-2], start=2)]
    ngrams = [[sentence_vectors[i - 2], sentence_vectors[i - 1], sentence_vectors[i], sentence_vectors[i + 1],
               sentence_vectors[i + 2]] for i, word in enumerate(sentence_vectors[2:-2], start=2)]
    ngrams_concentrate = [np.concatenate(five_gram) for five_gram in ngrams]
    return ngrams_concentrate


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
    # list of tags? convert tags to numbers

    vocab, pos_tags, sentences_extended, labels, sentences = read_file_to_list('pos/train')

    # make the embedding vector:
    embedding_matrix = np.random.random((len(set(vocab)) + 1, 50))

    # create embedding dict
    embedding_dict = {word: i for i, word in enumerate(vocab)}

    # glue together all the sentences
    run_on_sentence = []
    for sentence in sentences_extended:
        x = get_embedding_matrix_part_one(embedded_matrix=embedding_matrix, vocab_index=embedding_dict,
                                          sentence=sentence)
        run_on_sentence.extend(x)
    run_on_label = []
    for label in labels:
        run_on_label.extend(label)

    # convert all labels to numbers
    # create labels dict
    label_dict = {label: i for i, label in enumerate(pos_tags)}
    print("label dict is: ", label_dict)

    labels_numbered = []
    for label in run_on_label:
        labels_numbered.append(label_dict[label])

    # train
    part_one(run_on_sentence, labels_numbered)









"""
Questions to self:
    1. how to handle empty lines? 
    2. The words are one hot encoded or bi-grammed?
    3. Straight after the tanh there is softmax? or first there is another layer?
    4. how to handle ., etc?.	punctuation marks	. , ; ! - just dont tag them, use them only as words before or after
    5. How to handle sentence after sentence- sentence tokenization? will split by " " (space)
    6. should I switch to different package instead of pytorch?   does dynet use word embediing? watch video
    7. along what axis to concatenate 
    8. move sentences extension to new part- general refractor , do when have time. write code logic down
    9. number of empty sentences 37832- remove this part in read_file_to_list
    10. how to handle words which are in dev and not in train?

"""
