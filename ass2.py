

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
    vocab_with_special_characters = [minus_two_word, minus_one_word] + list(vocab_with_special_characters) + \
                                    [plus_one_word, plus_two_word]
    return vocab_with_special_characters, sentences_label, sentences_no_tag


def get_embedding_matrix_part_one(embedded_matrix, vocab_index, sentence):
    """
    Returns embedded sentence in 5-gram fashion
    """
    minus_one_word = "h%h"
    minus_two_word = "h%h%"
    plus_one_word = "%h"
    plus_two_word = "%h%h"
    sentence = [minus_two_word, minus_one_word] + sentence + [plus_one_word, plus_two_word]
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

