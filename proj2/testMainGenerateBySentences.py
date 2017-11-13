from spacy.en import English
import string

import collections
import math
import os
import random
from tempfile import gettempdir
import zipfile
import numpy as np
from six.moves import urllib
import tensorflow as tf
from six.moves import range
from six.moves.urllib.request import urlretrieve
from sklearn.manifold import TSNE
import gensim
import pickle

import re
import math
from scipy.spatial import KDTree

import time
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS


### fixed parameters
embedding_dim = 200
number_of_iterations = 100001
# Loss function, optimizer

# v 10000 is not good
### Tunable parameter
vocabulary_size = 10000
batch_size = 128      # Size of mini-batch for skip-gram model..
skip_window = 3       # How many words to consider left and right of the target word.
num_samples = 4         # How many times to reuse an input to generate a label.
num_sampled_ns = 64        # How many negative samples going to be chose, as suggested 10~30 for small dataset
learning_rate = 0.002

logs_path = './log/'

SELF_DEFINED_STOP_WORD = ['which','its','that','this','what','how', 'their', 'his', 'her', 'our']

global data_filled_with_num, counter, dictionary, reverse_dictionary
global parser
global total_word_count

def tokenizeText(corpus):
    global parser
    parser = English()
    # get the tokens using spaCy
    tokens = parser(corpus)

    # PUNC_SYMBOLS = " ".join(string.punctuation).split(" ") + ["-----", "---", "...", "“", "”", "'ve", " "]

    # lemmatize
    # 'VERB', 'NOUN', 'ADV', consider only use ADJ
    # important_pos = ['ADJ', 'NOUN']
    useless_pos = ['PRON', 'CONJ', 'PREP', 'DET', 'NUM', 'SYM'] # 'ADP'
    skip_pos = ['PUNCT', 'SPACE', 'PART']
    lemmas = []
    previous_word = ''

    for sentence in tokens.sents:
        sentence_list = []
        previous_word = ''
        for tok in sentence:
            if tok.ent_type_ != "" and "-"+tok.ent_type_+"-" != previous_word:
                sentence_list.append("-"+tok.ent_type_+"-")
            elif tok.pos_ in skip_pos:
                continue
            elif tok.pos_ == 'ADJ':
                sentence_list.append(tok.orth_.lower().strip()+'ADJ')
            elif tok.pos_ in useless_pos and tok.pos_ != previous_word:
                sentence_list.append(tok.pos_)
            else:
                sentence_list.append(tok.lemma_.lower().strip())
            # clean up
            if re.match('^[a-zA-Z\-]+$', sentence_list[-1]) != None:
                previous_word = sentence_list[-1]
            else:
                sentence_list.pop()

        lemmas.append(sentence_list)

    tokens = lemmas

    for tok in tokens[:100]:
        try:
            print(tok)
        except UnicodeEncodeError:
            print('UnicodeEncodeError')

    return tokens

def build_dataset(sentences, n_words):
    """Process raw inputs into a dataset.
       sentences: a list of sentences, i.e., the input data
       n_words: Vocab_size to limit the size of the vocabulary. Other words will be mapped to 'UNK'
    """
    words_flatten = [item for sublist in sentences for item in sublist]

    global total_word_count
    total_word_count = len(words_flatten)

    counter = [['UNK', -1]]
    counter.extend(collections.Counter(words_flatten).most_common(n_words - 1))
    assert len(counter) == n_words

    dictionary = dict()
    for word, _ in counter:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0

    for sentence in sentences:
        st_of_numbers = []
        for word in sentence:
            index = dictionary.get(word, 0)
            if index == 0:  # i.e., one of the 'UNK' words
                unk_count += 1
            st_of_numbers.append(index)
        data.append(st_of_numbers)

    counter[0][1] = unk_count

    print(counter[:n_words])

    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    print('data of sentences filled with numbers: ')
    for i in range(10):
        print(data[i])

    return data, counter, dictionary, reversed_dictionary


def is_keep_as_context(word):
    global total_word_count, counter
    frequency = float(counter[dictionary[word]][1])/float(total_word_count)
    prob = (math.sqrt(frequency/0.001) + 1) * (0.001/frequency)
    if random.uniform(0,1) < prob:
        return True
    else:
        return False


# used in generate_batch
sentence_index = 0
previous_sentence_index = 0
def generate_batch(batch_size, num_samples, skip_window):
    global sentence_index, previous_sentence_index
    global data_filled_with_num, counter, dictionary, reverse_dictionary, parser

    # assert batch_size % num_samples == 0
    assert num_samples <= 2 * skip_window

    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # span is the width of the sliding window

    ##### loop through data_filled_with_num get sentences, loop through sentences get span windows, and for each
    ##### span window, pick num_samples out of with consideration of subsampling
    entries_in_batch = 0
    length_of_data_filled_with_num = len(data_filled_with_num)
    while entries_in_batch < batch_size:
        if sentence_index == length_of_data_filled_with_num:
            # print('reset sentence index to zero')
            sentence_index = 0

        st_of_numbers = data_filled_with_num[sentence_index]
        # print('sentence_index: ', sentence_index, 'st_of_numbers: ', st_of_numbers)
        if len(st_of_numbers) < span:
            middle_index = len(st_of_numbers) // 2
            for i in range(len(st_of_numbers)):
                if i == middle_index:
                    continue
                if is_keep_as_context(reverse_dictionary[st_of_numbers[i]]):
                    batch[entries_in_batch] = st_of_numbers[middle_index]
                    labels[entries_in_batch, 0] = st_of_numbers[i] 
                    entries_in_batch += 1
                    # only need check it when entries_in_batch get checked
                    if entries_in_batch == batch_size:
                        previous_sentence_index = 0 # for safety, add it
                        sentence_index += 1
                        # print('Upon return 1, entries in batch: ', entries_in_batch)
                        return batch, labels
        else:
            # the context_word_index and center_word_index are the index in st_of_numbers
            if previous_sentence_index > 0:
                center_word_index = previous_sentence_index
            else:
                center_word_index = skip_window
            while center_word_index < len(st_of_numbers) - skip_window:
                ## pick center word
                if is_keep_as_context(reverse_dictionary[st_of_numbers[center_word_index]]) == False:
                    center_word_index += 1
                    continue
                ## pick context word
                context_words_indexes = [w for w in range(center_word_index - skip_window, center_word_index + skip_window + 1) if w != center_word_index]
                random.shuffle(context_words_indexes)
                words_to_use = collections.deque(context_words_indexes)
                context_word_count = 0
                while context_word_count < num_samples:
                    context_word_index = words_to_use.pop()
                    if len(words_to_use) == 0:
                        words_to_use.extend(context_words_indexes)

                    if is_keep_as_context(reverse_dictionary[st_of_numbers[context_word_index]]):
                        batch[entries_in_batch] = st_of_numbers[center_word_index]
                        labels[entries_in_batch, 0] = st_of_numbers[context_word_index] 
                        entries_in_batch += 1
                        context_word_count += 1
                        # if this sentence only get used less half of its length, then do not increase sentence_index
                        if entries_in_batch == batch_size and center_word_index < len(st_of_numbers) - skip_window:
                            # print('Upon return 2, entries in batch: ', entries_in_batch)
                            previous_sentence_index = center_word_index
                            return batch, labels
                        elif entries_in_batch == batch_size:
                            # print('Upon return 3, entries in batch: ', entries_in_batch)
                            previous_sentence_index = 0
                            sentence_index += 1
                            return batch, labels    
                ## move the extraction window by 1 after every extraction
                center_word_index += 1

        # After checking every sentence, increase the sentence_index by 1
        sentence_index += 1


def write_embedding_to_file(final_embeddings, embeddings_file_name):
    global dictionary, reverse_dictionary

    with open(embeddings_file_name, 'w') as f:
        f.write('{} {}\n'.format(vocabulary_size,embedding_dim))
        for index in reverse_dictionary.keys():
            f.write('{} {}\n'.format(reverse_dictionary[index], ' '.join(str(e) for e in final_embeddings[index])))
        f.close()


def process_data(input_data_dir):
    # if os.path.exists('processed_data.pkl'):
    #     with open("processed_data.pkl", "rb") as fp:   # Unpickling
    #         global parser
    #         parser = English()
    #         return pickle.load(fp)

    data = ''
    with zipfile.ZipFile(input_data_dir) as zipf:
        # data = tf.compat.as_str(f.read(f.namelist()[0])).split()
        for f in zipf.namelist():
            data += tf.compat.as_str(zipf.read(f)) + "\n"

    # print(data[:10], len(data))

    data = tokenizeText(data)

    # with open("processed_data.pkl", "wb") as fp:   #Pickling
    #     pickle.dump(data, fp)

    print('len(data): ',len(data))

    return data


def adjective_embeddings(data_file, embeddings_file_name, num_steps, embedding_dim):
    global data_filled_with_num, counter, dictionary, reverse_dictionary
    data_filled_with_num, counter, dictionary, reverse_dictionary = build_dataset(data_file, vocabulary_size)

    test_words =['basic', 'big', 'chief', 'clear', 'confident', 'conservative', 'corporate', 'difficult', 'fair', 'famous', 'few', 'final', 'former', 'free', 'full', 'high', 'huge', 'interested', 'large', 'low', 'main', 'malicious', 'many', 'mobile', 'modern', 'more', 'much', 'old', 'private', 'ready', 'same', 'significant', 'single', 'special', 'specific', 'successful', 'top', 'vital', 'wide', 'worth'] 
    # Specification of test Sample:
    sample_size = len(test_words)       # Random sample of words to evaluate similarity.
    sample_window = 100    # Only pick samples in the head of the distribution.
    sample_examples = []

    for tword in test_words:
        tword += 'ADJ'
        sample_examples.append(dictionary[tword])


    ## Constructing the graph...
    graph = tf.Graph()

    with graph.as_default():

        # with tf.device('/cpu:0'):
        # Placeholders to read input data.
        with tf.name_scope('Inputs'):
            train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
            train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

        # Look up embeddings for inputs.
        with tf.name_scope('Embeddings'):
            sample_dataset = tf.constant(sample_examples, dtype=tf.int32)
            embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_dim], -1.0, 1.0))
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)

            # Construct the variables for the NCE loss
            nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_dim],
                                                      stddev=1.0 / math.sqrt(embedding_dim)))
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

        # Compute the average NCE loss for the batch.
        # tf.nce_loss automatically draws a new sample of the negative labels each
        # time we evaluate the loss.
        print('in usage, learning_rate: {}, nsampled: {}, vocabulary_size: {}'.format(learning_rate, num_sampled_ns, vocabulary_size))
        with tf.name_scope('Loss'):
            loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=nce_weights, biases=nce_biases,
                                             labels=train_labels, inputs=embed,
                                             num_sampled=num_sampled_ns, num_classes=vocabulary_size))

        # Construct the Gradient Descent optimizer using a learning rate of 0.01.
        with tf.name_scope('Gradient_Descent'):
            optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

        # Normalize the embeddings to avoid overfitting.
        with tf.name_scope('Normalization'):
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
            normalized_embeddings = embeddings / norm

        sample_embeddings = tf.nn.embedding_lookup(normalized_embeddings, sample_dataset)
        similarity = tf.matmul(sample_embeddings, normalized_embeddings, transpose_b=True)

        # Add variable initializer.
        init = tf.global_variables_initializer()


        # Create a summary to monitor cost tensor
        tf.summary.scalar("cost", loss)
        # Merge all summary variables.
        merged_summary_op = tf.summary.merge_all()

    with tf.Session(graph=graph) as session:
        # We must initialize all variables before we use them.
        session.run(init)
        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

        print('Initializing the model')

        average_loss = 0
        for step in range(num_steps):
            batch_inputs, batch_labels = generate_batch(batch_size, num_samples, skip_window)
            # print('print batch_inputs and batch_labels, len(batch_inputs): ', len(batch_inputs))
            # for i in range(len(batch_inputs)):
            #     print('batch_inputs: ', batch_inputs[i])
            #     print('batch_labels: ', batch_labels[i][0])
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

            # We perform one update step by evaluating the optimizer op using session.run()
            _, loss_val, summary = session.run([optimizer, loss, merged_summary_op], feed_dict=feed_dict)

            summary_writer.add_summary(summary, step )
            average_loss += loss_val

            if step % 5000 == 0:
                if step > 0:
                    average_loss /= 5000

                    # The average loss is an estimate of the loss over the last 5000 batches.
                    print('Average loss at step ', step, ': ', average_loss)
                    average_loss = 0

            # Evaluate similarity after every 10000 iterations.
            if step % 10000 == 0:
                sim = similarity.eval() #
                for i in range(sample_size):
                    sample_word = reverse_dictionary[sample_examples[i]]
                    top_k = 10  # Look for top-10 neighbours for words in sample set.
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log_str = 'Nearest to %s:' % sample_word
                    for k in range(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        log_str = '%s %s,' % (log_str, close_word)
                    try:
                        print(log_str)
                    except UnicodeEncodeError:
                        print('UnicodeEncodeError:')
                print()

        final_embeddings = normalized_embeddings.eval()
        write_embedding_to_file(final_embeddings, embeddings_file_name)


def Compute_topk(model_file, input_adjective, top_k):
    global data_filled_with_num, counter, dictionary, reverse_dictionary, parser

    # 'ADP'
    sensitive_replace_word = ['UNK', 'PRON', 'CONJ', 'PREP', 'DET', 'NUM', 'SYM'] 

    model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=False)

    output = []
    # tword_token = parser(input_adjective)[0]
    temp_topk_multiplier = 5
    # distances, ndx = tree.query(dictionary[input_adjective], k = top_k * temp_topk_multiplier)
    temp_result = model.most_similar(positive=[input_adjective + 'ADJ'], topn= top_k * temp_topk_multiplier)
    words = [r[0] for r in temp_result]

    while len(output) < top_k:
        for word in words:
            if word in sensitive_replace_word:
                continue
            # if parser(word)[0].pos_ == tword_token.pos_:
            # and word[:-3] not in list(ENGLISH_STOP_WORDS)
            if word[-3:] == 'ADJ' and word[:-3] not in list(SELF_DEFINED_STOP_WORD):
                output.append(word[:-3])
            elif parser(word)[0].pos_ == 'ADJ':
                output.append(word)

        if temp_topk_multiplier > 10:
            break

        if len(output) < top_k:
            temp_topk_multiplier += 1
            temp_result = model.most_similar(positive=[input_adjective + 'ADJ'], topn= top_k * temp_topk_multiplier)
            words = [r[0] for r in temp_result]
            words = words[top_k * (temp_topk_multiplier -1) :]

    return output[:top_k]


if __name__ == "__main__":
    data_of_sentences = process_data('./BBC_Data.zip')
    model_file = 'adjective_embeddings.txt'
    adjective_embeddings(data_of_sentences, model_file, number_of_iterations, embedding_dim)
    top_k = 100

    from os import listdir
    from os.path import isfile, join

    match_counter = 0
    word_files = [f for f in listdir('dev_set/') if isfile(join('dev_set/', f))]

    for wf in word_files:
        computed_similar_words = Compute_topk(model_file, wf, top_k)
        ground_truth_words = []
        file = open(join('dev_set/', wf),'r')
        i = 0
        for w in file:
            if i >= 100:
                break
            ground_truth_words.append(w.strip())  
            i += 1

        print('Matched nearest to ', wf, ': ', end= " ")

        for csw in computed_similar_words:
            if csw in ground_truth_words:
                print(csw, " ", end=" ")
                match_counter += 1

        print("\n")

    print('average hits: ', match_counter / len(word_files))