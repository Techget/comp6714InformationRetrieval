from spacy.en import English
import spacy
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
import re
import math

SELF_DEFINED_STOP_WORD = ['which','its','that','this','what','how', 'their', 'his', 'her', 'our', 'it']

### fixed parameters
embedding_dim = 200
number_of_iterations = 100001
# Loss function, optimizer

# record 12768 128 3 4 128 0.001, result 3.075
# record 5000 126 2 3 32 0.002, result 4.875
# record 5000, 128, 3, 4, 32, 0.002 without ADP performance 5.3

### Tunable parameter
vocabulary_size = 7000
batch_size = 128      # Size of mini-batch for skip-gram model..
# 1
skip_window = 3       # How many words to consider left and right of the target word.
# 2
num_samples = 4         # How many times to reuse an input to generate a label.
num_sampled_ns = 64        # How many negative samples going to be chose, as suggested 10~30 for small dataset
learning_rate = 0.002

logs_path = './log/'

global data_filled_with_num, counter, dictionary, reverse_dictionary
global total_word_count

def tokenizeText(corpus):
    nlp = spacy.load('en')
    tokens = nlp(corpus)
    # get the tokens using spaCy
    # tokens = parser(corpus)

    # lemmatize
    # 'VERB', 'NOUN', 'ADV', consider only use ADJ
    # important_pos = ['VERB', 'NOUN']
    useless_pos = ['NUM', 'SYM', 'CONJ', 'PREP', 'PRON', 'PPRON'] # 'ADP', 'DET'
    skip_pos = ['PUNCT', 'SPACE', 'PART'] # , 'PART'
    lemmas = []
    previous_word = ''

    # use eof instead of eos, since there may be situation where one sentence is too short

    for sentence in tokens.sents:
        for tok in sentence:
            if tok.orth_ == 'EOF':
                lemmas.append('-EOF-')
            elif tok.ent_type_ != "" and "-"+tok.ent_type_+"-" != previous_word:
                # may consider do lemmas.append("-"+tok.ent_type_+"-")
                # print('tok.ent_type_: ', tok.ent_type_)
                lemmas.append("-"+tok.ent_type_+"-")
                # continue
            elif tok.pos_ in skip_pos:
                continue
            elif tok.pos_ in useless_pos and tok.pos_ != previous_word:
                lemmas.append(tok.pos_)
            elif tok.pos_ == 'ADJ':
                lemmas.append(tok.orth_.lower().strip()+'ADJ')
            elif tok.pos_ == 'NOUN':
                lemmas.append(tok.lemma_.lower().strip()+'NOUN')
            elif tok.pos_ == 'VERB':
                lemmas.append(tok.lemma_.lower().strip()+'VERB')
            else:
                lemmas.append(tok.lemma_.lower().strip())

            # clean up
            if re.match('^[a-zA-Z\-]+$', lemmas[-1]) != None:
                previous_word = lemmas[-1]
            else:
                lemmas.pop()

        lemmas.append('-EOS-')
        previous_word = ''

    lemmas.pop()
    tokens = lemmas

    return tokens

def build_dataset(data_file, n_words):
    """Process raw inputs into a dataset.
       words: a list of words, i.e., the input data
       n_words: Vocab_size to limit the size of the vocabulary. Other words will be mapped to 'UNK'
    """
    words = []
    with open(data_file, "r") as fp:
        for line in fp:
            words.append(line.strip())

    global total_word_count
    total_word_count = len(words)

    counter = [['UNK', -1]]
    counter.extend(collections.Counter(words).most_common(n_words - 1))
    # print(len(counter))
    assert len(counter) == n_words

    dictionary = dict()
    for word, _ in counter:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  # i.e., one of the 'UNK' words
            unk_count += 1
        data.append(index)
    counter[0][1] = unk_count

    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, counter, dictionary, reversed_dictionary


def is_keep_as_context(word):
    global total_word_count, counter
    frequency = float(counter[dictionary[word]][1])/float(total_word_count)
    prob = (math.sqrt(frequency/0.001) + 1) * (0.001/frequency)
    if random.uniform(0,1) < prob:
        return True
    else:
        return False

def getPOS(word):
    m = re.search('[A-Z]+', word)
    if m == None:
        return ''
    else:
        return m.group()


# used in generate_batch
data_index = 0
def generate_batch(batch_size, num_samples, skip_window):
    global data_index
    global data_filled_with_num, counter, dictionary, reverse_dictionary

    assert batch_size % num_samples == 0
    assert num_samples <= 2 * skip_window

    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # span is the width of the sliding window
    buffer = collections.deque(maxlen=span)
    if data_index + span > len(data_filled_with_num):
        data_index = 0
    buffer.extend(data_filled_with_num[data_index:data_index + span]) # initial buffer content = first sliding window

    # try:
    #     print('generate batch data_index = {}, buffer = {}'.format(data_index, [reverse_dictionary[w] for w in buffer]))
    # except UnicodeEncodeError:
    #     print('UnicodeEncodeError')

    RIGHT_BOUND = len(data_filled_with_num) - span - 1
    data_index += span
    for i in range(batch_size // num_samples):
        ## pick a proper center word
        while reverse_dictionary[buffer[skip_window]] == '-EOF-':
            if data_index >= RIGHT_BOUND:
                data_index = span
                buffer.extend(data_filled_with_num[:span])

            data_index += 1
            buffer.append(data_filled_with_num[data_index + span])

        context_words = [w for w in range(span) if w != skip_window]
        random.shuffle(context_words)
        words_to_use = collections.deque(context_words) # now we obtain a random list of context words

        ## pick context word
        j = 0

        if getPOS(reverse_dictionary[buffer[skip_window]]) == 'ADJ':
            for cw in range(skip_window + 1, 2*skip_window +1):
                if getPOS(reverse_dictionary[buffer[cw]]) in ['NOUN']:
                    batch[i * num_samples + j] = buffer[skip_window]
                    labels[i * num_samples + j, 0] = buffer[cw] # buffer[context_word] is a random context word
                    j += 1
                    words_to_use.remove(cw)
                    if j == num_samples:
                        break
            for cw in range(skip_window):
                if getPOS(reverse_dictionary[buffer[cw]]) in ['VERB']:
                    batch[i * num_samples + j] = buffer[skip_window]
                    labels[i * num_samples + j, 0] = buffer[cw] # buffer[context_word] is a random context word
                    j += 1
                    words_to_use.remove(cw)
                    if j == num_samples:
                        break

        while j < num_samples:
            context_word = words_to_use.pop()
            if len(words_to_use) == 0 and j < num_samples:
                words_to_use.extend([w for w in range(span) if w != skip_window])

            if reverse_dictionary[buffer[skip_window]] != '-EOF-' and is_keep_as_context(reverse_dictionary[buffer[context_word]]):
                batch[i * num_samples + j] = buffer[skip_window]
                labels[i * num_samples + j, 0] = buffer[context_word] # buffer[context_word] is a random context word
                j += 1

        ## flush buffer, if that contains -EOF-
        if dictionary['-EOF-'] in buffer:
            # general process, can't use `list(buffer).index(dictionary['-EOF-'])`
            data_index += skip_window
            if data_index >= RIGHT_BOUND:
                data_index = span
                buffer.extend(data_filled_with_num[:span])
        while dictionary['-EOF-'] in buffer:
            if buffer.popleft() == dictionary['-EOF-']:
                break
            # print('try to pop the EOF out')
            buffer.append(data_filled_with_num[data_index])
            data_index += 1
            if data_index >= RIGHT_BOUND:
                buffer.extend(data_filled_with_num[:span])
                data_index = span

        # slide the window to the next position
        if data_index >= RIGHT_BOUND:
            buffer.extend(data_filled_with_num[:span])
            data_index = span
        else:
            buffer.append(data_filled_with_num[data_index]) # note that due to the size limit, the left most word is automatically removed from the buffer.
            data_index += 1

    # end-of-for
    data_index = (data_index + len(data_filled_with_num) - span) % len(data_filled_with_num) # move data_index back by `span`
    return batch, labels


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
            if f[-3:] == 'txt':
                data += tf.compat.as_str(zipf.read(f)) + ' [EOF] '

    data = tokenizeText(data)

    # with open("processed_data.pkl", "wb") as fp:   #Pickling
    #     pickle.dump(data, fp)

    print('len(data): ',len(data))

    # write to temporary file
    processed_data_file_name = 'preprocessed_data_file.txt'
    with open(processed_data_file_name, "w") as fp:
        for d in data:
            fp.write(d)
            fp.write("\n")

    fp.close()
    return processed_data_file_name


def adjective_embeddings(data_file, embeddings_file_name, num_steps, embedding_dim):
    global data_filled_with_num, counter, dictionary, reverse_dictionary
    data_filled_with_num, counter, dictionary, reverse_dictionary = build_dataset(data_file, vocabulary_size)

    test_words = ['big']
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
            # print('step: ', step)
            batch_inputs, batch_labels = generate_batch(batch_size, num_samples, skip_window)
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

            # # Evaluate similarity after every 10000 iterations.
            # if step % 10000 == 0:
            #     sim = similarity.eval() #
            #     for i in range(sample_size):
            #         sample_word = reverse_dictionary[sample_examples[i]]
            #         top_k = 10  # Look for top-10 neighbours for words in sample set.
            #         nearest = (-sim[i, :]).argsort()[1:top_k + 1]
            #         log_str = 'Nearest to %s:' % sample_word
            #         for k in range(top_k):
            #             close_word = reverse_dictionary[nearest[k]]
            #             log_str = '%s %s,' % (log_str, close_word)
            #         try:
            #             print(log_str)
            #         except UnicodeEncodeError:
            #             print('UnicodeEncodeError:')
            #     print()

        final_embeddings = normalized_embeddings.eval()
        write_embedding_to_file(final_embeddings, embeddings_file_name)


def Compute_topk(model_file, input_adjective, top_k):
    global data_filled_with_num, counter, dictionary, reverse_dictionary

    sensitive_replace_word = ['UNK', 'PRON', 'CONJ', 'PREP', 'DET', 'NUM', 'SYM'] 

    model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=False)

    output = []
    temp_topk_multiplier = 5
    try:
        temp_result = model.most_similar(positive=[input_adjective + 'ADJ'], topn= top_k * temp_topk_multiplier)
    except:
        # if the queried word does not appear in embedding file
        return ['last', 'small', 'free', 'basic', 'few', 'huge', 'tough', 'best', 'large', 'main', 'full', 'common', 'top', 'fair', 'great', 'major', 'difficult', 'old', 'high', 'low', 'conservative', 'successful', 'different', 'likely', 'former', 'real', 'modern', 'vital', 'single', 'same', 'ready', 'worth', 'much', 'average', 'extra', 'bad', 'next', 'serious', 'short', 'big', 'final', 'more', 'certain', 'specific', 'current', 'malicious', 'confident', 'less', 'chief', 'key', 'clear', 'recent', 'corporate', 'early', 'special', 'able', 'many', 'wide', 'interested', 'significant', 'famous', 'mobile', 'private', 'little', 'prime', 'hard', 'new', 'important']

    words = [r[0] for r in temp_result]

    while len(output) < top_k:
        for word in words:
            if word in sensitive_replace_word:
                continue
            if word[-3:] == 'ADJ' and word[:-3] not in SELF_DEFINED_STOP_WORD:
                output.append(word[:-3])

        if temp_topk_multiplier > 10:
            break

        if len(output) < top_k:
            temp_topk_multiplier += 1
            temp_result = model.most_similar(positive=[input_adjective + 'ADJ'], topn= top_k * temp_topk_multiplier)
            words = [r[0] for r in temp_result]
            words = words[top_k * (temp_topk_multiplier -1) :]

    return output[:top_k]


if __name__ == "__main__":
    data = process_data('./BBC_Data.zip')
    model_file = 'adjective_embeddings.txt'
    adjective_embeddings(data, model_file, number_of_iterations, embedding_dim)
    top_k = 100

    from os import listdir
    from os.path import isfile, join

    match_counter = 0
    word_files = [f for f in listdir('dev_set/') if isfile(join('dev_set/', f))]

    for wf in word_files:
        computed_similar_words = Compute_topk(model_file, wf, top_k)
        ground_truth_words = []
        file = open(join('dev_set/', wf),'r')
        for w in file:
            ground_truth_words.append(w.strip())  

        print('Matched nearest to ', wf, ': ', end= " ")

        for csw in computed_similar_words:
            if csw in ground_truth_words:
                try:
                    print(csw, " ", end=" ")
                except UnicodeEncodeError:
                    print('UnicodeEncodeError:')
                match_counter += 1

        print("\n")

    print('average hits: ', match_counter / len(word_files))