from os import listdir
from os.path import isfile, join
from testMainDonotRemoveWords import Compute_topk

model_file = 'adjective_embeddings.txt'
top_k = 100

match_counter = 0
word_files = [f for f in listdir('dev_set/') if isfile(join('dev_set/', f))]

for wf in word_files:
    # print('word file: ', wf)
    computed_similar_words = Compute_topk(model_file, wf, top_k)
    print(computed_similar_words[:10])
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

    print()

print('average hits: ', match_counter / len(word_files))