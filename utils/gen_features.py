
# 
import numpy as np
import io
def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.array([float(x) for x in tokens[1:]])
    return data

idx2word = {}
words = []
with open("../words.txt") as fp:
    for i, line in enumerate(fp):
        word = line.strip().split()[0]
        idx2word[i] = word
        words.append(word)

word_vectors = load_vectors('../Combine/wiki-news-300d-1M.vec')
dim=300

features = []
for i in range(len(words)):
    feature = word_vectors[idx2word[i]]
    features.append(feature)

features = np.array(features)
np.savez_compressed("../features.npz", features=features)
