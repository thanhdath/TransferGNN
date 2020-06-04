import glob 
import sys 
import os

files = glob.glob("data/*news*/")
print(files)

import pandas as pd 
import numpy as np
import os 
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import json
from scipy.io import savemat
word2poss = []
wordcoss = []
window_size = 999
for file in files:
    filename= [x for x in file.split("/") if len(x) > 0][-1]
    if not os.path.isfile(f"word2vec/word2pos-{filename}.npy"):
        tokenize_file = file + "/tokenize.npz"
        npz = np.load(tokenize_file, allow_pickle=True)
        sen_words = npz["sen_words"][()]
        sen_postags = npz["sen_postags"][()]
        word2pos = {}
        word_cooccur = {}

        for sentence, postags in tqdm(zip(sen_words, sen_postags)):
            for i in range(len(sentence)):
                wordi = sentence[i]
                word2pos[wordi] = word2pos.get(wordi, []) + [postags[i]]
                for j in range(max(0, i-window_size//2), min(len(sentence), i+window_size//2)):
                    if i == j: continue
                    wordj = sentence[j]
                    word_cooccur[(wordi, wordj)] = word_cooccur.get((wordi,wordj), 0) + 1
                    word_cooccur[(wordj, wordi)] = word_cooccur.get((wordj,wordi), 0) + 1
        word2pos = {k: list(set(v)) for k, v in word2pos.items()}
        np.save(f"word2vec/word2pos-{filename}.npy", np.array(word2pos))
        np.save(f"word2vec/word_cooccur-{filename}.npy", np.array(word_cooccur))
    else:
        word2pos = np.load(f"word2vec/word2pos-{filename}.npy", allow_pickle=True)[()]
        word_cooccur = np.load(f"word2vec/word_cooccur-{filename}.npy", allow_pickle=True)[()]
    word2poss.append(word2pos)
    wordcoss.append(word_cooccur)
    # average number of postags per word
    print("Average number of postags/word: ", np.mean([len(x) for x in word2pos.values()]))

all_postags = set()
for i, word2pos in enumerate(word2poss):
    postags = set()
    for val in word2pos.values():
        postags = postags.union(set(val))
        all_postags = all_postags.union(set(val))
    print("File", i, "Number of postags: ", len(postags))
print("Number of postags: ", len(all_postags))
postag2idx = {k: i for i, k in enumerate(all_postags)}

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
word_vectors = {}
if not os.path.isfile("word2vec/en.npz"):
    word_vectors["en"] = load_vectors('word2vec/wiki.en.vec')
    np.save("word2vec/en.npy", np.array(word_vectors["en"]))
else:
    word_vectors["en"] = np.load('word2vec/en.npy', allow_pickle=True)[()]

if not os.path.isfile("word2vec/fr.npz"):
    word_vectors["fr"] = load_vectors('word2vec/wiki.fr.vec')
    np.save("word2vec/fr.npy", np.array(word_vectors["fr"]))
else:
    word_vectors["fr"] = np.load('word2vec/fr.npz', allow_pickle=True)[()]

if not os.path.isfile("word2vec/es.npz"):
    try: 
        word_vectors["sp"] = load_vectors('word2vec/wiki.es.vec')
        np.save("word2vec/es.npy", np.array(word_vectors["sp"]))
    except:
        pass
else:
    word_vectors["sp"] = np.load('word2vec/es.npy', allow_pickle=True)[()]
dim=300

import numpy as np
for ifile, file in enumerate(files):
    print(ifile, file)
    filename= [x for x in file.split("/") if len(x) > 0][-1]
    lang = filename[:2]
    print(lang)
    word2pos = word2poss[ifile]
    word_cooccur = wordcoss[ifile]
    edgelist = []
    labels = []
    features = []
    words_has_features = [x for x in word2pos.keys() if x in word_vectors[lang]]
    word2idx = {word: i for i, word in enumerate(words_has_features)}
    idx2word = {v:k for k, v in word2idx.items()}
    added = {}
    print("== edgelist")
    for src, trg in word_cooccur.keys():
        if (src,trg) not in added and src in word2idx and trg in word2idx:
            edgelist.append([word2idx[src], word2idx[trg], word_cooccur[(src, trg)]])
            added[(src,trg)] = True
            added[(trg,src)] = True
    print("== labels")
    for j in range(len(word2idx)):
        word = idx2word[j]
        label = [postag2idx[x] for x in word2pos[word]]
        labels.append(label)
    print("== features")
    for j in range(len(word2idx)):
        word = idx2word[j]
        feature = word_vectors[lang][word]
        features.append(feature)
    features = np.array(features)
    edgelist = np.array(edgelist)
    if not os.path.isdir(f"data/{filename}"):
        os.makedirs(f"data/{filename}")
    with open(f"data/{filename}/{filename}.txt", "w+") as fp:
        for src,trg, w in edgelist:
            fp.write("{},{},{}\n".format(int(src), int(trg), w))
    with open(f"data/{filename}/labels.txt", "w+") as fp:
        for j in range(len(word2idx)):
            label = labels[j]
            fp.write("{}\t{}\n".format(j, "\t".join([str(x) for x in label])))
    np.savez_compressed(f"data/{filename}/features.npz", features=features)

