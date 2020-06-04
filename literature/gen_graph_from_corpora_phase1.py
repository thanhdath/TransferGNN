import glob 
import sys 
import os 
import stanfordnlp

files = [sys.argv[1]]
# files = glob.glob("corpora/*/")[:2]
print(files)
nlp = stanfordnlp.Pipeline(lang=sys.argv[2]) # lang: en, fr, es = spanish

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
    print(file)
    filename= [x for x in file.split("/") if len(x) > 0][-1]

    sentences = []
    word2pos = {}
    word_cooccur = {} 
    sen_words = []
    sen_postags = []

    with open(f"corpora/{filename}/{filename}-sentences.txt") as fp:
        for line in fp:
            sid, sentence = line.strip().split("\t")
            sentences.append(sentence)

    for sentence in tqdm(sentences):
        doc = nlp(sentence)
        for sent in doc.sentences:
            sen_words.append([sent.words[i].text.lower() for i in range(len(sent.words))])
            sen_postags.append([sent.words[i].upos for i in range(len(sent.words))])
    #         for i in range(len(sent.words)):
    #             wordi = sent.words[i].text.lower()
    #             word2pos[wordi] = word2pos.get(wordi, []) + [sent.words[i].pos]
    #             for j in range(max(0, i-window_size//2), min(len(sent.words), i+window_size//2)):
    #                 wordj = sent.words[j].text.lower()
    #                 word_cooccur[(wordi, wordj)] = word_cooccur.get((wordi,wordj), 0) + 1
    #                 word_cooccur[(wordj, wordi)] = word_cooccur.get((wordj,wordi), 0) + 1
    # word2pos = {k: list(set(v)) for k, v in word2pos.items()}
    # word2poss.append(word2pos)
    # wordcoss.append(word_cooccur)
    try:
        if not os.path.isdir(f"data/{filename}"):
            os.makedirs(f"data/{filename}")
        # np.savez_compressed(f"data/{filename}/tokenize.npz", word2pos=np.array(word2pos),
        #     word_cooccur=np.array(word_cooccur))
        np.savez_compressed(f"data/{filename}/tokenize.npz", sen_words=np.array(sen_words),
            sen_postags=np.array(sen_postags))
    except Exception as err:
        print(err)
