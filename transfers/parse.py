
import glob 
import re

logs = glob.glob("logs/transfers/*")
logs = sorted(logs, key=lambda x: int(x.split("/")[-1]))

# for log in logs:
#     one = open(log+"/1.txt").read()
#     vals = re.findall(r"Val: [0-9\.]+", one)[:2]
#     tf = open(log+"/1-from-0.txt").read()
#     valstf = re.findall(r"Val: [0-9\.]+", tf)[:2]
#     idx = log.split("/")[-1]
#     print(f"xxxxx Label {idx} =============")
#     for i, (a, b) in enumerate(zip(vals, valstf)):
#         print(f"Epoch {i}\t{a}\t{b}")

scores = []

for log in logs:
    one = open(log+"/1.txt").read()
    vals = re.findall(r"Val: [0-9\.]+", one)
    vals = [vals[0], vals[-1]]
    vals = [float(x.replace("Val: ", "")) for x in vals]
    
    # vals = [abs(0.5-vals[0]), abs(1-vals[1])]
    if vals[0] < 0.5:
        vals[0] = 1-vals[0]
    score = vals[1]-vals[0]
    scores.append((score, vals))
import numpy as np
scores = np.array(scores)
ids = np.argsort(-scores[:,0])
for id in ids:
    print(f"Label {id}: {scores[id][1][0]:.3f} - {scores[id][1][1]:.3f}")