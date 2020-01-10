import re
import glob 
import numpy as np
import os

logtypes = glob.glob("logs/transfers-syn*")
for logtype in sorted(logtypes):
    print("\nLog type:", logtype)
    sublogtype = os.listdir(logtype)
    sublogtype = [re.sub("seed[0-9]+-", "", x) for x in sublogtype]
    sublogtype = list(set(sublogtype))
    for subtype in sorted(sublogtype):
        print("\tSub type:", subtype)
        seedfiles = glob.glob(logtype + "/*" + subtype)
        nofinetune = []
        bestvals = []
        for file in seedfiles:
            content = open(file).read()
            vals = re.findall("Val: [0-9\.]+", content)
            nofinetune.append(float(vals[0].replace("Val: ", "")))
            bestvals.append(float(vals[-1].replace("Val: ", "")))
        print(f"\t\tNo finetune: {np.mean(nofinetune):.3f}+-{np.std(nofinetune):.3f}")
        print(f"\t\tBest val: {np.mean(bestvals):.3f}+-{np.std(bestvals):.3f}")