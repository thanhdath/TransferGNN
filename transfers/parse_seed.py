import re
import glob 
import numpy as np
import os

logtypes = glob.glob("logs/transfers-sbm*")
# for logtype in sorted(logtypes):
#     print("\nLog type:", logtype)
#     sublogtype = os.listdir(logtype)
#     sublogtype = [re.sub("seed[0-9]+-", "", x) for x in sublogtype]
#     sublogtype = list(set(sublogtype))
#     for subtype in sorted(sublogtype):
#         print("\tSub type:", subtype)
#         seedfiles = glob.glob(logtype + "/*" + subtype)
#         nofinetune = []
#         bestvals = []
#         for file in seedfiles:
#             content = open(file).read()
#             vals = re.findall("Val: [0-9\.]+", content)
#             nofinetune.append(float(vals[0].replace("Val: ", "")))
#             bestvals.append(float(vals[-1].replace("Val: ", "")))
#         print(f"\t\tNo finetune: {np.mean(nofinetune):.3f}+-{np.std(nofinetune):.3f}")
#         print(f"\t\tBest val: {np.mean(bestvals):.3f}+-{np.std(bestvals):.3f}")

groups = {

}

for seed in range(100, 110):
    logdir = "logs/end2end-sbm-seed{}".format(seed)
    if not os.path.isdir(logdir):
        print("Not found {}".format(logdir))
        continue
    try:
        files = os.listdir(logdir)
        files = [x for x in files if "from-" in x]
        for file in files:
            if file not in groups:
                groups[file] = []
        for file in files:
            content = open(logdir + "/" + file).read()
            nofinetune = re.findall("Val: [0-9\.]+", content)[0]
            nofinetune = float(nofinetune.replace("Val: ", ""))
            groups[file].append(nofinetune)
    except:
        continue

for key, v in groups.items():
    print(f"{key}: {np.mean(v):.3f}+-{np.std(v):.3f}")
