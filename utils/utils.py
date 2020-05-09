

def sra(A, C):
    """
    A: list of accuracy Atest-Xtest from Atrain-Xtrain
    C: list of accuracy A2-Xtest from AtrainF-Xtrain
    """
    L = len(A)  # number of models
    sra = 0
    for j in range(L):
        for k in range(L):
            if k == j:
                continue
            sra += int((A[j] - A[k]) * (C[j] - C[k]) > 0)
    sra /= L * (L - 1)
    return sra

A = """
0.682+-0.016
0.874+-0.054
0.480+-0.014
0.616+-0.040
0.679+-0.005
"""
C = """
0.873+-0.016
0.875+-0.040
0.671+-0.076
0.723+-0.086
0.872+-0.013
"""

A = [float(x.split("+-")[0]) for x in A.strip().split("\n")]
C = [float(x.split("+-")[0]) for x in C.strip().split("\n")]

print(sra(A, C))
