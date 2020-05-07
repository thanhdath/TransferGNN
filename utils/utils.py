

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
0.5346
0.9365
0.3897
0.4978
0.5478
"""
C = """
0.4155
0.548
0.3937
0.4134
0.4067
"""

A = [float(x) for x in A.strip().split("\n")]
C = [float(x) for x in C.strip().split("\n")]

print(sra(A, C))
