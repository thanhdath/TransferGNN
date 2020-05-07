

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
0.918+-0.006
0.901+-0.011
0.917+-0.007
0.907+-0.010
0.911+-0.010
"""
C = """
0.625+-0.007
0.737+-0.013
0.621+-0.010
0.673+-0.009
0.672+-0.009
"""

A = [float(x.split("+-")[0]) for x in A.strip().split("\n")]
C = [float(x.split("+-")[0]) for x in C.strip().split("\n")]

print(sra(A, C))
