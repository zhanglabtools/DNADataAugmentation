# last time modified: 2018/6/25
# 操纵DNA序列的一些函数
# 如果是N就直接转化成0即可

import numpy as np


# 主要作用是，这里要把形状变成[4, len(sequence), 1]
# 这里的one-hot coding还不太一样，NA数据直接补全成四个0.25
def dna_one_hot_coding(seq):
    res = np.zeros((4, len(seq), 1), dtype=np.float32)
    for i, nuc in enumerate(list(seq)):
        if nuc == 'A':
            res[0, i, 0] = 1.0
        elif nuc == 'C':
            res[1, i, 0] = 1.0
        elif nuc == 'G':
            res[2, i, 0] = 1.0
        elif nuc == 'T':
            res[3, i, 0] = 1.0
        else:
            res[:, i, 0] = 0
    return res


# 这里要保证只有A,C,G,T,N五种情况
def reverse_complement_enhance(s,
                               complement={'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}):
    result = [complement[i] for i in list(s.upper())]
    result = reversed(result)
    return ''.join(result)

# 只做mapping但是不做reverse
def mapping_enhance(s, complement={'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}):
    result = [complement[i] for i in list(s.upper())]
    return ''.join(result)
