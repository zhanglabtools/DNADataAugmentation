# last time modified: 2018/9/25
# 用于在线数据扩增的几个函数
import random
from HelpFunc.MISC import rep
from HelpFunc.DNA import mapping_enhance
from HelpFunc.DNA import reverse_complement_enhance


# 0) 只做反转
def ReverseAug(seqs, labels, shuffle=False):
    augLabel = rep(labels, each=2)
    augSeq = []
    for s in seqs:
        augSeq.append(s)
        augSeq.append(s[::-1])
    if shuffle:
        temp = list(zip(augSeq, augLabel))
        random.shuffle(temp)
        augSeq[:], augLabel[:] = zip(*temp)
    return augSeq, augLabel


# 1) 只做mapping
def MappingAug(seqs, labels, mapping=None, shuffle=False):
    augLabel = rep(labels, each=2)
    augSeq = []
    if mapping is None:
        for s in seqs:
            augSeq.append(s)
            augSeq.append(mapping_enhance(s))
    else:
        for s in seqs:
            augSeq.append(s)
            augSeq.append(mapping_enhance(s, complement=mapping))
    if shuffle:
        temp = list(zip(augSeq, augLabel))
        random.shuffle(temp)
        augSeq[:], augLabel[:] = zip(*temp)
    return augSeq, augLabel


# 1) Reverse + mapping
def ReverseMappingAug(seqs, labels, mapping=None, shuffle=False):
    augLabel = rep(labels, each=2)
    augSeq = []
    if mapping is None:
        for s in seqs:
            augSeq.append(s)
            augSeq.append(reverse_complement_enhance(s))
    else:
        for s in seqs:
            augSeq.append(s)
            augSeq.append(reverse_complement_enhance(s, complement=mapping))
    if shuffle:
        temp = list(zip(augSeq, augLabel))
        random.shuffle(temp)
        augSeq[:], augLabel[:] = zip(*temp)
    return augSeq, augLabel
