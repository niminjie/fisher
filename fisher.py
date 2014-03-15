import sys
import numpy as np

class Fisher():
    pass

def file_to_matrix(file_path, head=True):
    trn_matrix, trn_label = [], []
    trn_list, trn_list_label = [], []

    cnt = 0
    if head:
        cnt += 1
    lines = open(file_path).readlines()
    for i in range(cnt, len(lines)):
        trn_list.append(lines[i].strip().split()[0:2])
        trn_list_label.append(lines[i].strip().split()[2])
    trn_matrix = np.array(trn_list)
    trn_label = np.array(trn_list_label)
    return trn_matrix, trn_label

def main():
    trn, trn_label = file_to_matrix(r'./synth/synth.te')
    tst, tst_label = file_to_matrix(r'./synth/synth.tr')

if __name__ == '__main__':
    main()
