import matplotlib.pyplot as plt
import numpy as np
import sys

class Fisher():
    def __init__(self, trn, trn_label, tst, tst_label):
        self.trn = trn
        self.trn_label = trn_label
        self.tst = tst
        self.tst_label = tst_label

    def getter_idx(self, idx, _):
        result = []
        for i in idx:
            result.append(_[i])
        return np.array(result)

    def separate(self, trn_label):
        idx_0 = []
        idx_1 = []
        for i in range(0, len(trn_label)):
            if trn_label[i] == 0:
                idx_0.append(i)
            else:
                idx_1.append(i)
        return idx_0, idx_1

    def avgPoint(self, m):
        return np.average(m, axis=0)

    def unitize_vector(self, v):
        return v / np.linalg.norm(v)

    def cal_threshold_1(self, w, avg_point_0, avg_point_1):
        return np.dot(w.transpose(), avg_point_0 + avg_point_1) / 2

    def cal_threshold_2(self, w, avg_point_0, avg_point_1):
        return np.dot(w.transpose(), len(avg_point_1) * avg_point_0 + len(avg_point_0) * avg_point_1) / (len(avg_point_1) + len(avg_point_0))

    def cal_threshold_3(self, w, avg_point_0, avg_point_1):
        n0 = len(avg_point_0)
        n1 = len(avg_point_1)

    def cal_w(self, cal_thresh=cal_threshold_2):
        idx_0, idx_1 = self.separate(self.trn_label)
        point_0 = self.getter_idx(idx_0, self.trn)
        point_1 = self.getter_idx(idx_1, self.trn)

        t_idx_0, t_idx_1 = self.separate(self.tst_label)
        t_point_0 = self.getter_idx(t_idx_0, self.tst)
        t_point_1 = self.getter_idx(t_idx_1, self.tst)

        plt.xlabel(u'x')
        plt.ylabel(u'y')
        #plt.plot(point_0[:, 0], point_0[:, 1], 'ro')
        #plt.plot(point_1[:, 0], point_1[:, 1], 'bo')
        plt.plot(t_point_0[:, 0], t_point_0[:, 1], 'rx')
        plt.plot(t_point_1[:, 0], t_point_1[:, 1], 'bx')

        avg_point_0 = self.avgPoint(point_0)
        print 'avg_point_0 ', avg_point_0
        avg_point_1 = self.avgPoint(point_1)
        print 'avg_point_1 ', avg_point_1

        # diff_0 = np.tile(avg_point_0, (len(idx_0), 1)) - point_0
        # diff_1 = np.tile(avg_point_1, (len(idx_1), 1)) - point_1
        # print 'diff_0', diff_0
        # print 'diff_1', diff_1
        diff_0 = point_0 - np.tile(avg_point_0, (len(idx_0), 1))
        diff_1 = point_1 - np.tile(avg_point_1, (len(idx_1), 1))

        s0 = np.dot(diff_0.transpose(), diff_0)
        s1 = np.dot(diff_1.transpose(), diff_1)
        sw = s0 + s1
        #sw_ = sw ** -1
        sw_ = np.linalg.inv(sw)
        w = np.dot(sw_, avg_point_0 - avg_point_1)
        x = np.arange(-0.1, 0.2, 0.1)
        y = w[1] / w[0] * x
        plt.plot(x, y, 'g-', linewidth=2.0)
        plt.show()
        n0 = len(point_0)
        n1 = len(point_1)
        thresh = cal_thresh(self, w, avg_point_0, avg_point_1)
        return w, thresh

    def train(self):
        w, thresh = self.cal_w()
        return w, thresh

    def test(self):
        w, thresh = self.cal_w()
        result = []
        hits = 0
        for i in range(0, len(self.tst)):
            y = w.transpose().dot(self.tst[i])
            # print y, thresh
            if thresh >= y:
                result.append(1)
            else :
                result.append(0)
        for i in range(0, len(self.tst_label)):
            if self.tst_label[i] == result[i]:
                hits += 1
        print hits * 1.0 / len(self.tst_label)


def convert_to_float(l):
   return [float(d) for d in l]

def file_to_matrix(file_path, head=True):
    trn_matrix, trn_label = [], []
    trn_list, trn_list_label = [], []
    cnt = 0
    if head:
        cnt += 1
    lines = open(file_path).readlines()
    for i in range(cnt, len(lines)):
        trn_list.append(convert_to_float(lines[i].strip().split()[0:2]))
        trn_list_label.append(convert_to_float(lines[i].strip().split()[2]))
    trn_matrix = np.array(trn_list)
    trn_label = np.array(trn_list_label)
    return trn_matrix, trn_label

def main():
    trn, trn_label = file_to_matrix(r'./synth/synth.tr')
    tst, tst_label = file_to_matrix(r'./synth/synth.te')

    fisher = Fisher(trn, trn_label, tst, tst_label)
    fisher.train()
    fisher.test()

if __name__ == '__main__':
    main()
