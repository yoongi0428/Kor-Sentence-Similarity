import os
import numpy as np

from util.kor_char_parser import decompose_str_as_one_hot
from util.kor_eum_parser import decompose_str_as_one_hot_eum


class Dataset:

    def __init__(self, dataset_path, mode, max_len=420, eumjeol=False):
        self.PATH = dataset_path
        self.data_file = 'data'
        self.label_file = 'label'
        self.eumjeol = eumjeol
        self.strmaxlen = max_len

        queries_data = os.path.join(dataset_path, mode + '_' + self.data_file)
        labels_data = os.path.join(dataset_path, mode + '_' + self.label_file)

        with open(queries_data, 'rt', encoding='utf8') as f:
            q1_2 = f.readlines()

            q1 =[]
            q2 =[]
            for line in q1_2:
                lines = line.split('\t')
                q1.append(lines[0])
                q2.append(lines[1])

            self.queries1 = preprocess(q1, max_len, eumjeol)
            self.queries2 = preprocess(q2, max_len, eumjeol)

        with open(labels_data) as f:
            self.labels = np.array([[np.float32(x)] for x in f.readlines()])

        self.shuffle()


    def __len__(self):
        return len(self.queries1)

    def __getitem__(self, idx):
        return self.queries1[idx], self.queries2[idx], self.labels[idx]

    def shuffle(self):
        p = np.arange(len(self.queries1))
        np.random.shuffle(p)
        self.queries1 = self.queries1[p, :]
        self.queries2 = self.queries2[p, :]
        self.labels = self.labels[p]

def preprocess(data: list, max_length: int, eumjeol: bool):
    vectorized_data = [decompose_str_as_one_hot_eum(datum, warning=False) if eumjeol else decompose_str_as_one_hot(datum, warning=False) for datum in data]
    zero_padding = np.zeros((len(data), max_length), dtype=np.int32)
    for idx, seq in enumerate(vectorized_data):
        length = len(seq)
        if length >= max_length:
            length = max_length
            zero_padding[idx, :length] = np.array(seq)[:length]
        else:
            zero_padding[idx, :length] = np.array(seq)
    return zero_padding