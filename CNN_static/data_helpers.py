import pandas as pd
import re
import numpy as np
import nltk
from nltk.corpus import stopwords

def load_data_and_labels(data_file):
    df = pd.read_csv(data_file, encoding='latin-1')
    text = df['x_input']
    label = df['y_label']
    x_text = [clean_str(sent) for sent in text]
    set_label = set(label)
    length = len(set_label)
    dic_label = dict(zip(range(len(set_label)), set_label))
    y = list()
    for name in label:
        l = [0] * length
        for k, v in dic_label.items():
            if name is v:
                l[k] = 1
                y.append(l)
    return [x_text, np.asarray(y)]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # shuffle data
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def clean_str(string):
    stop_word = stopwords.words("english")                #停词表
    #porter=nltk.PorterStemmer()                           #词干提取器
    string = re.sub(r"\(.*?\)", " ", string)
    string = re.sub(r'[^a-zA-Z \']', " ", string)
    string = [w for w in string.split() if w not in stop_word]
    string = " ".join(string)
    return string.lower()
