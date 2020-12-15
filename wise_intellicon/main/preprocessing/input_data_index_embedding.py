from preprocessing.preprocessing_code_190418 import preprocess, title_catcher, date_process, phone_process, time_process, title_process
import pickle as pkl
import re
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from collections import Counter

from konlpy.tag import Komoran
from keras.preprocessing.sequence import pad_sequences

komoran_dir = './data/userdict_190411.txt'
komoran = Komoran(userdic = komoran_dir)
morp_type = 'morp'

def load_dataset(data_dir) :
    origin_data = pd.read_excel(data_dir)
    if len(origin_data.columns) == 9:
        origin_data.columns = ['doc_id', 'par_id', 'art_id', 'line_id', 'text', 'par_label', 'line_label', 'none1', 'none2']
        origin_data['split_id'] = origin_data['doc_id'].map(str) + '_' + origin_data['par_label']
    elif len(origin_data.columns) == 7:
        origin_data.columns = ['doc_id', 'par_id', 'art_id', 'line_id', 'text', 'par_label', 'line_label']
        origin_data['split_id'] = origin_data['doc_id'].map(str) + '_' + origin_data['par_label']
    else :
        print('version v01')
        origin_data.columns = ['text', 'label', 'par_label', 'id', 'par_id']
        origin_data['split_id'] = origin_data['id'].map(str) + '_' + origin_data['par_label']
    return origin_data

def split_dataset(data, type, seed):
    if type == 'par' :
        split_column = 'split_id'
    elif type == 'line' :
        split_column = 'doc_id'

    contract_names = np.unique(data[split_column])
    train, test = train_test_split(contract_names, test_size=0.3, random_state=seed)
    valid, test = train_test_split(test, test_size=0.5, random_state=seed)

    x_all = []
    y_all = []

    x_train = []
    y_train = []
    matrix_train = []
    x_valid = []
    y_valid = []
    matrix_valid = []
    x_test = []
    y_test = []
    matrix_test = []

    for name in contract_names:
        temp = data[data[split_column] == name]
        temp_contract = []
        temp_answer = []
        temp_par = []
        for c, l, p in zip(temp['text'].values, temp['line_label'].values, temp['par_label']):
            temp_contract.append(c)
            temp_answer.append(l)
            temp_par.append(p)
        if name in train:
            x_train.append(temp_contract)
            y_train.append(temp_answer)
            matrix_train.append(temp_par)
        elif name in valid:
            x_valid.append(temp_contract)
            y_valid.append(temp_answer)
            matrix_valid.append(temp_par)
        elif name in test:
            x_test.append(temp_contract)
            y_test.append(temp_answer)
            matrix_test.append(temp_par)

        x_all.append(temp_contract)
        y_all.append(temp_answer)

    return x_all, y_all, x_train, y_train, x_valid, y_valid, x_test, y_test, matrix_train, matrix_valid, matrix_test

def text_preprocess(text, morp):

    text = preprocess(text)
    text = title_process(text)
    text = time_process(text)
    text = date_process(text)
    text = phone_process(text)
    text = re.sub('[^가-힣".,()~%_ ]+', '', text)
    if morp=='morp' :
        try:
            text = ' '.join(np.array(komoran.pos(text))[:, 0])
        except:
            text = '_빈칸_'

    return text

def join_date(all_data):
    p = re.compile('[0-9]{4}[ .년]{0,3}[0-9]{1,2}[ .월]{0,3}[0-9]{1,2}[ .일]{0,3}')
    w = re.compile("[^년월일\d ,]")

    #split_date_idx : p정규 표현식에 의해 날짜가 있고 PR-04-13 이외의 날짜 문장들을 제외하기 위해 find함수와 w정규 표현식을 통해 제거하고 join할 문장들의 index를 반환
    # split_date_idx = [idx for idx, lines in enumerate(all_data[['text']].values) if len(p.findall(str(lines)))==1 and len(lines)<=15]
    
    split_date_idx = []
    for idx, lines in enumerate(all_data[['text']].values) :
        if len(p.findall(str(lines)))>=1 and str(lines).find(':')==-1 and str(lines).find(',')!=-1:
            if len(w.findall(str(lines)))<=10 :
                split_date_idx.append(idx)
    
    
    
    #split_date_idx에서 앞뒤로 1을 빼고 3까지 sequential한 경우
    date_diff = [i for i in range(len(split_date_idx)-1) if split_date_idx[i+1]-split_date_idx[i]>=3]
    
    #sequential한 index만 추가
    seq_date_idx = []
    seq_date_idx.append(split_date_idx[0:date_diff[0]+1])
    for i in range(len(date_diff)-1) :
        seq_date_idx.append(split_date_idx[date_diff[i]+1:date_diff[i+1]+1])
    
    #split 되어있는 date를 한줄에 합침
    for j in seq_date_idx :
        all_data.iloc[j[0], 4] = ' '.join(all_data.iloc[j]['text'].values[0:50])
    
    #나머지 sequantial date drop
    all_data = all_data.drop(np.concatenate([i[1:] for i in seq_date_idx]))
    
    return all_data

def bow_vocab(train_data, vocab_to_int_dir, int_to_vocab_dir):
    corpus = [text_preprocess(line, morp_type) for contract in train_data for line in contract if not title_catcher(preprocess(line))]
    vocab = np.unique(['PUNC' if len(re.sub('[^가-힣_]+', '', w)) == 0 else re.sub('[^가-힣_]+', '', w) for w in np.unique(' '.join(corpus).split())])
    # vocab내부에 정규표현식에 걸리면 ''로 replace해주고, vocab자체가 정규표현식에 걸리면 PUNC로 replace

    vocab_to_int = {word: ii for ii, word in enumerate(vocab, 2)}
    int_to_vocab = {j: i for i, j in vocab_to_int.items()}

    with open(vocab_to_int_dir, 'wb') as h:
        pickle.dump(vocab_to_int, h)
    with open(int_to_vocab_dir, 'wb') as h:
        pickle.dump(int_to_vocab, h)
    return vocab_to_int, int_to_vocab

def load_bow_vocab(vocab_to_int_dir, int_to_vocab_dir):
    with open(vocab_to_int_dir, 'rb') as h:
        vocab_to_int = pickle.load(h)
    with open(int_to_vocab_dir, 'rb') as h:
        int_to_vocab = pickle.load(h)
    return vocab_to_int, int_to_vocab

def bow_label(valid_class):
    label_to_int = {word: i for i, word in enumerate(valid_class)}
    int_to_label = {i: word for i, word in enumerate(valid_class)}

    return label_to_int, int_to_label

def max_length(all_data):
    max_len = np.max([len(text_preprocess(w, morp_type).split()) for w in np.concatenate(all_data)])
    max_row = np.max([len(contract) for contract in all_data])
    mean_len = np.mean([len(text_preprocess(w, morp_type).split()) for w in np.concatenate(all_data)])
    mean_row = np.mean([len(contract) for contract in all_data])
    return max_len, max_row, mean_len, mean_row

def x_data_set(sentences, max_len, vocab_to_int):
    def sentence_to_idx(sentence):
        def word_to_idx(text):
            try:
                re_text = re.sub('[^가-힣".,()~%_ ]+', '', text)
                re_text = re.sub('[^가-힣_]+', 'PUNC', re_text)
                return vocab_to_int[re_text]
            except:
                return 1
        p = re.compile('([ㄱ-ㅎㅏ-ㅣ]+)')
        return [word_to_idx(word) for word in sentence.split() if len(p.findall(word)) == 0]
    sentences = [sentence_to_idx(text_preprocess(sentence, morp_type)) for sentence in np.concatenate(sentences)]
    return pad_sequences(sentences, maxlen=max_len, padding='post')

def y_data_set(sentences, valid_class, max_row):
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(valid_class.reshape(-1,1))
    y_labels = np.array([enc.transform(i.reshape(-1,1)).toarray()[0] for i in np.concatenate(sentences)])
    return y_labels

def int_to_label(y_vectors, valid_class):
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(valid_class.reshape(-1,1))
    labels = enc.inverse_transform(y_vectors)
    return labels

def labels_to_vecs(labels, par_class, label_to_int):
    output = np.zeros(len(par_class))
    for label in labels:
        if label in label_to_int.keys():
            output += np.eye(len(par_class))[label_to_int[label]]
    return output.tolist()

def make_index_embed(row, col, par_class, maxrow):
    index_embed = []
    
    for idx in range(len(row)):
        init_matrix = np.zeros([maxrow,len(par_class)])
        init_matrix[row[idx], np.where(1==col[idx])] = 1
        index_embed.append(init_matrix)
    
    return np.array(index_embed)

def evaluate(input_data, real, model):
    # evaluate function format
    # model_input_data = [x_test_idx, x_test_]
    # real = y_test_
    # model = idec

    pred_data = model.predict(input_data)
    
    pred_label = [np.argmax(i) for i in pred_data]
    real_label = [np.argmax(i) for i in real]
    
    accuracy = np.mean([pred==real for pred, real in zip(pred_label, real_label)])
    
    return accuracy

def split_newdataset_sw(data, standard, seed):
    contract_names = np.unique(data[standard])
    train, test = train_test_split(contract_names, test_size = 0.3, random_state = seed)
    valid, test = train_test_split(test, test_size = 0.5, random_state = seed)
    
    x_all, y_all = [], []
    x_train, y_train, matrix_train = [], [], []
    x_valid, y_valid, matrix_valid = [], [], []
    x_test, y_test, matrix_test = [], [], []
    
    for name in contract_names:
        temp = data[data[standard] == name]
        temp_contract, temp_answer, temp_par = [], [], []

        for c, l, p in zip(temp['doc'].values, temp['line_label'].values, temp['label']):
            temp_contract.append(c)
            temp_answer.append(l)
            temp_par.append(p)

        if name in train:
            x_train.append(temp_contract)
            y_train.append(temp_answer)
            matrix_train.append(temp_par)
        elif name in valid:
            x_valid.append(temp_contract)
            y_valid.append(temp_answer)
            matrix_valid.append(temp_par)
        elif name in test:
            x_test.append(temp_contract)
            y_test.append(temp_answer)
            matrix_test.append(temp_par)

        x_all.append(temp_contract)
        y_all.append(temp_answer)
    
    matrix_train = [m*len(l[0]) for m,l in zip(matrix_train, y_train)]
    matrix_valid = [m*len(l[0]) for m,l in zip(matrix_valid, y_valid)]
    matrix_test = [m*len(l[0]) for m,l in zip(matrix_test, y_test)]
    
    x_train = [x_train[con][0] for con in range(len(x_train))]
    x_valid = [x_valid[con][0] for con in range(len(x_valid))]
    x_test = [x_test[con][0] for con in range(len(x_test))]
    x_all = [x_all[con][0] for con in range(len(x_all))]

    y_train = [y_train[con][0] for con in range(len(y_train))]
    y_valid = [y_valid[con][0] for con in range(len(y_valid))]
    y_test = [y_test[con][0] for con in range(len(y_test))]
    y_all = [y_all[con][0] for con in range(len(y_all))]

    return x_all, y_all, x_train, y_train, x_valid, y_valid, x_test, y_test, matrix_train, matrix_valid, matrix_test

def split_ptl_inference(data, standard):
    contract_names = np.unique(data[standard])
     
    x_all = []
    
    for name in contract_names:
        temp = data[data[standard] == name]
        temp_contract = []

        for c in temp['doc'].values:
            temp_contract.append(c)
        x_all.append(temp_contract)
    x_all = [x_all[con][0] for con in range(len(x_all))]

    return x_all

def document_label_dataset_training(processed_data):
    processed_data = processed_data.reset_index()
    contents = processed_data['text'].tolist()
    paragraph_class = processed_data['par_label'].tolist()
    line_class = processed_data['line_label'].tolist()
    
    temp = []
    for text in processed_data['text']:
        try:
            result = title_catcher(text)
            temp.append(result)
        except:
            temp.append(False)
    processed_data['title'] = temp
    
    start_idx = processed_data[processed_data['title'] == True].index.tolist()
    end_idx = start_idx[1:]
    end_idx.append(processed_data.index[-1] + 1)
    
    contract = []
    line_label = []
    for start, end in zip(start_idx, end_idx):
        temp = processed_data['text'][start:end]
        contract.append(temp)
        temp2 = line_class[start:end]
        line_label.append(temp2)
    
    label = [paragraph_class[value] for value in start_idx]
    new_df = pd.DataFrame({"doc" : contract, "label" : label, "line_label" : line_label}).reset_index()

    return new_df

def document_label_dataset_infer(processed_data):
    processed_data = processed_data.reset_index()
    contents = processed_data.iloc[:, 5].tolist()
    # paragraph_class = processed_data.iloc[:, 6].tolist()
    
    temp = []
    for text in processed_data['text']:
        try:
            result = title_catcher(text)
            temp.append(result)
        except:
            temp.append(False)
    processed_data['title'] = temp
    
    start_idx = processed_data[processed_data['title'] == True].index.tolist()
    end_idx = start_idx[1:]
    end_idx.append(processed_data.index[-1] + 1)
    
    contract = []
    for start, end in zip(start_idx, end_idx):
        temp = processed_data['text'][start:end]
        contract.append(list(temp.values))
    
    # label = [paragraph_class[value] for value in start_idx]
    new_df = pd.DataFrame({"doc" : contract}).reset_index()
    return new_df    

def tagging_row_index(x_data):
    tagging_row = np.concatenate([[line_idx for line_idx, line in enumerate(par)] for par in x_data])
    return tagging_row

def row_embed(rows, max_row):
    row_embed = []
    for idx,row in enumerate(rows) :
        init_matrix = np.zeros([max_row])
        init_matrix[row]=1
        row_embed.append(init_matrix)
    return np.array(row_embed)

def vecs2labels(vecs, int_to_label):
    output = []
    for i, vec in enumerate(vecs):
        if vec == 1:
            output.append(int_to_label[i])
    return output