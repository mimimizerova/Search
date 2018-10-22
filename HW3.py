from gensim.models import Word2Vec, KeyedVectors
import nltk
from nltk.corpus import stopwords
from pymystem3 import Mystem
from nltk.tokenize import word_tokenize

w2v_model = Word2Vec.load('araneum_none_fasttextcbow_300_5_2018/araneum_none_fasttextcbow_300_5_2018.model')
mystem = Mystem()

def preprocessing(input_text, del_stopwords=True, del_digit=True):
    """
    :input: raw text
    1. lowercase, del punctuation, tokenize
    2. normal form
    3. del stopwords
    4. del digits
    :return: lemmas
    """
    russian_stopwords = set(stopwords.words('russian'))
    words = [x.lower().strip(punctuation+'»«–…') for x in word_tokenize(input_text)]
    lemmas = [mystem.lemmatize(x)[0] for x in words if x]

    lemmas_arr = []
    for lemma in lemmas:
        if del_stopwords:
            if lemma in russian_stopwords:
                continue
        if del_digit:
            if lemma.isdigit():
                continue
        lemmas_arr.append(lemma)
    return lemmas_arr
     
from judicial_splitter import split_paragraph, get_sentences
 
def get_paragraphs(file):
    with open(file, 'r', encoding='utf-8') as f:
        text = f.read() 
        paragraphs = split_paragraph(get_sentences(text), 4) 
        return paragraphs
 
def get_w2v_vectors(paragraph, model): 
    """
    Получает вектор документа
    (Так как в качестве документа нужно брать параграфы текста, данная функция получает вектор параграфа)
    """
    n = 0
    vector = [0] * 300
    lemmas = preprocessing(paragraph)
    for lemma in lemmas:
        try:
            vector += model.wv[lemma]
            n += 1
        except:
            None
    if n != 0:
        vector = vector / n
    return vector


def save_w2v_base(path, model):
    """Индексирует всю базу для поиска через word2vec"""
    w2v_base = []
    for file in os.listdir(path=path):
        for paragraph in get_paragraphs(path+'/'+file):
            vec = {}
            vec['id'] = file
            vec['text'] = paragraph
            vec['vector'] = get_w2v_vectors(paragraph, model)
            w2v_base.append(vec)
    return w2v_base

path = '/Users/kata/article_test'

w2v_base = save_w2v_base(path, w2v_model)

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

def get_tagged_data(path):
    tagged_data = []
    i = 0
    for file in os.listdir(path=path):
        for paragraph in get_paragraphs(path+'/'+file):
            data = preprocessing(paragraph, del_stopwords=False)
            tagged_data.append(TaggedDocument(words=data, tags=[i]))
            i+=1
    return tagged_data
    
def train_doc2vec(tagged_data):
    model = Doc2Vec(vector_size=100, min_count=5, alpha=0.025, min_alpha=0.025, epochs=100, workers=2, dm=1)
    model.build_vocab(tagged_data)
    model.random.seed(12345)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    
    return model
    
d2w_model = train_doc2vec(get_tagged_data(path))

def get_d2v_vectors(text, model):
    """Получает вектор документа"""
    return model.infer_vector(text)

def save_d2v_base(path, model):
    """Индексирует всю базу для поиска через doc2vec"""
    d2v_base = []
    for file in os.listdir(path=path):
        for paragraph in get_paragraphs(path+'/'+file):
            vec = {}
            vec['id'] = file
            vec['text'] = paragraph
            vec['vector'] = get_d2v_vectors(paragraph, model)
            d2v_base.append(vec)
    return d2v_base
 
d2v_base = save_d2v_base(path, d2w_model)
 
 
from gensim import matutils
import numpy as np 

def similarity(v1, v2):
    v1_norm = matutils.unitvec(np.array(v1))
    v2_norm = matutils.unitvec(np.array(v2))
    return np.dot(v1_norm, v2_norm)
    
import math
from math import log
from collections import Counter
import numpy as np


def get_information(path):
    """
    Create inverted index by input doc collection
    Get essential information from input doc collection
    :return: inverted index, 
    """
    dictionary = {}
    files_len = {}
    for file in os.listdir(path):
        with open(path +'/'+ file, 'r', encoding='utf-8') as f:
            text = f.read()
            words = preprocessing(text)
            files_len[file] = len(words)
            counter = Counter(words)
            for word in counter:
                if word in dictionary:
                    dictionary[word][file] = counter[word]
                else:
                    dictionary[word] = {}
                    dictionary[word][file] = counter[word]
    return files_len, dictionary
    
 files_len, index = get_information(path)
 
 
 k1 = 2.0
b = 0.75

def score_BM25(qf, dl, avgdl, k1, b, N, n) -> float:
    """
    Compute similarity score between search query and documents from collection
    :return: score
    """
    score = math.log((N-n+0.5)/(n+0.5)) * (k1+1)*qf/(qf+k1*(1-b+b*(dl/avgdl)))
    return score

def compute_sim(word, index, files_len):
    """
    Compute similarity score between search query and documents from collection
    :return: score
    """
    N = len(files_len)
    avgdl = sum(files_len.values())/N
    if word in index:
        n = len(index[word])
        result = {}
        for file in index[word]:
            qf = index[word][file]
            score = score_BM25(qf, files_len[file], avgdl, k1, b, N, n)
            result[file] = score
        return result
    else:
        return {}


def get_search_result(inquiry):
    """
    Compute sim score between search query and all documents in collection
    :return: list of files
    """
    global index, files_len
    score = defaultdict(int)
    words = preprocessing(inquiry)
    for word in words:
        result = compute_sim(word, index, files_len)
        for file in result:
            score[file] += result[file]  
        
    return sorted(score, key=score.get)[:10]
    
from collections import defaultdict
get_search_result('суд')

#['83.txt',
#'55.txt',
#'94.txt',
#'51.txt',
#'95.txt',
#'3.txt',
#'53.txt',
#'60.txt',
#'45.txt',
#'46.txt']


def search_w2v(inquiry, base, model):
    v1 = get_w2v_vectors(inquiry, model)
    score = defaultdict(int)
    for vec in base:
        v2 = vec['vector']
        sim = similarity(v1, v2)
        score[vec['id']] = sim
    return sorted(score, key=score.get)[:10]
 
#['74.txt',
# '16.txt',
# '2.txt',
# '24.txt',
# '4.txt',
# '53.txt',
# '73.txt',
# '77.txt',
# '99.txt',
# '100.txt']
        

def search_d2v(inquiry, base, model):
    v1 = get_d2v_vectors(inquiry, model)
    score = defaultdict(int)
    for vec in base:
        v2 = vec['vector']
        sim = similarity(v1, v2)
        score[vec['id']] = sim
    return sorted(score, key=score.get)[:10]
    
#['2.txt',
# '73.txt',
# '74.txt',
# '24.txt',
# '4.txt',
# '53.txt',
# '99.txt',
# '16.txt',
# '77.txt',
# '100.txt']
