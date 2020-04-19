import os, sys, time
import pickle
from argparse import ArgumentParser
import xml.etree.ElementTree as ET
import numpy as np
import scipy
from sklearn.preprocessing import normalize

_LightGray = '\x1b[38;5;251m'
_Bold = '\x1b[1m'
_Underline = '\x1b[4m'
_Orange = '\x1b[38;5;215m'
_SkyBlue = '\x1b[38;5;38m'
_Reset = '\x1b[0m'

class EventTimer():
    def __init__(self, name = '', description = ''):
        self.name = name
        self.description = description

    def __enter__(self):
        print(_LightGray + '------------------ Begin "' + _SkyBlue + _Bold + _Underline + self.name + _Reset + _LightGray +
                '" ------------------' + _Reset, file = sys.stderr)
        self.beginTimestamp = time.time()

    def __exit__(self, type, value, traceback):
        elapsedTime = time.time() - self.beginTimestamp
        print(_LightGray + '------------------ End   "' + _SkyBlue + _Bold + _Underline + self.name + _Reset + _LightGray +
                ' (Elapsed ' + _Orange + f'{elapsedTime:.4f}' + _Reset + 's)" ------------------' + _Reset + '\n', file = sys.stderr)

def getArguments():
    parser = ArgumentParser()
    parser.add_argument('-r', action = 'store_true',
                        help = 'If specified, turn on relevance feedback.')
    parser.add_argument('-c', type = str, nargs = 1, metavar = 'my-model-dir',
                        help = 'My model directory')
    parser.add_argument('-i', type = str, nargs = 1, metavar = 'query-file',
                        help = 'The input query file')
    parser.add_argument('-o', type = str, nargs = 1, metavar = 'ranked-list',
                        help = 'The output ranked list file')
    parser.add_argument('-m', type = str, nargs = 1, metavar = 'model-dir',
                        help = 'The input model directory')
    parser.add_argument('-d', type = str, nargs = 1, metavar = 'NTCIR-dir',
                        help = 'The directory of NTCIR documents.')
    parser.add_argument('--preprocessing', action = 'store_true',
                        help = 'Perform preprocessing')
    parser.add_argument('--tfidf', action = 'store_true',
                        help = 'Calculate TFIDF')
    parser.add_argument('--model-name', type = str, nargs = 1, metavar = 'MODEL-NAME', dest = 'name',
                        help = 'The name used in TF IDF models')
    return parser.parse_args()

def SavePkl(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
def LoadPkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
def SaveSPM(matrix, path):
    scipy.sparse.save_npz(path, matrix)
def LoadSPM(path):
    return scipy.sparse.load_npz(path)
def SaveNPY(array, path):
    np.save(path, array)
def LoadNPY(path):
    return np.load(path)

def getVocabularies(vocFilePath):
    with open(vocFilePath, 'rb') as f:
        encoding = f.readline().strip().decode()
        rawLines = f.readlines()

        ID2Voc = {i + 1 : v for i, v in enumerate(map(lambda s : s.strip().decode(encoding), rawLines))}
        Voc2ID = {v : k for k, v in ID2Voc.items()}

    return ID2Voc, Voc2ID

def getFileList(fileListPath, docDirectory):
    with open(fileListPath, 'r') as f:
        return list(map(lambda l : os.path.join(docDirectory, l.strip()), f.readlines()))

def getDocumentIDs(docList):
    def getID(doc):
        root = ET.parse(doc).getroot().find('doc')
        return root.find('id').text
    return list(map(getID, docList))

def getQueryList(queryFilePath):
    # Returns a document of (ID, title, question, narrative, concept)
    tree = ET.parse(queryFilePath)
    root = tree.getroot()
    queries = [{ele.tag : ele.text.strip() for ele in topic} for topic in root]

    for q in queries:
        q['concepts'] = q['concepts'].split('、')

    return queries

def translateQuery(queryList, terms, Voc2ID, features = [], weights = None):
    term2ID = {t : i for i, t in enumerate(terms)}

    def str2Tokens(s):
        return [Voc2ID[c] if c in Voc2ID else 0 for c in s]

    def tokens2Terms(l):
        ret = []
        for u in l:
            if u in term2ID:
                ret.append(term2ID[u])
        for b in zip(l[:-1], l[1:]):
            if b in term2ID:
                ret.append(term2ID[b])
        return ret

    def trans(Q):
        ret = np.zeros(len(terms))
        for q in Q:
            ret[q] += 1
        return ret

    if weights is None:
        weights = [1] * len(features)

    assert len(features) == len(weights), "The weights length should be the same as features"

    queryIDs = []
    queryVectors = []

    for query in queryList:
        vec = np.zeros(len(terms))
        for f, w in zip(features, weights):
            if f == 'concepts':
                for c in query[f]:
                    Q = tokens2Terms(str2Tokens(c))
                    vec += trans(Q) * w
            else:
                Q = tokens2Terms(str2Tokens(query[f]))
                vec += trans(Q) * w

        queryIDs.append(query['number'][-3:])
        queryVectors.append(vec)

    return queryIDs, normalize(np.stack(queryVectors), axis = 1, copy = False)

