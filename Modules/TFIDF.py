import os
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.preprocessing import normalize

from Modules.helper import *

class VSM():
    def __init__(self, name, TF = None, IDF = None):
        self.name = name
        self.TF = None
        self.IDF = None
        self.score = None
        pass

    def Save(self, modelDir):
        directory = os.path.join(modelDir, self.name)

        try:
            os.mkdir(directory)
        except FileExistsError:
            pass

        SaveSPM(self.TF, os.path.join(directory, 'tf.npz'))
        SaveNPY(self.IDF, os.path.join(directory, 'idf.npy'))
        SaveSPM(self.score, os.path.join(directory, 'score.npz'))

    def Load(self, modelDir):
        directory = os.path.join(modelDir, self.name)
        self.TF = LoadSPM(os.path.join(directory, 'tf.npz'))
        self.IDF = LoadNPY(os.path.join(directory, 'idf.npy'))
        self.score = LoadSPM(os.path.join(directory, 'score.npz'))

    def Query(self, queryVectors):
        ret = []
        for qvec in queryVectors:
            qvec = normalize(qvec.reshape((-1, 1)), axis = 0)
            scores = np.array(self.score.multiply(qvec).sum(axis = 0)).reshape(-1)
            docs = np.argsort(scores)[-100:][::-1]
            pairs = list(zip(docs, scores[docs]))
            ret.append(pairs)
        return ret

    def Normalize(self):
        raise NotImplementedError

class OkapiBM25(VSM):
    def __init__(self, name, N = None, M = None, TF = None, IDF = None, calculate = False, k1 = 1.5, b = 0.75):
        def CalculateTF():
            data, rows, cols = TF
            documentLength = np.zeros(M)

            docUsed = set()

            for d, c in zip(data, cols):
                documentLength[c] += d
                docUsed.add(c)

            avgDocLength = np.mean(documentLength)

            newData = [d * (k1 + 1) / (d + k1 * (1 - b + b * documentLength[c] / avgDocLength)) for d, c in zip(data, cols)]
            return csr_matrix((newData, (rows, cols)), shape = (N, M))

        def CalculateIDF():
            return np.clip(np.log((M - IDF + 0.5) / (IDF + 0.5)).reshape((-1, 1)), 0, None)

        super().__init__(name, TF, IDF)

        if calculate:
            self.TF = CalculateTF()
            self.IDF = CalculateIDF()
            self.score = self.TF.multiply(self.IDF)
            self.score = normalize(self.score, axis = 0, copy = False)

    def Rank(self, queryVectors):
        answers = self.Query(queryVectors)
        return [[d for d, s in ans[:25]] for ans in answers]

