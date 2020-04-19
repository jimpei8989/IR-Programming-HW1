import os
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.preprocessing import normalize

from Modules.helper import *

class VSM():
    def __init__(self, name, TF = None, IDF = None):
        self.name = name
        self.TF = None
        self.TF_CSC = None
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

    def Query(self, queryVectors, k3 = 0):
        #  scoreMatrix = ((k3 + 1) * queryVectors / (k3 + queryVectors)) @ self.score
        scoreMatrix = queryVectors @ self.score
        rankedDocuments = np.argsort(-scoreMatrix, axis = 1)[:, :100]
        return rankedDocuments, np.stack([s[i] for s, i in zip(scoreMatrix, rankedDocuments)])

    def Rank(self, queryVectors, k3 = 0, top = 100, threshold = 0):
        #  return [list(filter(lambda k : k[1] > threshold, zip(*pair)))[:top] for pair in zip(self.Query(queryVectors))]
        rankedDocuments, rankedScores = self.Query(queryVectors, k3 = k3)
        return rankedDocuments

    def CalculateCentroid(self, relevanceDocuments):
        if self.TF_CSC is None:
            self.TF_CSC = self.TF.tocsc()
        ret = [np.mean(np.stack(list(map(lambda k : self.TF_CSC.getcol(k).toarray().reshape(-1), rel))), axis = 0) for rel in relevanceDocuments]
        return normalize(np.stack(ret), axis = 1)


class OkapiBM25(VSM):
    def __init__(self, name, N = None, M = None, TF = None, IDF = None, calculate = False, k1 = 1.5, b = 0.75, calDocLen = True):
        def CalculateTF():
            data, rows, cols = TF

            if calDocLen:
                documentLength = np.zeros(M)
                docUsed = set()

                for d, c in zip(data, cols):
                    documentLength[c] += d
                    docUsed.add(c)

                with open('doc-len.npy', 'wb') as f:
                    np.save(f, documentLength)
            else:
                with open('doc-len.npy', 'rb') as f:
                    documentLength = np.load(f)

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

