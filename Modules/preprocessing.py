import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

from Modules.helper import EventTimer

def handleInvertedFile(invertedFilePath, numOfDocuments):
    unigrams, bigrams = list(), list()
    unigramIdx, bigramIdx = 0, 0

    uniData, uniRows, uniCols = list(), list(), list()
    biData, biRows, biCols = list(), list(), list()
    unigramIDF, bigramIDF = list(), list()

    with open(invertedFilePath, 'r') as f:
        while True:
            try:
                x, y, M = map(int, f.readline().split())
                occurences = [tuple(map(int, f.readline().split())) for _ in range(M)]
            except ValueError:
                break

            if y == -1: # is unigram
                unigrams.append((x,))

                uniData += [c for d, c in occurences]
                uniRows += [unigramIdx] * M
                uniCols += [d for d, c in occurences]

                unigramIDF.append(M)
                unigramIdx += 1
            else:       # is bigram
                bigrams.append((x, y))

                biData += [c for d, c in occurences]
                biRows += [bigramIdx] * M
                biCols += [d for d, c in occurences]

                bigramIDF.append(M)
                bigramIdx += 1

    numOfUnigrams = len(unigrams)
    numOfBigrams = len(bigrams)

    print(numOfUnigrams)
    print(numOfBigrams)

    unigramTF = (uniData, uniRows, uniCols)
    #  unigramTF = csr_matrix((uniData, (uniRows, uniCols)), shape = (numOfUnigrams, numOfDocuments))
    #  unigramDocLength = np.asarray(unigramTF.sum(axis = 0)).reshape(-1)

    bigramTF = (biData, biRows, biCols)
    #  bigramTF = csr_matrix((biData, (biRows, biCols)), shape = (numOfBigrams, numOfDocuments))
    #  bigramDocLength = np.asarray(bigramTF.sum(axis = 0)).reshape(-1)
    
    return unigrams, bigrams, unigramTF, bigramTF, np.array(unigramIDF), np.array(bigramIDF)

