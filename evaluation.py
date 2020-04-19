import sys
import numpy as np
from Modules.helper import EventTimer

def MAP(T, R):
    def AP(T, R):
        precisions = []
        cnt = 0
        for i, d in enumerate(R):
            if d in T:
                cnt += 1
                precisions.append(cnt / (i + 1))
        return sum(precisions) / len(T)

    APs = [AP(a, b) for a, b in zip(T, R)]
    return np.mean(APs)

def main():
    def getRetrievedDocs(path):
        with open(path) as f:
            lines = f.readlines()[1:]
            return [line.split(',')[1].split(' ') for line in lines]

    answerFile, predictionFile = sys.argv[1], sys.argv[2]
    with EventTimer('Calculating MAP') as f:
        groundTruth = getRetrievedDocs(answerFile)
        rankedList = getRetrievedDocs(predictionFile)

        print('MAP:', MAP(groundTruth, rankedList))

if __name__ == '__main__':
    main()

