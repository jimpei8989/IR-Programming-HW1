import os, sys, time
from Modules import helper, preprocessing
from Modules.helper import *
from Modules.TFIDF import OkapiBM25

def main():
    args = helper.getArguments()
    name = args.name[0]

    try:
        os.mkdir(args.c[0])
    except FileExistsError:
        pass

    # Read vocabularies & files
    with EventTimer(name = 'Get vocabularies and files'):
        # Indexed from 1
        ID2Voc, Voc2ID = helper.getVocabularies(os.path.join(args.m[0], 'vocab.all'))
        numOfVocabularies = len(ID2Voc)

        # Indexed from 0
        documentList = helper.getFileList(os.path.join(args.m[0], 'file-list'), args.d[0])
        idx2DocID = helper.getDocumentIDs(documentList)
        numOfDocuments = len(documentList)

        print(f'#Vocabularies:\t{numOfVocabularies}')
        print(f'#Files:\t\t{numOfDocuments}')

    # Handle Query File
    with EventTimer(name = 'Get query list'):
        queryList = getQueryList(args.i[0])

    # Handle Raw TF
    ### unigrams, bigrams: list of tuples of size 1 or 2
    ### RawTFs: sparse matrix of term frequency
    ### RawIDFs: np-arrays n(t)
    if args.preprocessing:  # Perform Preprocessing
        with EventTimer(name = 'Get Raw TF & IDF'):
            unigrams, bigrams, unigramRawTF, bigramRawTF, unigramRawIDF, bigramRawIDF = preprocessing.handleInvertedFile(
                    os.path.join(args.m[0], 'inverted-file'), numOfDocuments)
            SavePkl(unigrams, os.path.join(args.c[0], 'unigram.pkl'))
            SavePkl(bigrams, os.path.join(args.c[0], 'bigram.pkl'))
            SavePkl(unigramRawTF, os.path.join(args.c[0], 'unigram-raw-tf.pkl'))
            SavePkl(bigramRawTF, os.path.join(args.c[0], 'bigram-raw-tf.pkl'))
            SaveNPY(unigramRawIDF, os.path.join(args.c[0], 'unigram-raw-idf.npy'))
            SaveNPY(bigramRawIDF, os.path.join(args.c[0], 'bigram-raw-idf.npy'))
    else:   # Loading
        with EventTimer(name = 'Load Raw TF & IDF'):
            unigrams = LoadPkl(os.path.join(args.c[0], 'unigram.pkl'))
            bigrams = LoadPkl(os.path.join(args.c[0], 'bigram.pkl'))
            unigramRawTF = LoadPkl(os.path.join(args.c[0], 'unigram-raw-tf.pkl'))
            bigramRawTF = LoadPkl(os.path.join(args.c[0], 'bigram-raw-tf.pkl'))
            unigramRawIDF = LoadNPY(os.path.join(args.c[0], 'unigram-raw-idf.npy'))
            bigramRawIDF = LoadNPY(os.path.join(args.c[0], 'bigram-raw-idf.npy'))

    numOfUnigrams, numOfBigrams = len(unigrams), len(bigrams)

    terms = bigrams
    rawTF = bigramRawTF
    rawIDF = bigramRawIDF

    # Handle Normalized TF
    if args.tfidf:
        with EventTimer('Calculating OkapiBM25 - ' + name):
            vsm = OkapiBM25(name, numOfBigrams, numOfDocuments, rawTF, rawIDF, calculate = True, k1 = 2)
            vsm.Save(args.c[0])
    else:
        with EventTimer('Loading OkapiBM25 - ' + name):
            vsm = OkapiBM25(name)
            vsm.Load(args.c[0])

    with EventTimer('Transform query to vectors'):
        # Keys are: 'number', 'title', 'question', 'narrative', 'concepts'
        queryIDs, queryVectors = translateQuery(queryList, terms, Voc2ID,
                features = ['concepts'])

    with EventTimer('Finding relevance documents'):
        retrievedDocs = [[idx2DocID[i] for i in ans] for ans in vsm.Rank(queryVectors)]

        with open(args.o[0], 'w') as f:
            print('query_id,retrieved_docs', file = f)
            for QID, ANS in zip(queryIDs, retrievedDocs):
                print(QID + ',' + ' '.join(ANS), file = f)

if __name__ == '__main__':
    with EventTimer(name = 'MAIN'):
        main()

