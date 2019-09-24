import sys, os, csv
import glob
import random
import re
import pickle
from collections import Counter
import numpy as np

import nltk
from nltk.corpus import stopwords 
import string
from pattern3.en import lemma


class DocReader():
    def __init__(self):
        pass

    def atoi(self,text):
	    return int(text) if text.isdigit() else text

    def natural_keys(self,text):
        '''
        alist.sort(key=natural_keys) sorts in human order
        http://nedbatchelder.com/blog/200712/human_sorting.html
        '''
        return [ self.atoi(c) for c in re.split('(\d+)', text) ]


    def create_bag_of_words(self,filePaths):
        '''
        Input:
          filePaths: Array. A list of absolute filepaths
        Returns:
          bagOfWords: Array. All tokens in files
        '''
        bagOfWords = []
        
        for filePath in filePaths:
            with open(filePath, encoding="utf-8", errors='ignore') as f:
                raw = f.read()
                tokens = raw.split()
                for token in tokens:
                    bagOfWords.append(token)
        return bagOfWords

    def remove_stop_words(self, bagOfWords):
        nltk.download('stopwords')
        stop_word = set(stopwords.words('english'))
        stop_punc = string.punctuation
        for w in stop_word:
            try: bagOfWords.remove(w)
            except: pass
        for p in stop_punc:
            try: bagOfWords.remove(p)
            except: pass
        return bagOfWords

    def lemmatization(self, bagOfWords):
        bagOfWords = [lemma(w) for w in bagOfWords]
        return bagOfWords

    def get_feature_matrix(self, filePaths, featureDict):
        '''
        create feature/x matrix from multiple text files
        rows = files, cols = features
        '''
        featureMatrix = np.zeros(shape=(len(filePaths),
                                          len(featureDict)),
                                   dtype=float)
        
        for i,filePath in enumerate(filePaths):
            with open(filePath, encoding="utf-8", errors='ignore') as f:
                raw = f.read()
                tokens = raw.split()
                fileUniDist = Counter(tokens)
                for key,value in fileUniDist.items():
                    if key in featureDict:
                        featureMatrix[i,featureDict[key]] = value
        return featureMatrix

    def regularize_vectors(self,featureMatrix):
        '''
        Input:
          featureMatrix: matrix, where docs are rows and features are columns
        Returns:
          featureMatrix: matrix, updated by dividing each feature value by the total
          number of features for a given document
        '''
        for doc in range(featureMatrix.shape[0]):
            totalWords = np.sum(featureMatrix[doc,:],axis=0)
            featureMatrix[doc,:] = np.multiply(featureMatrix[doc,:],(1/(totalWords + 1e-5)))
        return featureMatrix

    def input_data(self,datadir,percentTest,cutoff):
        files = os.listdir(datadir)
        abs_files=[]
        for file in files:  # all the .eml files
            if 'TRAIN' in file : 
                abs_files.append(os.path.join(datadir, file))
        abs_files.sort(key=self.natural_keys)
        # get test set as random subsample of all data
        numTest = int(percentTest * len(files))
        test_data = abs_files[:numTest]
        print(len(test_data))

        # delete testing data from superset of all data
        train_data = abs_files[numTest:]
        print(len(train_data))

        # create feature dictionary of n-grams
        bagOfWords = self.create_bag_of_words(train_data)
        bagOfWords = self.remove_stop_words(bagOfWords)
        bagOfWords = self.lemmatization(bagOfWords)

        # throw out low freq words
        freqDist = Counter(bagOfWords)  # 'word': fq
        newBagOfWords=[]
        for word,freq in freqDist.items(): 
            if freq > cutoff:
                newBagOfWords.append(word)
        pickle.dump(newBagOfWords, open('BagOfWords.p', 'wb'))
        features = set(newBagOfWords)
        featureDict = {feature:i for i,feature in enumerate(features)}

        # make feature matrices
        trainX = self.get_feature_matrix(train_data,featureDict)
        testX = self.get_feature_matrix(test_data,featureDict)
        # print(trainX)

        # regularize length
        trainX = self.regularize_vectors(trainX)
        testX = self.regularize_vectors(testX)

        # add label to each row, 0 for spam, 1 for ham
        with open("spam-mail.csv") as f:
            spamlabel = csv.reader(f)
            Y = np.array([[int(row[1]) for row in spamlabel]])

        trainX = np.concatenate((trainX, Y[:,250:].T), axis=1)
        testX = np.concatenate((testX, Y[:,:250].T), axis=1)

        return trainX, testX


if __name__ == '__main__':
    print('Input source directory: ') #ask for source    
    datadir = input()
    reader = DocReader()
    trainX, testX = reader.input_data(datadir=datadir,
                                      percentTest=.1,
                                      cutoff=20)
    numTest = testX.shape[0]
    print(trainX.shape)  
    print(testX.shape)
    
    np.savetxt("trainX.csv", trainX, delimiter=",", fmt='%f')
    np.savetxt("testX.csv", testX, delimiter=",", fmt='%f')
