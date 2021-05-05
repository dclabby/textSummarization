#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 16:10:31 2021

@author: dclabby
"""
import pandas as pd
import numpy as np

# 1. TF-IDF
def calcTFIDFcoeffs(wordTokens):
    termCounts = pd.DataFrame()
    nDocs = len(wordTokens)
    for iDoc in range(0, nDocs):
        for sentence in wordTokens[iDoc]:
            for word in sentence:
                try:
                    termCounts[word][iDoc] += 1
                except:
                    termCounts[word] = pd.Series(np.zeros(nDocs).astype(int))
                    termCounts[word][iDoc] += 1

    nTerms = len(termCounts.columns)

    C_i = termCounts.sum(axis=1)
    B_j = termCounts.astype(bool).sum(axis=0)

    TF =  np.array(termCounts)/ np.transpose(np.tile(C_i, (nTerms, 1)))
    IDF = np.log(nDocs/np.array(B_j))
    TFIDF = TF*np.tile(IDF, (nDocs, 1))

    colNames = [c for c in termCounts]
    TF = pd.DataFrame(TF, columns = colNames)
    IDF = pd.Series(IDF, index = colNames)
    TFIDF = pd.DataFrame(TFIDF, columns = colNames)
    return TFIDF, TF, IDF

def calcScoresTFIDF(wordTokens, TFIDF, sentenceTokens, docIndices=[]):
    
    if len(docIndices) == 0:
        docIndices = [x for x in range(0,len(wordTokens))]
    
    scores = []
    for iDoc in docIndices:
        doc = wordTokens[iDoc]    
        docScores = []
        for sentence in doc:
            sentenceScore = 0
            
            #if len(sentence) == 0:
            #    print(iDoc)
            
            #for word in sentence:
            #    sentenceScore += TFIDF[word][iDoc]
            #sentenceScore /= len(sentence)
            if len(sentence) > 0:
                for word in sentence:
                    sentenceScore += TFIDF[word][iDoc]
                sentenceScore /= len(sentence)
            docScores.append(sentenceScore)
        dfTmp = pd.DataFrame({"rawSentence": sentenceTokens[iDoc], "sentenceScore": docScores})
        scores.append(dfTmp)
        
        #scores.append(docScores)
    
    return scores   

# 2. TextRank
def sentenceSimilarity(wordTokens, docIndices=[]):#, sentenceTokens, docIndices=[]):
    
    if len(docIndices) == 0:
        docIndices = [x for x in range(0,len(wordTokens))]
        
    simMatrices = []
    for iDoc in docIndices:
        doc = wordTokens[iDoc]    
        nSentences = len(doc)
        denom = np.ones((nSentences, nSentences))
        docSim = np.zeros((nSentences, nSentences))
        for i, sentence_i in enumerate(doc):
            #n_i = len(set(sentence_i)) # number of unique words in ith sentence
            n_i = len(sentence_i) # number of words in ith sentence
            for j, sentence_j in enumerate(doc):                
                #n_j = len(set(sentence_j)) # number of unique words in jth sentence
                n_j = len(sentence_j) # number of words in jth sentence
                #if i > j: # only need to calculate lower triangular matrix since edges undirected and matrix is symmetrical (similarity i to j = similarity j to i)
                if i != j:
                    #if n_i > 0 and n_j > 0:
                    denom[i, j] = np.log(n_i) + np.log(n_j)
                    if denom[i, j] > 0:
                        docSim[i, j] = len(set(sentence_i).intersection(set(sentence_j)))/denom[i, j]
        simMatrices.append(docSim)
    return simMatrices   

def calcScoresTextRank(similarityMatrix, rawSentences, d=0.85, initialScores=1, tol = 0.0001, docIndices=[]):
    
    textRankScores = []
    for iDoc, simMat in enumerate(similarityMatrix):
        simMat = np.abs(simMat) #NOTE: cosine similarity has a range -1 to 1 whereas text rank expects similarity in the range 0 to 1
        nSentences = np.shape(simMat)[0]
        sentenceScores = np.ones((nSentences))*initialScores
        prev = np.zeros((nSentences))

        #N = 30
        #error = []
        #for n in range(0,N):
        error = 1
        while error > tol:
            for i, score_i in enumerate(sentenceScores):
                tmp = 0
                for j, score_j in enumerate(sentenceScores):
                    w_ji = simMat[j, i]
                    w_jk = np.sum(simMat[j])
                    #if w_jk > 0:
                    if w_jk != 0:
                        tmp += w_ji*score_j/w_jk

                sentenceScores[i] = (1-d) + d*tmp
            #error.append(np.sum(np.abs(sentenceScores[i] - prev)))
            error = np.sum(np.abs(sentenceScores[i] - prev))
            prev = sentenceScores[i]
        #if len(rawSentences) != len(sentenceScores):
        #    print("sentence length: " + str(len(rawSentences)) + "; \nscore length: " + str(len(sentenceScores)) + "\n")
        #    return []
        df = pd.DataFrame({"rawSentence": rawSentences[iDoc], "sentenceScore": sentenceScores})
        #return df
    
        textRankScores.append(df)
    return textRankScores

# 3. Sentence Embeddings
def cosineSimilarity(u, v):
    return (np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))).item()


def generateSentenceVecs(sentenceTokens, sbert_model, docIndices=[]):
    if len(docIndices) == 0:
        docIndices = [x for x in range(0,len(sentenceTokens))]
        
    sentenceVectors = []
    for iDoc in docIndices:
        doc = sentenceTokens[iDoc]
        sentenceVectors.append(sbert_model.encode(doc))
    return sentenceVectors


#def sentenceEmbeddingSimilarity(sentenceTokens, sbert_model, docIndices=[]):
def sentenceEmbeddingSimilarity(docVecs, docIndices=[]):
    
    if len(docIndices) == 0:
        docIndices = [x for x in range(0,len(docVecs))]
        
    simMatrices = []
    for iDoc in docIndices:
    #    doc = sentenceTokens[iDoc]
    #    sentenceVectors = sbert_model.encode(doc)
        sentenceVectors = docVecs[iDoc]
                
        #nSentences = len(doc)
        nSentences = len(sentenceVectors)
        docSim = np.zeros((nSentences, nSentences))
        for i, vec_i in enumerate(sentenceVectors):
            for j, vec_j in enumerate(sentenceVectors):
                if i != j:
                    docSim[i, j] = cosineSimilarity(vec_i, vec_j)
        
        simMatrices.append(docSim)
    return simMatrices   

# 4. Unsupervised Classification
def knn(data, k):
    """
    data is an array with:
    - dimension 0 (rows) containing sample points;
    - dimension 1 (columns) containing dimensions
    - example: data of size n x d contains n data points, each of dimension d
    - for a given document, this will take sentence vectors where n is the number of sentences & d is the size of the vector
    """
    nSamples = np.shape(data)[0]
    D = np.shape(data)[1]
    
    #assign random starting positions
    i_z = np.random.choice(nSamples, k, replace=False)
    z_k = data[i_z, :]
    z_initial = np.array(z_k, copy=True)
    
    # iteratively assign data to nearest z
    continueLoop = True # initialize variable to terminate while loop
    prevClusterNos = np.zeros(nSamples) # initialize array of zeros to store previous cluster assignments
    n = 0 # initialize loop counter
    maxIts = 2*nSamples # define maximum iterations to prevent infinite loop
    while continueLoop and n < maxIts:    
        n += 1 # increment the loop counter
        cost = np.zeros([nSamples, k]) # initialize the cost matrix
        for i, z in enumerate(z_k): # iterate through each cluster...
            cost[:, i] = np.linalg.norm(data - z_k[i],axis=1) # calculate the distance between each data point and the ith cluster centre
        clusterNos = np.argmin(cost,axis=1) # identify the closest cluster to each data point
        clusterData = []
        for i_k in range(0,k): # iterate through each cluster
            clusterData.append(data[clusterNos == i_k, :]) # extract the data for the ith cluster
            z_k[i_k, :] =  np.mean(clusterData[-1], axis=0) # update the centre of the cluster based on the average of its data
        continueLoop = not np.all(clusterNos == prevClusterNos) # continue the loop if the present and previous cluster assignments differ
        prevClusterNos = clusterNos # update previous cluster numbers (for the next iteration) based on this iteration's cluster numbers
        
    return clusterData, clusterNos

def calcClusterCentres(docVecs, rawSentences, k):

    clusterCentreScores = []
    for sentenceVecs, sentencesRaw in zip(docVecs, rawSentences):
        sentenceClusters, clusterNos = knn(sentenceVecs, k)
        clusterDicts = []
        for i_c, cluster in enumerate(sentenceClusters):
            clusterMean = np.mean(cluster,axis=0)
            clusterScores = []
            for u in cluster:
                clusterScores.append(cosineSimilarity(u, clusterMean)) # cosine similarity is not sensitive to sentence size
                #clusterScores.append(np.linalg.norm(u - clusterMean)) 

            clusterSentences = []
            for s, c in zip(sentencesRaw, clusterNos):
                if c == i_c:
                    clusterSentences.append(s)

            clusterDicts.append(pd.DataFrame({"rawSentence": clusterSentences, "sentenceScore": clusterScores}))
        clusterCentreScores.append(clusterDicts)
    return clusterCentreScores

def generateClusterSummaries(clusterCentreScores, nSentences=3):
    docSummary = []
    for docCluster in clusterCentreScores:
        #nClusters = len(docCluster)
        sentencesPerCluster = [len(s) for s in docCluster]
        sentencesPerCluster = np.round((sentencesPerCluster/np.sum(sentencesPerCluster))*nSentences).astype('int')
        #summary = []
        summaryTmp = ""
        for clusterData, nC in zip(docCluster, sentencesPerCluster):
            if nC > 0:
                                
                summaryData = clusterData.sort_values(by=['sentenceScore'], ascending = False)
                summaryData = summaryData.head(nC)
                summaryData = summaryData.sort_index()

                #summaryTmp = ""
                for sentence in summaryData['rawSentence']:
                    summaryTmp += sentence
            #summary.append(summaryTmp)
                
        docSummary.append(summaryTmp)
        #docSummary.append(summary)
    return docSummary

def calcClusterRank(textRankScores, clusterCentres):
    clusterRank = []
    for tr, cc in zip(textRankScores, clusterCentres):
        clusterRankTmp = []
        for c in cc:
            clusterRankTmp.append(pd.merge(tr, c, on="rawSentence").rename(columns={"sentenceScore_x": "sentenceScore"}))
        clusterRank.append(clusterRankTmp)
    return clusterRank