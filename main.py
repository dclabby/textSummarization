#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 16:23:29 2021

@author: dclabby
"""
import pandas as pd
import numpy as np
import os
import time

import preProc as p
import analyseTransform as a
import genSummary as s
import evaluate as e

# 0.1 Load data
folderName = "./data"#"/home/dclabby/Documents/Springboard/HDAIML_SEP/Semester02/ArtIntel/Project/TextSummCode/data"#
folderContents = os.listdir(folderName)
nFiles = 10
files = folderContents[0:nFiles]
dmData = p.load_DailyMail_Data(files, folderName)
cleanWords, rawSentences = p.cleanCorpus(dmData["Text"])
ref = dmData["Summary"]
N = np.round(dmData["nLinesSummary"].mean()).astype('int').item()
k = N

# 0.2 Generate Sentence embeddings
#!pip install sentence-transformers
from sentence_transformers import SentenceTransformer
sbert_model = SentenceTransformer('stsb-roberta-base') # optimized for Semantic Textual Similarity (STS). 
print("Generating Sentence Embeddings...")
t1 = time.time()
docVecs =  a.generateSentenceVecs(rawSentences, sbert_model)
tEmbedding = time.time() - t1
print("Sentence Embeddings completed in " + str(tEmbedding) + "s")

# 1.0 Initialize variables for evaluation
rogueScore = pd.DataFrame()
execTime = []
wordCountRatio = []
rLength = np.array([len(d.split(" ")) for d in ref])

## 1.1 TF-IDF
print("Calculating TF-IDF...")
t1 = time.time()
tfidf, tf, idf = a.calcTFIDFcoeffs(cleanWords)
sentenceScores = a.calcScoresTFIDF(cleanWords, tfidf, rawSentences)
tfIdfSummaries = s.generateSummary(sentenceScores, N)
rogueScore["TF-IDF"] = e.calcRogue(tfIdfSummaries, ref)
execTime.append(time.time() - t1)
wordCountRatio.append(rLength/np.array([len(d.split(" ")) for d in tfIdfSummaries]))

## 1.2 TextRank
print("Calculating TextRank...")
t1 = time.time()
trSimilarity = a.sentenceSimilarity(cleanWords)#, sentenceTokens)
textRankScores = a.calcScoresTextRank(trSimilarity, rawSentences)
textRankSummaries = s.generateSummary(textRankScores, N)
rogueScore["TextRank"] = e.calcRogue(textRankSummaries, ref)
execTime.append(time.time() - t1)
wordCountRatio.append(rLength/np.array([len(d.split(" ")) for d in textRankSummaries]))


## 1.3 CosRank
print("Calculating CosRank...")
t1 = time.time()
#docVecs =  a.generateSentenceVecs(rawSentences, sbert_model)
cosSimMat = a.sentenceEmbeddingSimilarity(docVecs)
cosRankScores = a.calcScoresTextRank(cosSimMat, rawSentences) 
cosRankSummaries = s.generateSummary(cosRankScores, N)
rogueScore["CosRank"] = e.calcRogue(cosRankSummaries, ref)
execTime.append(time.time() - t1 + tEmbedding)
wordCountRatio.append(rLength/np.array([len(d.split(" ")) for d in cosRankSummaries]))

## 1.4 Cluster Center
print("Calculating Cluster Centres...")
t1 = time.time()
#docVecs =  a.generateSentenceVecs(rawSentences, sbert_model)
np.random.seed(0)
clusterCentres = a.calcClusterCentres(docVecs, rawSentences, k)
clusterCentreSummaries = s.generateClusterSummaries(clusterCentres, N)
rogueScore["ClusterCentre"] = e.calcRogue(clusterCentreSummaries, ref)
execTime.append(time.time() - t1 + tEmbedding)
wordCountRatio.append(rLength/np.array([len(d.split(" ")) for d in clusterCentreSummaries]))

## 1.5 Cluster Rank
print("Calculating ClusterRank...")
t1 = time.time()
#trSimilarity = sentenceSimilarity(cleanWords)#, sentenceTokens)
#textRankScores = calcScoresTextRank(trSimilarity, rawSentences)
#docVecs =  a.generateSentenceVecs(rawSentences, sbert_model)
np.random.seed(0)
clusterCentres = a.calcClusterCentres(docVecs, rawSentences, k)
#clusterRank = calcClusterRank(textRankScores, clusterCentres)
clusterRank = a.calcClusterRank(cosRankScores, clusterCentres)
clusterRankSummaries = s.generateClusterSummaries(clusterRank, N)

rogueScore["ClusterRank"] = e.calcRogue(clusterRankSummaries, ref)
execTime.append(time.time() - t1 + tEmbedding)
#wordCountRatio.append(np.mean(rLength/np.array([len(d.split(" ")) for d in clusterRankSummaries])))
wordCountRatio.append(rLength/np.array([len(d.split(" ")) for d in clusterRankSummaries]))

## 2. Post Processing
def printSummaries(iFile):
    print("\nReference Summary:\n" + dmData["Summary"][iFile] + "\n")
    print("\nTF-IDF: Rogue1 = " + str(rogueScore["TF-IDF"][iFile]) +"\n" + tfIdfSummaries[iFile] + "\n")
    print("\nTextRank: Rogue1 = " + str(rogueScore["TextRank"][iFile]) +"\n" + textRankSummaries[iFile] + "\n")
    print("\nCosRank: Rogue1 = " + str(rogueScore["CosRank"][iFile]) +"\n" + cosRankSummaries[iFile] + "\n")
    print("\nClusterCenter: Rogue1 = " + str(rogueScore["ClusterCentre"][iFile]) +"\n" + clusterCentreSummaries[iFile] + "\n")
    print("\nClusterRank: Rogue1 = " + str(rogueScore["ClusterRank"][iFile]) +"\n" + clusterRankSummaries[iFile] + "\n")