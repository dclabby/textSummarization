#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 16:13:15 2021

@author: dclabby
"""
import numpy as np

def generateSummary(sentenceScores, nSentences=3):
    summary = []
    for doc in sentenceScores:
        #avgScore = doc['sentenceScore'].mean()
        #summaryData = doc[doc['sentenceScore'] > 1.5*avgScore]
        #n = 3#len(doc)//5
        summaryData = doc.sort_values(by=['sentenceScore'], ascending = False)
        summaryData = summaryData.head(nSentences)
        summaryData = summaryData.sort_index()
        
        summaryTmp = ""
        for sentence in summaryData['rawSentence']:
            summaryTmp += sentence
        summary.append(summaryTmp)
        #summary.append(summaryData)
    return summary

def generateClusterSummaries(clusterCentreScores, nSentences=3):
    docSummary = []
    for docCluster in clusterCentreScores:
        nClusters = len(docCluster)
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