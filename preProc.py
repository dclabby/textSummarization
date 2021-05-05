#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 16:07:20 2021

@author: dclabby
"""
import nltk
import pandas as pd
import string

def load_DailyMail_Data(fileList, folderName=""):
    summary = []
    text = []
    nLinesSummary = []
    for iFile in fileList:
        
        tmpFile = open(folderName + "/" + iFile, 'r')
        rawText = tmpFile.readlines()
        tmpFile.close()
        
        strippedText = ""
        for t in rawText:
            strippedText += t
        
        strippedText = strippedText.split("@highlight") #separate highlights from the main article by splitting on "@highlight"
        nLinesSummary.append(len(strippedText) - 1)
        textTmp = strippedText[0] # article is the first element in the split text
        summaryTmp = "" # initialize string for the summary
        for h in strippedText[1:]: # iterate through the highlights (all elements of the split text other than the first)
            summaryTmp += h + ". " # append highlights to the summary string, & add full stop and space between highlights
        
        summary.append(summaryTmp)
        text.append(textTmp)
    
    df = pd.DataFrame({"Source": fileList, "Summary":summary, "Text": text, "nLinesSummary": nLinesSummary})
    return df

def cleanCorpus(textSeries, removeStopWords=True):
    cleanText = []
    rawText = []
    for text in textSeries:
        cleanList, rawList = cleanDocument(text, removeStopWords)
        cleanText.append(cleanList)
        rawText.append(rawList)
    return cleanText, rawText

def cleanDocument(rawText, removeStopWords=True):
    """
    will return:
    - clean list: [ [sentence], [[word], [word]] ]
    - raw list: [[sentence], [sentence]]
    """
    
    rawText = rawText.replace("\n", " .").replace("\xa0", " .") # replace end of line tage with " ."
    rawText = nltk.tokenize.sent_tokenize(rawText)
        
    tr = str.maketrans("", "", string.punctuation)
    remove_digits = str.maketrans('', '', string.digits)
    stop_words = nltk.corpus.stopwords.words("english")
    
    cleanList = []
    rawList = []
    for sentence in rawText:
        sentence = sentence.lstrip(".")
        cleanSentence = sentence.translate(tr) # remove punctuation from string
        cleanSentence = cleanSentence.translate(remove_digits) # remove numbers
        cleanSentence = cleanSentence.lower() # convert all characters to lower
        if removeStopWords:
            cleanSentence = [s for s in nltk.tokenize.word_tokenize(cleanSentence) if s not in stop_words] # remove stop words & split sentence into words
        else:            
            cleanSentence = [s for s in nltk.tokenize.word_tokenize(cleanSentence)] 
        if len(cleanSentence):
            cleanList.append(cleanSentence)
            rawList.append(sentence)
    
    return cleanList, rawList

def printLines(rawSentences, iFile):
    for i, s in enumerate(rawSentences[iFile]):
        print("\nSentence " + str(i) + ": " + str(s) + "\n")