#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 20:17:25 2021

@author: dclabby
"""
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

def exportStats(rogueScore, execTime, wordCountRatio, destFile):
    df2 = pd.DataFrame([execTime, np.mean(wordCountRatio,axis=1)], columns=[c for c in rogueScore])
    summaryDf = pd.concat([rogueScore, df2])
    summaryDf.to_excel(destFile)

def makeBoxPlot(rogueScore, title=""):
    data = rogueScore.values
    barNames = list(rogueScore.columns)
    plt.figure()
    plt.boxplot(np.array( data ))
    plt.ylim(-0.1,1.1)
    plt.xticks([i+1 for i in range(0,len(barNames))], barNames)
    plt.ylabel('Rogue score')
    plt.title(title)
