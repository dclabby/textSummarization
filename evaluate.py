#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 16:22:11 2021

@author: dclabby
"""
import preProc as p

def calcRogue(candidate, reference, removeStopWords=True):
    rogueScores = []
    for c, r in zip(candidate, reference):
        candidateClean, candidateRaw = p.cleanDocument(c, removeStopWords)
        referenceClean, referenceRaw = p.cleanDocument(r, removeStopWords)

        candidateSet = set(sum(candidateClean, []))
        referenceSet = set(sum(referenceClean, []))
        rogueScores.append(len(candidateSet.intersection(referenceSet))/len(referenceSet))
    return rogueScores

