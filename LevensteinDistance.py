#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 19:23:32 2018

@author: Karina
"""

def levenshteinDistance(temp, temp2):
    if len(temp) > len(temp2):
        temp = temp2
        temp2 = temp
    
    dist = range(len(temp) + 1)
    
    for ind, ind2 in enumerate(temp2):
        temp_dist = [ind+1]
        for iind, iind2 in enumerate(temp):
            if iind2 != ind2:
                temp_dist.append(1 + min((dist[iind], \
                                dist[iind + 1], temp_dist[-1])))
            else:
                temp_dist.append(dist[iind])
           
        dist = temp_dist

    return dist[-1]

