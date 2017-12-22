#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
import numpy as np

 
def NMS(listOfBoxes, threshold):
    """    
    input is list of boxes and threshold
    each box consists of [x1,y1,x2,y2,score]
    output is list indexes of the boxes to keep
   
    """
 
    x1List, y1List, x2List, y2List, scoreList  = listOfBoxes[:, 0],\
    listOfBoxes[:, 1],\
    listOfBoxes[:, 2],\
    listOfBoxes[:, 3],\
    listOfBoxes[:, 4]
   
    areaList = (x2List - x1List + 1) * (y2List - y1List + 1)
    orderedScoreList = scoreList.argsort()[::-1]
   
    goodBoxesList = []
    while orderedScoreList.size > 0:
        #work with a box with the highiest score at a time
        #and remove other boxes that overlap with the current box
        print(orderedScoreList)
        indexOfBestArea = orderedScoreList[0]
        goodBoxesList.append(indexOfBestArea)
        #get lists of the maximum of either (current x1) or (all the other x1 except current)
        #and the same goes for other coordinates
        newX1List = np.maximum(x1List[indexOfBestArea], x1List[orderedScoreList[1:]])
        newY1List = np.maximum(y1List[indexOfBestArea], y1List[orderedScoreList[1:]])
        newX2List = np.minimum(x2List[indexOfBestArea], x2List[orderedScoreList[1:]])
        newY2List = np.minimum(y2List[indexOfBestArea], y2List[orderedScoreList[1:]])
       
        width = np.maximum(0.0, newX2List - newX1List + 1)
        height = np.maximum(0.0, newY2List - newY1List + 1)
        intersection = width * height
        #overlap of the current box with other boxes still ordered by scores
        overlap = intersection / (areaList[indexOfBestArea] + areaList[orderedScoreList[1:]] - intersection)
        indxes = np.where(overlap <= threshold)[0]
        #now keep in the ordered list only indexes of boxes that overlap
        #is lower than threshold, the "+1" is because "indexes" are only
        #boxes other than the current box
        orderedScoreList = orderedScoreList[indxes + 1]
 
 
    return goodBoxesList
if __name__ == '__main__':
    nmsTestBoxesNPARRAY = np.random.rand(7, 5)
    nmsTestBoxesNPARRAY[0] = [1,1,4,5,.7]
    nmsTestBoxesNPARRAY[1] = [3,4,6,4,.6]
    nmsTestBoxesNPARRAY[2] = [2,3,4,6,.9]
    nmsTestBoxesNPARRAY[3] = [6,6,10,10,.9]
    nmsTestBoxesNPARRAY[4] = [5,7,8,9,.4]
    nmsTestBoxesNPARRAY[5] = [8,10,10,10,.6]
    nmsTestBoxesNPARRAY[6] = [5,5,9,9,.8]
    nmsTestThreshold = 0.97
    ## data is 10 samples
    NMS(nmsTestBoxesNPARRAY, nmsTestThreshold)