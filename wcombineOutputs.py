import sys
import csv
import math
import numpy as np
from scipy import sparse
import pandas as pd
from scipy.sparse import lil_matrix
from surprise import NMF
from surprise import SVD
from surprise import SVDpp
from surprise import Dataset
from surprise import evaluate, print_perf
from surprise import Reader

numTests = 158024
numFiles = 0

scores = {"ALSScaled.csv": 1.44151,
          "CC100E2.csv": 1.47016,
          "CC100E.csv": 1.46421,
          "coclustering.csv": 1.46340,
          "defaultSVD.csv": 1.31436,
          "NMF.csv": 1.52098,
          "SVD50F001LR.csv": 1.31623,
          "SVD50Factors.csv": 1.31023,
          "SVDpp50F30E001LR.csv": 1.34448,
          "SVDpp50F50E005LR.csv": 1.34327,
          "SVDppDefault.csv": 1.31466
         }

def calcWeights(sDict):
    weights = sDict
    minScore = min(sDict.values())
    denom = max(sDict.values()) - minScore

    for f in sDict:
        score = sDict[f]
        weight = 1 - (score - minScore)/denom
        weights[f] = weight
    return weights


# Read and process the data
def readData(filename):
	dataset = []
	with open(filename) as csvDataFile:
		csvReader = csv.reader(csvDataFile)
		for row in csvReader:
			dataset.append(row)

	dataset = dataset[1:]
	# Transform numbers from string to floats
	for i in range(len(dataset)):
		for j in range(len(dataset[i])) :
			dataset[i][j] = float(dataset[i][j])

	return dataset


# Main program	    		
if __name__ == '__main__':
	weights = calcWeights(scores)
	scoreFiles = (sys.argv[1:])
	weightSum = sum(weights.values())

	# Create a dictionary for all the files with their test results
	dataDict = {}
	for f in scoreFiles :
		dataDict[f] = readData(f)

	# Fill the final matrix with 0s
	finalOut = []
	for i in range(numTests) :
		finalOut.append(0)

	# Add up all the test ratings
	for fName, dataM in dataDict.items():
		i = 0
		for elem in dataM :
			#print("for file %s the weight is %f" % (fName,weights[fName]))
			finalOut[i] += weights[fName]*elem[1]
			i += 1

	# Divide them by the number of files to get the mean
	for i in range(numTests) :
		finalOut[i] = finalOut[i]/weightSum

	# Write the final results
	f = open('combinedOutput.csv','w')
	f.write("test_id,rating\n")
	for i in range(numTests) :
		f.write(str(i)+","+str(finalOut[i])+'\n')

	f.close()


