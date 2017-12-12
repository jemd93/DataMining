import sys
import operator
import re, string
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
	scoreFiles = (sys.argv[1:])
	numFiles = len(scoreFiles)

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
			finalOut[i] += elem[1]
			i += 1

	# Divide them by the number of files to get the mean
	for i in range(numTests) :
		finalOut[i] = finalOut[i]/numFiles

	# Write the final results
	f = open('combinedOutput.csv','w')
	f.write("test_id,rating\n")
	for i in range(numTests) :
		f.write(str(i)+","+str(finalOut[i])+'\n')

	f.close()


