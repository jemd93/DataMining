import sys
import operator
import re, string
import csv
import math
import numpy as np
from scipy import sparse
import pandas as pd
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from sklearn.decomposition import NMF
from scipy.sparse.linalg import svds
from surprise import SVD
from surprise import Dataset
from surprise import evaluate, print_perf
from surprise import Reader


numUsers = 693208
numItems = 145302
# numUsers = 200000
# numItems = 40000

def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
	Q = Q.T
	for step in range(steps):
		for i in range(numUsers):
			for j in range(numItems):
				if R[i,j] > 0:
					eij = R[i,j] - np.dot(P[i,:],Q[:,j])
					for k in range(K):
						P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
						Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
		eR = np.dot(P,Q)
		e = 0
		for i in range(numUsers):
			for j in range(numItems):
				if R[i,j] > 0:
					e = e + pow(R[i,j] - np.dot(P[i,:],Q[:,j]), 2)
					for k in range(K):
						e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
		if e < 0.001:
			break
	return P, Q.T

# Check if a string is a number or not
def is_number(s):
	try:
		float(s)
		return True
	except ValueError:
		return False

# Read and process the data
def readData(filename):
	dataset = []
	with open(filename) as csvDataFile:
		csvReader = csv.reader(csvDataFile)
		for row in csvReader:
			dataset.append(row)

	# Transform numbers from string to int
	for i in range(len(dataset)):
		for j in range(len(dataset[i])) :
			if (is_number(dataset[i][j])) :
				dataset[i][j] = int(dataset[i][j])

	return dataset


# Main program	    		
if __name__ == '__main__':

	# Read csv into a pandas dataframe
	dfRatings = pd.read_csv(sys.argv[1])
	dfTest = pd.read_csv(sys.argv[2])
	# Delete unused columns
	del dfRatings['date']
	del dfRatings['train_id']
	del dfTest['date']
	del dfTest['test_id']


	# Set the rating scale and create the data for Surprise to use
	reader = Reader(rating_scale=(1, 5))
	data = Dataset.load_from_df(dfRatings[['user_id', 'business_id', 'rating']], reader)

	# Cross validation for tuning
	# Split in 5 folds
	# data.split(5) 

	# factors = 50
	# epochs = 20
	# lr = 0.01
	# reg = 0.02
	# while lr < 0.2 :
	# 	algo = SVD(n_factors=factors,n_epochs=epochs,lr_all=lr,reg_all=reg)

	# 	# Evaluate the model with 5-fold cross validation
	# 	perf = evaluate(algo,data,measures=['RMSE'])
	# 	print("------------------------------")
	# 	print("LearnRate = " + str(lr))
	# 	print_perf(perf)

	# 	lr += 0.01



	# This part is to use all the data to train and get the output
	# So far 0.005 LR and 50 factors is the best.
	train_set = data.build_full_trainset() 
	# Use SVD with surprise
	algo = SVD(n_factors=50)
	algo.train(train_set)

	f = open('testOutput.csv','w')
	for i in range(len(dfTest)) :
		prediction = algo.predict(dfTest.at[i,'user_id'],dfTest.at[i,'business_id'],r_ui=4,verbose=True)
		predRating = prediction.est
		f.write(str(i)+","+str(predRating)+'\n')

	f.close()



	# algo = SVD()
	# perf = evaluate(algo,data,measures=['RMSE','MAE'])
	# print(perf)

	# LEAVING THIS STUFF FOR NOW IN CASE WE WANNA TEST OR SOMETHING.
	# Training instance ID, user ID and item ID, rating, date
	# ratings = readData(sys.argv[1])
	# userRatings = lil_matrix((numUsers,numItems))
	# for row in ratings :
	# 	if row[0] != "train_id" :
	# 		if (int(row[1]) < numUsers) and (int(row[2]) < numItems) : 	
	# 			userRatings[row[1]-1,row[2]-1] = row[3]

	# CODE WITH SINGULAR VALUE DECOMPOSITION (SVD) :
	# U, sigma, Vt = svds(userRatings, k = 50) # U and Vt are the user feature and item feature matrix
	# sigma = np.diag(sigma) # Diagonal matrix
	# allPredictions = np.dot(np.dot(U, sigma), Vt)

	# print(allPredictions)

	# CODE OBTAINED FROM A BLOG
	# features = 20
	# userMatrix = np.random.rand(numUsers,features)
	# itemMatrix = np.random.rand(numItems,features)

	# nUMatrix, nIMatrix = matrix_factorization(userRatings,userMatrix,itemMatrix,features)
	# trainedMatrix = np.dot(nUmatrix,nImatrix)

	# print(trainedMatrix)

	# CODE WITH NON-NEGATIVE MATRIX FACTORIZATION FROM SCIKIT-LEARN
	# model = NMF(n_components=50, init='random', random_state=0)
	# W = model.fit_transform(userRatings);
	# H = model.components_;
	# print(W)
	# nR = np.dot(W,H)
	# print(nR)

	# OTHER OLDER STUFF
	# reader = Reader(rating_scale=(1, 5))
	# data = Dataset.load_from_df(df[['trainID', 'userID', 'itemID','rating','date']], reader)

	# umatrix = []
	# for i in range(numUsers) :
	# 	new = []
	# 	for j in range (numItems) : 	
	# 		new.append(0)
	# 	umatrix.append(new)

	# print(umatrix)




