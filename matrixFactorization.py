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


numUsers = 693208
numItems = 145302

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
	data.split(5) 

	factors = 50
	epochs = 20
	lr = 0.01
	reg = 0.02

	# algo = SVD(n_factors=factors,n_epochs=epochs,lr_all=lr,reg_all=reg)
	# algo = SVD(n_factors=50,n_epochs=50,lr_all=0.02)
	# Evaluate the model with 5-fold cross validation
	# perf = evaluate(algo,data,measures=['RMSE'])
	# print("------------------------------")
	# print("LearnRate = " + str(lr))
	# print_perf(perf)

	# This part is to use all the data to train and get the output
	# So far 0.005 LR and 50 factors is the best with SVD
	train_set = data.build_full_trainset() 

	# Use SVD with surprise
	algo = SVDpp()
	algo.train(train_set)

	f = open('testOutput.csv','w')
	f.write("test_id,rating\n")
	for i in range(len(dfTest)) :
		prediction = algo.predict(dfTest.at[i,'user_id'],dfTest.at[i,'business_id'],r_ui=4,verbose=True)
		predRating = prediction.est
		f.write(str(i)+","+str(predRating)+'\n')

	f.close()



	# algo = SVD()
	# perf = evaluate(algo,data,measures=['RMSE','MAE'])
	# print(perf)