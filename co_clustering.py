# This code to predict ratings for user-item pairs from the Yelp dataset 
# using the co-clustering algorithm provided by the Surprise library
# Written by Brie Hoffman, Nov 4 2017 for CMPT 741 final project


import pandas as pd
import sys
from surprise import CoClustering, Dataset, Reader, evaluate, GridSearch

# import training set as a pandas dataframe
train_file_path = sys.argv[1]
dftrain = pd.read_csv(train_file_path)
dftrain = dftrain.drop(['train_id', 'date'], axis=1)

# import test set as a pandas dataframe
test_file_path = sys.argv[2]  
dftest = pd.read_csv(test_file_path)
dftest = dftest.drop(['test_id', 'date'], axis=1)

# create a trainset object 
reader = Reader()
data = Dataset.load_from_df(dftrain, reader)
trainset = data.build_full_trainset()

"""
# The code here in quotes was for the cross-validation gridsearch for the best
# hyperparamters


#param_grid = {'n_cltr_u': [2,3,4,5,6,7,8,9,10],
#              'n_cltr_i': [2,3,4,5,6,7,8,9,10],
#              'n_epochs': [10,20,30,40,50,60,70,80,90,100]}


# Evaluate the model with 5-fold cross validation
#data.split(5)

#grid_search = GridSearch(CoClustering, param_grid, measures=['RMSE'])
#grid_search.evaluate(data)
#print ("after grid_search.evaluate(data)")
#print_perf(perf)

#results_df = pd.DataFrame.from_dict(grid_search.cv_results)
#print(results_df) """


# create a co-clustering algorithm
algo = CoClustering(n_cltr_u=3, n_cltr_i=3, n_epochs=100)
algo.train(trainset)


# use the trained algorithm to predict ratings for every user in the test set
f = open('testOutput.csv','w')
f.write("test_id,rating\n")
for i in range(len(dftest)) :
    prediction = algo.predict(dftest.at[i,'user_id'],dftest.at[i,'business_id'],r_ui=4,verbose=True)
    predRating = prediction.est
    f.write(str(i)+","+str(predRating)+'\n')

f.close()





