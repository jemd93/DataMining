# This code to predict ratings for user-item pairs from the Yelp dataset 
# using a user-based collaborative filtering recommender
# Written by Brie Hoffman, Nov 4 2017 for CMPT 741 final project



import pandas as pd
from surprise import KNNWithMeans, Dataset, Reader, evaluate


# import training set as a pandas dataframe
#train_file_path = sys.argv[1]
train_file_path = r'/home/brieh/Documents/DataMining/project/data/DataMining/train_rating.txt'
dftrain = pd.read_csv(train_file_path)
dftrain = dftrain.drop(['train_id', 'date'], axis=1)
# slice only the first 1000 rows of the training set just to make it work
dftrain = dftrain.iloc[:1000]

# import test set as a pandas dataframe
#test_file_path = sys.argv[2]  
test_file_path = r'/home/brieh/Documents/DataMining/project/data/DataMining/test_rating.txt'
dftest = pd.read_csv(test_file_path)
dftest = dftest.drop(['test_id', 'date'], axis=1)


# create a trainset object 
reader = Reader()
data = Dataset.load_from_df(dftrain, reader)
trainingSet = data.build_full_trainset()


# create a user-based K-nearest neighbours algorithm
# - uses the Pearson correlation to measure user similarites 
# - takes user bias into account 
sim_options = {'name':'pearson'}
algo = KNNWithMeans(sim_options=sim_options)

# train the algorithm using the training set
########### fails here with MemoryError when I try to use the full set
algo.train(trainingSet)

# use the trained algorithm to predict ratings for the test set 
# output to a csv file
f = open('ub_testOutput.csv', 'w')
for i in range (len(dftest)):
    pred = algo.predict(dftest.at[i,'user_id'], dftest.at[i, 'business_id'], r_ui=4, verbose=True)
    predRating = pred.est
    f.write(str(i) + ", " + str(predRating) + '\n')
f.close()


