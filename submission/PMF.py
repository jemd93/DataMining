import pandas as pd
import sys
from surprise import Dataset
from surprise import SVD
from surprise import Reader


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

    # This part is to use all the data to train and get the output
    train_set = data.build_full_trainset()

    # Use PMF with surprise. To use PMF you use SVD with the parameter biased = False
    algo = SVD(biased = False)
    algo.train(train_set)

    f = open('PMFOutput.csv','w')
    f.write("test_id,rating\n")
    for i in range(len(dfTest)) :
        prediction = algo.predict(dfTest.at[i,'user_id'],dfTest.at[i,'business_id'],r_ui=4,verbose=True)
        predRating = prediction.est
        f.write(str(i)+","+str(predRating)+'\n')

    f.close()
