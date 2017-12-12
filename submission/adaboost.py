import pandas as pd
import numpy as np
import sys, random
from surprise import SVD, Dataset, Reader, evaluate, GridSearch
from sklearn.preprocessing import normalize


# draw: [float] -> int
# pick an index from the given list of floats proportionally
# to the size of the entry (i.e. normalize to a probability
# distribution and draw according to the probabilities).
def draw(weights):
    choice = random.uniform(0, sum(weights))
    choiceIndex = 0
    for weight in weights:
        choice -= weight
        if choice <= 0:
            return choiceIndex
        choiceIndex += 1
    assert False, "Shouldn't get here"


# compute the weighted error of a given hypothesis on a distribution
def computeError(h, examples, weights=None):
    if weights is None:
        weights = [1.] * l
    for i in range(l):
        abserr[i] = math.abs(examples.at[i,'rating'] - algo.predict(examples.at[i,'user_id'],examples.at[i,'business_id']).est)
    hypothesisResults = [h(x)*y for (x,y) in examples] # +1 if correct, else -1

    return hypothesisResults, sum(w for (z,w) in zip(hypothesisResults, weights) if z < 0)

# boost: [(list, label)], learner, int -> (list -> label)
# where a learner is (() -> (list, label)) -> (list -> label)
# boost the weak learner into a strong learner
def boost(examples, rounds=10):

    l = len(examples)

    distr = normalize([1.]*l)
    hypotheses = [None] * rounds
    alpha = [0] * rounds

    for t in range(rounds):

        #create a training set based on the weight distribution
        for i in range(l):
            examples[i] = examples[draw(distr)]

        # create a trainset object
        reader = Reader()
        data = Dataset.load_from_df(examples, reader)
        trainset = data.build_full_trainset()

        # Use SVD with surprise
        algo = SVD()
        algo.train(trainset)
        hypotheses[t] = algo

        for i in range(l):
            abserr[i] = math.abs(examples.at[i,'rating'] - algo.predict(examples.at[i,'user_id'],examples.at[i,'business_id']).est)

        # update weights 
        delta = sum(x*y for x,y in zip(distr,abserr) if abserr > delta)
        hypRes = np.where(abserr > delta,-1,1)
        alpha[t] = 0.5 * math.log((1 - delta) / (.0001 + delta))

        distr = normalize([d * math.exp(-alpha[t] * h) for (d,h) in zip(distr, hypRes)]) 

       
    def finalHypothesis(x):
        return sign(sum(a * h(x) for (a, h) in zip(alpha, hypotheses))) 

    return finalHypothesis


def main():
    # import training set as a pandas dataframe
    train_file_path = r'/home/brieh/workspace/741/project/DataMining/train_rating.txt'
    dfTrain = pd.read_csv(train_file_path)
    dfTrain = dfTrain.drop(['train_id', 'date'], axis=1)

    adaboostHyp = boost(dfTrain)


if __name__ == '__main__':
    main()
