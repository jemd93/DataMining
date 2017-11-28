import pandas as pd
import numpy as np
import sys
from surprise import SVD, Dataset, Reader, evaluate, GridSearch


# compute the weighted error of a given hypothesis on a distribution
def computeError(h, examples, weights=None):
    if weights is None:
        weights = [1.] * len(examples)
    hypothesisResults = [h(x)*y for (x,y) in examples] # +1 if correct, else -1

    return hypothesisResults, sum(w for (z,w) in zip(hypothesisResults, weights) if z < 0)



# import training set as a pandas dataframe
#train_file_path = sys.argv[1]
train_file_path = r'/home/brieh/Documents/DataMining/project/data/DataMining/train_rating.txt'
examples = pd.read_csv(train_file_path)
examples = examples.drop(['train_id', 'date'], axis=1)



# create a trainset object
reader = Reader()
data = Dataset.load_from_df(examples, reader)
trainset = data.build_full_trainset()

dfTest = examples.drop(['rating'], axis=1)


# boost: [(list, label)], learner, int -> (list -> label)
# where a learner is (() -> (list, label)) -> (list -> label)
# boost the weak learner into a strong learner
def boost(examples, weakLearner, rounds=10):
   distr = normalize([1.] * len(examples))
   hypotheses = [None] * rounds
   alpha = [0] * rounds

   for t in range(rounds):
    # Use SVD with surprise
      algo = SVD()
      algo.train(data)

      hypotheses[t] = algo
      hypothesisResults, error = computeError(hypotheses[t], examples, distr)

      alpha[t] = 0.5 * math.log((1 - error) / (.0001 + error))
      distr = normalize([d * math.exp(-alpha[t] * h)
                         for (d,h) in zip(distr, hypothesisResults)])
      print("Round %d, error %.3f" % (t, error))

   def finalHypothesis(x):
      return sign(sum(a * h(x) for (a, h) in zip(alpha, hypotheses)))

   return finalHypothesis