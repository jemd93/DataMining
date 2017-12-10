The code we're submitting for the project includes multiple .py programs. 

Each program represents a specific algorithm and will run a training process and output a set of predictions corresponding
to the given test data set.

The command to run these programs is 'python3 <program>.py <trainingset.txt> <testset.txt>'

For the ALS algorithm, since it uses Apache Spark, the command to run is 'spark-submit als_spark.py <trainingset.txt> <testingset.txt>'. Optionally you can include in the end of the command '2>/dev/null' in order to hide the outputs with Spark execution progress.

We're also including "combineOutputs.py", a program that can be run by using python3 combineOutputs.py file1 file2 file3 ... fileX , and it'll produce an output file that represents the average of the predicted scores of all the files in the list. We used this in the project in order to obtain the ensemble average of the predictions, which ended up giving us a better score in Kaggle.
