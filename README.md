# Exercise 12: DL4J


This exercise has the goal of training and evaluating two neural artificial neural networks.

## Step 1
Please `clone` the repository `https://github.com/idh-cologne-machine-learning-mit-java/exercise-12`.

Create a new branch, using your UzK username.

As always, inspect the code provided to you. The `pom.xml` contains a lot of dependencies, please also have a look at them. You can speed up things if you have a look at [this page](https://deeplearning4j.org/cpu) and add the platform-specific libraries into the pom.xml file.

## Step 2

The class `Titanic` contains a network similar to the one we discussed in class, using the Titanic data set.
If you run it as-is, you should get the following results:

```
========================Evaluation Metrics========================
 # of classes:    2
 Accuracy:        0,6154
 Precision:       0,0000
 Recall:          0,0000
 F1 Score:        0,0000
Precision, recall & F1: reported for positive class (class 1 - "1") only

Warning: 1 class was never predicted by the model and was excluded from average precision
Classes excluded from average precision: [1]

=========================Confusion Matrix=========================
   0   1
---------
 192   0 | 0 = 0
 120   0 | 1 = 1

Confusion matrix format: Actual (rowClass) predicted as (columnClass) N times
==================================================================
```

This is obviously not very satisfactory. Change the network such that the F1 score is higher than zero. What's the highest performance you can reach? 

Obvious things to try: Activation functions, layer number and size, [dropout](https://en.wikipedia.org/wiki/Dilution_(neural_networks)). 


## Step 3
In step 2, we have worked with a very small data set and pushed it into the network in one go. We will now switch to a more realistic setting: Loading a very large corpus of amazon reviews and their sentiment scores.

The file `src/main/resources/amazon/test_small.ft.txt.bz2` contains 10000 reviews and is only a small part of the corpus (bz2 is a compression format, such that its only 2.1 MB).  A slightly larger (100000 reviews) data set (21MB) can be downloaded from [here](https://www.dropbox.com/s/uwts2olluunohr9/train_small.ft.txt.bz2?dl=0). Do that, and put the file next to the test_small file. Do not uncompress any of these files.


## Step 4
The class in `de.ukoeln.idh.teaching.jml.ex12.Exercise12Main` contains the main code for this exercise, using the small data sets. As you can see, the `fit()` function this time gets an instance of `DataSetIterator` as argument, which can load the data set iteratively. This is needed to process large data sets.

As it is, the "network" contains a single output layer, which leads to a accuracy of 0.8279. Reconfigure the network, such that it has two hidden layers with 200 and 100 nodes respectively. The first layer should have a sigmoid activation function, the second uses the hyperbolic tangent `TANH`. 


## Step 5 (optional)
This is not really an exercise, but a chance to play with a larger data set and to test your GPU (if available). You can also download the full [test](https://www.dropbox.com/s/cfhj4oi9z3g4dai/test.ft.txt.bz2?dl=0) and [train](https://www.dropbox.com/s/rskzewv2pi1t0bc/train.ft.txt.bz2?dl=0) file. The full test set contains 400000 reviews and has a (compressed) file size of 50MB (too large for git). The full training set contains 3600000 reviews and has a compressed size of 443 MB. Adapt the code to load the large files and let it run on the entire data set. Before that, verify that your setup is as good as possible -- i.e., use a CUDA enabled GPU and the appropriate libraries.


## Step 6
As always, commit and push your code.


## Relevant Links

- [Javadoc deeplearning4j-core](https://javadoc.io/doc/org.deeplearning4j/deeplearning4j-core/latest/index.html)
- [Javadoc deeplearning4j-nn](https://javadoc.io/doc/org.deeplearning4j/deeplearning4j-nn/latest/index.html)
- [Amazon Reviews for Sentiment Analysis](https://www.kaggle.com/bittlingmayer/amazonreviews)