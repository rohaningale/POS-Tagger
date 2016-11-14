# POS-Tagger
Implemented a POS tagger to predict the POS for each word in the given set of sentences.
Implemented 3 approaches for the prediction task:
1. Naive Bayes:
Used the Naive Bayes assumption to consider the POS of each word independent of POS of other words. Predicted the POS of the word based on the probabilities obtained from training data.
2. Hidden Markov Model:
Formulated a hidden markov model for the words and POS of each sentence. Implemented viterbi algorithm to determine the most probable sequence of POS for the given sentence.
3. Forward/Backward Algorithm:
Implemented the Forward/Backward algorithm to determine the most probable POS for each word such that the POS of each word is equally dependent on the POS of all the words and not just previous word. Thus incorrectly predicting any of the previous words in the sentence won't affect the prediiction of word currently under consideration.

Run Command: python label.py <test_data> <train_data>

Input Parameters:

 test_data -> Path to test file
 
 train_data -> Path to train file

 Example Run Command: python label.py bc.test bc.train
