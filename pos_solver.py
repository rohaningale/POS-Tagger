###################################
# CS B551 Fall 2016, Assignment #3
#
# Your names and user ids:
#
# (Based on skeleton code by D. Crandall)
#
#
####
###################################
# CS B551 Fall 2016, Assignment #3
#
# Rohan Ingale (ringale)
# Rutuja Kulkarni (rutakulk)
# Sanket Mhaiskar (smhaiska)
# (Based on skeleton code by D. Crandall)

# '''
# A brief report on the program:
#
#  Run Command: python label.py <test_data> <train_data>
#
#  Input Parameters:
#  <test_data> -> Path to test file
#  <train_data> -> Path to train file
#
#  Example Run Command: python label.py bc.test bc.train
#
# Abstraction:
# 	1. Initial_probability: Probability that a part of speech appears at the start of the sentence.
# 	2. Emission probability: Probability of observing a word given the part of speech.
# 	3. Transition probability: Probability of one POS followed by the other.
#
# Assumptions:
# 	1. If word present,
# 			POS(word) = All POS(words) in training data
# 	2. If word is not present,
# 			POS(word) = All POS(words with suffix)
# 	3. If word is not present and suffix is also not found,
# 			POS(word) = NOUN with probability 1/N
#
# Algorithm:
# 	Simplified:
# 	For each word perform the following operation:
# 		For every possible POS:
# 			a. Compute Likelihood  = P(word)|POS
# 			   Compute Prior = P(POS)
# 			b. Calculate Prior * Likelihood
# 			   (We need not compute the denominator since we are trying to find maximum probability and not actual probability)
# 			c. Assign the word POS with maximum probability
#
#
# 	HMM:
# 	For first word, (initial_prob for each word * emission_prob for all words)
# 	For all other words,
# 		For all possible POS of current word perform:
# 			Consider every possible POS for previous word,
# 			Compute Transition Probablity
# 				Consider all possible pos for previous word,
# 					Compute the transition probability
# 					Choose max[(transition_probablity * previous_pos)]
# 					P(POS for current word) = max[(transition_probablity * previous_pos)] * Emission_probability(current_word)
#
# 		Consider the pos with maximum probability for the last word and then backtrack to get the entire sequence of pos.
# '''
####

import random
import math
import numpy as np

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:

    def __init__(self):
        self.marginal_probabilities = {}
        self.pos_word_dict = {}
        self.next_pos_dict = {}
        self.next_next_pos_dict = {}
        self.initial_probabilities = {}
        self.pos_count_dict = {}
        self.word_given_pos_probabilities = {}
        self.transition_probabilities = {}
        self.prev_transition_probabilities = {}
        self.pos_list = ['adj', 'adv', 'adp', 'conj', 'det', 'noun', 'num', 'pron', 'prt', 'verb', 'x', '.']
        self.total_words = 0
        self.total_pos = 0
        self.total_sentences = 0
        self.initial_pos_count_dict = {}
        self.suffix_pos_count = {}
        self.common_suffixes =  ['sion', 'tion', 'ance', 'ence', 'hood', 'ment', 'ness', 'like', 'able', 'ible', \
                                'ion', 'acy', 'age', 'ism', 'ist', 'ity', 'ful', 'ish', 'ous', 'ate', 'ify', 'ate', 'ize', \
                                 'ar', 'or', 'ic', 'al', 'ly', 'en', 'y']

    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling
    def posterior(self, algorithm, sentence, label):
        posterior = 0.0
        # print algorithm
        # print "marginal probabilities", self.marginal_probabilities[algorithm]
        # raw_input()

        if algorithm in self.marginal_probabilities and len(self.marginal_probabilities[algorithm]) == len(sentence):
            for i in range(len(self.marginal_probabilities[algorithm])):
                posterior += math.log(self.marginal_probabilities[algorithm][i])
        return posterior

    # Do the training!
    #
    def train(self, data):
        start_words = []
        prev_prev_pos = None
        prev_pos = None
        self.total_sentences = len(data)
        for sentence in data:
            for index in range(len(sentence[0])):
                self.total_words +=1
                if index == 0:
                    self.compute_initial_count(sentence[1][index])
                self.compute_pos_word_count(sentence[0][index], sentence[1][index])
                self.compute_suffix(sentence[0][index], sentence[1][index])
                if sentence[1][index] in self.pos_count_dict:
                    self.pos_count_dict[sentence[1][index]] +=1
                else:
                    self.pos_count_dict[sentence[1][index]] = 1
                if prev_pos <> None:
                    self.compute_pos_next_pos_count(prev_pos, sentence[1][index])
                    if prev_prev_pos <> None:
                        self.compute_pos_next_next_pos_count(prev_prev_pos, sentence[1][index])
                    prev_prev_pos = prev_pos
                prev_pos = sentence[1][index]
        self.smooth_initial_count()
        self.smooth_transition_count()
        self.compute_total_words()
        self.compute_initial_probabilities()
        self.compute_word_given_pos_probabilities()
        self.compute_transition_probabilities()
        self.compute_alternate_words_transition_probabilities()

    #pos count for start words
    def compute_initial_count(self, pos):
        if pos in self.initial_pos_count_dict:
            self.initial_pos_count_dict[pos] +=1
        else:
            self.initial_pos_count_dict[pos] = 1

    #smooth the initial pos count
    def smooth_initial_count(self):
        for pos in self.pos_list:
            if pos in self.initial_pos_count_dict:
                self.initial_pos_count_dict[pos] += 1
            else:
                self.initial_pos_count_dict[pos] = 1

    # smoothing of transition count to consider transitions not in train data
    def smooth_transition_count(self):
        for pos1 in self.pos_list:
            for pos2 in self.pos_list:
                key = pos1 + "_" + pos2
                if key in self.next_pos_dict:
                    self.next_pos_dict[key] +=1
                else:
                    self.next_pos_dict[key] = 1
                if key in self.next_next_pos_dict:
                    self.next_next_pos_dict[key] +=1
                else:
                    self.next_next_pos_dict[key] =1

    #compute the pos count for the suffixes
    def compute_suffix(self, word, pos):
        for suffix in self.common_suffixes:
            if word.endswith(suffix):
                key = suffix+"_"+pos
                if key in self.suffix_pos_count:
                    self.suffix_pos_count[key] +=1
                else:
                    self.suffix_pos_count[key] =1

    # compute the total number of words in train data
    def compute_total_words(self):
        for pos in self.pos_count_dict:
            self.total_pos += self.pos_count_dict[pos]

    #compute the probability transition counts
    def compute_pos_next_pos_count(self, prev_pos, curr_pos):
        key = prev_pos+"_"+curr_pos
        if key in self.next_pos_dict:
            self.next_pos_dict[key] +=1
        else:
            self.next_pos_dict[key] =1

    #used to compute the transition probability from s1 to s3 (independent of s2)
    def compute_pos_next_next_pos_count(self, prev_prev_pos, curr_pos):
        key = prev_prev_pos+"_"+curr_pos
        if key in self.next_next_pos_dict:
            self.next_next_pos_dict[key] +=1
        else:
            self.next_next_pos_dict[key] =1

    #compute the pos count for each word
    def compute_pos_word_count(self, word, pos):
        key = word+"_"+pos
        if key in self.pos_word_dict:
            self.pos_word_dict[key] +=1
        else:
            self.pos_word_dict[key] = 1

    #compute the probabilities for pos of start words
    def compute_initial_probabilities(self):
        for key in self.initial_pos_count_dict:
            self.initial_probabilities[key] = float(self.initial_pos_count_dict[key]) / float(self.total_sentences + len(self.pos_list))

    #compute  the emission probability
    def compute_word_given_pos_probabilities(self):
        for key in self.pos_word_dict:
            pos = key.split("_")[1]
            self.word_given_pos_probabilities[key] = float(self.pos_word_dict[key]) / float(self.pos_count_dict[pos])

    #compute the transition probabilities
    def compute_transition_probabilities(self):
        total = 0.0
        for key in self.next_pos_dict:
            total += self.next_pos_dict[key]
        for key in self.next_pos_dict:
            self.transition_probabilities[key] = float(self.next_pos_dict[key]) / float(total)

    #compute transition probabilities for alternate words
    def compute_alternate_words_transition_probabilities(self):
        total = 0.0
        #for key in self.next_pos_dict:
        #    total += self.next_pos_dict[key]
        #for key in self.next_pos_dict:
        #    self.transition_probabilities[key] = float(self.next_pos_dict[key]) / float(total)
        for key in self.next_next_pos_dict:
            pos = key.split("_")[0]
            total = 0.0
            for key1 in self.next_next_pos_dict:
                if key1.startswith(pos):
                    total +=self.next_next_pos_dict[key1]
            self.prev_transition_probabilities[key] =  float(self.next_next_pos_dict[key]) / float(total)

    #compute the probability given suffix
    def get_suffix_probable(self, word):
        posterior_probability = ["", 0.0]
        for suffix in self.common_suffixes:
            if word.endswith(suffix):
                current_probability = self.get_pos_for_suffix(suffix)
                if current_probability[1] > posterior_probability:
                    posterior_probability[0] = current_probability
                    posterior_probability[1] = current_probability
        return posterior_probability

    #to get the pos of words with given suffix
    def get_pos_for_suffix(self, suffix):
        total = 0.0
        likely_pos = ["", 0.0]
        for pos in self.pos_list:
            key = suffix + "_" + pos
            if key in self.suffix_pos_count:
                total += self.suffix_pos_count[key]
        for pos in self.pos_list:
            key = suffix + "_" + pos
            if key in self.suffix_pos_count:
                current_count = self.suffix_pos_count[key]
                likelihood = float(current_count) / float(total)
                prior = self.pos_count_dict[pos] / self.total_pos
                posterior = (likelihood) * (prior)
                if posterior > likely_pos[1]:
                    likely_pos[0] = pos
                    likely_pos[1] = posterior
        return likely_pos

    #gives most probable sequence for hmm by backtracking
    def get_most_probable_sequence(self, pos_chain, last_key, index):
        #print "pos chain, last key",pos_chain[index], last_key
        #raw_input()
        pos_sequence = []
        marginal_probabilities = []
        key = last_key
        while index >=0:
            #print "pos chain, last key", pos_chain[index], key
            #raw_input()
            pos_sequence.append(key)
            marginal_probabilities.append(pos_chain[index][key][1])
            key = pos_chain[index][key][0]
            index -=1
        return [pos_sequence, marginal_probabilities]

    #gives all possible pos of given word
    def get_word_pos_list(self, word):
        word_pos_list = []
        for pos in self.pos_list:
            key = word+"_"+pos
            if key in self.pos_word_dict:
                word_pos_list.append(pos)
        return word_pos_list

    #this method is used for hmm to compute the probability of word at start of sentence
    def get_start_word_probabilities(self, word):
        all_pos = {}
        current_word_pos_list = self.get_word_pos_list(word)
        if len(current_word_pos_list) > 0:
            for pos in current_word_pos_list:
                key = word+"_"+pos
                initial_probability = self.initial_probabilities[pos]
                emission_probability = self.word_given_pos_probabilities[key]
                total_probability = initial_probability * emission_probability
                all_pos[pos] = total_probability
        else:
            current_posterior = self.get_suffix_probable(word)
            if current_posterior[1] == 0.0:
                all_pos['noun'] = 1/ float(self.total_words)
            else:
                all_pos[current_posterior[0]] = current_posterior[1]
        return all_pos

    #this method is used for hmm to compute the probability of a word in sentence except start word
    def get_word_probabilities(self, word, previous_word_pos):
        all_pos = {}
        current_word_pos_list = self.get_word_pos_list(word)
        if len(current_word_pos_list) > 0:
            for current_pos in current_word_pos_list:
                transition_probabilities = {}
                for previous_pos in previous_word_pos:
                    transition_probabilities[previous_pos] = self.get_transition_probability(current_pos, previous_pos) * previous_word_pos[previous_pos]
                best_previous_pos = max(transition_probabilities, key=transition_probabilities.get)
                transition_probability = transition_probabilities[best_previous_pos]
                emission_probability = self.get_emission_probability(word, current_pos)
                total_probability = transition_probability * emission_probability
                all_pos[current_pos] = [best_previous_pos, total_probability]
        else:
            best_previous_pos = max(previous_word_pos, key=previous_word_pos.get)
            current_posterior = self.get_suffix_probable(word)
            if current_posterior[1] == 0.0:
                all_pos['noun'] = [best_previous_pos, 1 / float(self.total_words)]
            else:
                all_pos[current_posterior[0]] = [best_previous_pos, current_posterior[1]]

        return all_pos

    #this method is used in hmm to compute the emission probability
    def get_emission_probability(self, word, current_pos):
        key = word + "_" +current_pos
        return self.word_given_pos_probabilities[key]

    #this method is used in hmm to get the transition probability
    def get_transition_probability(self, current_pos, previous_pos):
        key = previous_pos + "_" + current_pos
        return self.transition_probabilities[key]

    #this method is used in hmm and gives the pos of last word with max probability
    def get_max_probability_pos(self, last_word):
        max_probability = -1.0
        max_pos = ""
        #print "last_word", last_word
        #raw_input()
        for pos in last_word:
            if last_word[pos][1] > max_probability:
                max_probability = last_word[pos][1]
                max_pos = pos
        #print "pos", max_pos
        return max_pos

    # Functions for each algorithm.
    #
    def simplified(self, sentence):
        print "simplified!!!"
        #print sentence
        #print self.pos_word_dict
        max_posterior = ["", 0.0]
        pos_sequence = []
        pos_sequence_probabilities = []
        for word in sentence:
            max_posterior[0] = ""
            max_posterior[1] = 0.0
            for pos in self.pos_list:
                word_pos_count = 0
                if pos not in self.pos_count_dict:
                    continue
                key = word+"_"+pos
                if key in self.pos_word_dict:
                    #print "got the word!!!"
                    word_pos_count = self.pos_word_dict[key]
                likelihood = float(word_pos_count) / float(self.pos_count_dict[pos])
                posterior = likelihood * (float(self.pos_count_dict[pos])/float(self.total_pos))
                if posterior > max_posterior[1]:
                    max_posterior[0] = pos
                    max_posterior[1] = posterior
            if max_posterior[1] == 0.0:
                if len(word)<=2:
                    max_posterior[0] = "noun"
                    max_posterior[1] = float(self.pos_count_dict['noun']) / float(self.total_pos)
                else:
                    current_posterior = self.get_suffix_probable(word)
                    if current_posterior[1] > 0.0:
                        max_posterior[0] = current_posterior[0]
                        max_posterior[1] = current_posterior[1]
                    else:
                        max_posterior[0] = "noun"
                        max_posterior[1] = float(1) / float(self.total_pos)
            pos_sequence.append(max_posterior[0])
            pos_sequence_probabilities.append(max_posterior[1])
        #print pos_sequence
        self.marginal_probabilities['Simplified'] = pos_sequence_probabilities
        return [[pos_sequence], [pos_sequence_probabilities ]]
            #if posterior == 0:
                #compute based on sufixes
        #return [ [ [ "noun" ] * len(sentence)], [[0] * len(sentence),] ]

    def hmm(self, sentence):
        pos_chain = {}
        prev_pos = self.get_start_word_probabilities(sentence[0])
        for pos in prev_pos:
            if 0 in pos_chain:
                pos_chain[0][pos] = ["", prev_pos[pos]]
            else:
                pos_chain[0] = {pos : ["", prev_pos[pos]]}
        for i in range(1, len(sentence)):
            all_possible_pos = self.get_word_probabilities(sentence[i], prev_pos)
            prev_pos = {}
            for current_pos in all_possible_pos:
                #print "current pos", all_possible_pos[current_pos]
                prev_pos[current_pos] = all_possible_pos[current_pos][1]
                #print "all possible pos", all_possible_pos
                #print "pos chain, index", pos_chain[i-1],i-1
                #raw_input()
                if i in pos_chain:
                    #print "already one element present!!!"
                    pos_chain[i][current_pos] = [all_possible_pos[current_pos][0], all_possible_pos[current_pos][1]]
                else:
                    pos_chain[i] = {current_pos: [all_possible_pos[current_pos][0], all_possible_pos[current_pos][1]]}

        last_word_pos = self.get_max_probability_pos(pos_chain[len(sentence)-1])
        #print "----------Chain------------"
        #print pos_chain
        #raw_input()
        most_probable_sequence = self.get_most_probable_sequence(pos_chain, last_word_pos, len(sentence)-1)
        #print most_probable_sequence[0]
        self.marginal_probabilities['HMM'] = most_probable_sequence[1][::-1]
        return [ [ most_probable_sequence[0][::-1]], [most_probable_sequence[1][::-1]] ]

    def get_transition_matrix(self, transition_probabilities):
        transition_matrix = [[0 for x in range(len(self.pos_list))] for y in range(len(self.pos_list))]
        #for key in transition_probabilities:
        #    pos = key.split("_")
        #    col = [i for i,x in enumerate(self.pos_list) if x==pos[0]][0]
        #    row = [i for i,x in enumerate(self.pos_list) if x==pos[0]][0]
        #    transition_matrix[row][col]  = transition_probabilities[key]
        for i in range(len(self.pos_list)):
            for j in range(len(self.pos_list)):
                key = self.pos_list[i]+"_"+self.pos_list[j]
                transition_matrix[j][i] = self.prev_transition_probabilities[key]
        return transition_matrix

    def complex(self, sentence):
        pos_sequence = []
        pos_probabilities = []
        prev_next_transition = self.get_transition_matrix(self.transition_probabilities)
        prev_prev_transition = self.get_transition_matrix(self.prev_transition_probabilities)
        #print prev_prev_transition
        #raw_input()
        transition_product = self.get_transition_product(prev_next_transition, prev_prev_transition)
        #print "transition product = ",transition_product[0]
        #raw_input()
        initial_probability = [[float(1)/ float(len(self.pos_list))] for y in range(len(self.pos_list))]
        for word in sentence:
            pos = None
            max_probability = 0.0
            emission_probabilities = self.get_emission_probabilities(word)
            #print "emission probabilities",emission_probabilities
            compute_probability = self.get_variable_elimination_probability(transition_product, emission_probabilities, initial_probability)
            #print compute_probability
            #raw_input()
            cntr = 0
            #print "computed prob",compute_probability
            #for row in compute_probability:
            #    if row > max_probability:
            #        max_probability = row[0]
            #        pos = self.pos_list[cntr]
            for i in range(len(compute_probability)):
                if compute_probability[i] > max_probability:
                     max_probability = compute_probability[i]
                     pos = self.pos_list[i]
            if pos is None:
                pos = "noun"
                max_probability = 1/float(self.total_words)
            #print "pos----",pos
            pos_sequence.append(pos)
            pos_probabilities.append(max_probability)
            self.marginal_probabilities['Complex'] = pos_probabilities
        return [[pos_sequence], [[pos_probabilities] ]]
        #return [ [ [ "noun" ] * len(sentence)], [pos_probabilities] ]

    # get the product of transition probabilities for variable elemination
    def get_transition_product(self,prev_next_transition, prev_prev_transition):
        prev_next_transition = np.array(prev_next_transition)
        prev_prev_transition = np.array(prev_prev_transition)
        return np.dot(prev_next_transition, prev_prev_transition)

    # get the final probability of all pos for given word
    def get_variable_elimination_probability(self, transition_product, emission_probability, initial_probability):
        #print "in varrrrrrrrrrrrrrrrrrrrrrrrrrrrr"
        prob_count = [0] * len(self.pos_list)
        for pos in emission_probability:
            index = 0;
            for i in range(len(self.pos_list)):
                if self.pos_list[i] == pos:
                    index = i
                    break
            for i in range(len(transition_product[index])):
                prob_count[index]+=(transition_product[index][i] * emission_probability[pos])
        #print "prob count",prob_count
        #print "initial probability",initial_probability
        #prob_count = np.matrix(prob_count)
        #initial_probability = np.matrix(initial_probability)
        #final_probability = np.dot(prob_count, initial_probability)
        final_probability = self.matrix_multiply(prob_count, initial_probability)
        #print "final probability", final_probability
        return final_probability
        #transition_product = np.array(transition_product)
        #emission_probability = np.array(emission_probability)
        #emission_transition_product = np.dot(emission_probability, transition_product)
        #print "emission transition product", emission_transition_product
        #raw_input()
        #final_probability = np.dot(emission_transition_product, initial_probability)

        #return final_probability
        #get_normalized_probability(final_probability)

    #def get_normalized_probability(self, final_probability):
    #    for
    #this is used for variable elimination method to get the matrix containing the word_pos probabilities
    def get_emission_probabilities(self, word):
        all_pos = {}
        current_word_pos_list = self.get_word_pos_list(word)
        if len(current_word_pos_list) > 0:
            for pos in current_word_pos_list:
                key = word+"_"+pos
                emission_probability = self.word_given_pos_probabilities[key]
                all_pos[pos] = emission_probability
        return all_pos

    def matrix_multiply(self, prob_count, initial_probability):
        probability = []
        for i in range(len(prob_count)):
            for j in range(len(initial_probability[0])):
                probability.append(prob_count[i] * initial_probability[0][j])
        return probability


    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It's supposed to return a list with two elements:
    #
    #  - The first element is a list of part-of-speech labelings of the sentence.
    #    Each of these is a list, one part of speech per word of the sentence.
    #
    #  - The second element is a list of probabilities, one per word. This is
    #    only needed for simplified() and complex() and is the marginal probability for each word.
    #
    def solve(self, algo, sentence):
        if algo == "Simplified":
            return self.simplified(sentence)
        elif algo == "HMM":
            return self.hmm(sentence)
        elif algo == "Complex":
            return self.complex(sentence)
        else:
            print "Unknown algo!"

