#!/usr/bin/python
import sys
import time

from nltk import ngrams
import nltk
from nltk.util import unique_list

# String="AM WILLING TO ENTER INTO COMPETITION WITH THE ANCIENTS AND FEEL ABLE TO SURPASS THEM FOR SINCE THERE'S EARLY DAYS IN WHICH I MADE THE METALS OF POKE CLEMENT I HAVE LEARNED SO MUCH THAT I CANNOT PRODUCE FAR BETTER PIECES OF THE KIND OF THE E OF THE E OF THE E OF THE E OF THE E OF THE E OF THE E OF THE E OF THE E OF THE E OF THE E OF THE E OF THE E OF THE E OF THE E OF THE E OF THE E OF THE E OF THE E OF THE E OF THE E OF THE E OF THE E OF THE E OF THE E OF THE E OF THE E OF THE E OF THE E OF THE E OF THE E OF THE E OF THE E OF THE E OF THE E OF THE E OF THE E OF THE E OF THE E OF THE E OF THE E OF THE E OF THE E OF THE E OF THE E OF THE E OF THE E OF THE E OF THE E OF THE E OF THE E OF THE E OF THE E OF THE E OF THE E OF THE E OF THE E OF THE E OF THE E OF THE E OF THE E OF THE E OF THE E OF THE E OF THE E OF THE E OF THE E OF THE E OF THE E OF THE"
# ngram_limit=4
# repetitions=2

#5338-24615-0002
#2902-9006-0015
#422-122949-0013
#8842-304647-0002
#2902-9006-0015

def prune_ngrams(String,ngram_limit=4,repetitions=2):
	""" As given in ----------: SELF-TRAINING FOR END-TO-END SPEECH RECOGNITION ----------:

	Some times the Looping erros can cause a huge shift in WER especiallly if there is no language model,
	Atlest language model wil guide the model to be in correct sequence of states
	....usuvally has these problems for very long sequences.
	"""
	if len(String) > ngram_limit:

		Ng=list(ngrams(String, ngram_limit))
		# print(Ng)
		Ng_Count_dict = {i : Ng.count(i) for i in Ng}
		Ng_Count_dict_rec= {i :0 for i in Ng}

		output_list=[]
		for key in Ng_Count_dict.keys():

			if Ng_Count_dict_rec[key] < repetitions:
				Ng_Count_dict_rec[key] += 1
				#-------------------------
				if not output_list:
		 			output_list.append(" ".join(list(key)))
				else:
					output_list.append(list(key)[-1])
				#-------------------------- 
	else:
		output_list = String
	return " ".join(output_list)

#String=['YES','YES','YES','YES','YES']
#print(prune_ngrams(String,ngram_limit=4,repetitions=2))
# print(Ng_Count_dict)
# print(String)
# print(" ".join(output_list))




