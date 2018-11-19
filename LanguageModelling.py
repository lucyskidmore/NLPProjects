import sys, re, nltk
from collections import Counter

#processing training text - splitting into sentences and words
training_text = open(sys.argv[1], encoding ='utf8') 
training_words = re.findall(r'\w+', open(sys.argv[1], encoding = 'utf8').read().lower())
training_sents = []
for line in training_text:
	training_sents.append(re.findall(r'\w+', line.lower()))

#getting unigram and bigram counts for training text 
unigram_counts = Counter(training_words)
bigrams = []
for sent in training_sents: 
	bigrams.extend(nltk.bigrams(sent, pad_left=True, pad_right=True))
bigram_counts = Counter(bigrams)
 
#unigram function, outputs probability of sentence 
def unigram_LM(sentence_x):
	prob_x = 1.0
	for word in sentence_x:
		word_prob = (unigram_counts[word]/len(training_words))
		prob_x = prob_x * word_prob
	return prob_x

#bigram function with smoothing option 
def bigram_LM(sentence_x, smoothing=0.0):
    unique_words = len(unigram_counts.keys()) + 2 # For the None paddings
    x_bigrams = nltk.bigrams(sentence_x, pad_left=True, pad_right=True)
    prob_x = 1.0
    for bg in x_bigrams:
        if bg[0] == None:
            prob_bg = (bigram_counts[bg]+smoothing)/(len(training_sents)+unique_words*smoothing)
        else:
            prob_bg = (bigram_counts[bg]+smoothing)/(unigram_counts[bg[0]]+unique_words*smoothing)
        prob_x = prob_x * prob_bg
    return prob_x

#function to compare probabilities and output to console 
def compareProb(prob_v1, prob_v2, words):
	print("V1 PROB: ", prob_v1)
	print("V2 PROB: ", prob_v2)
	if prob_v1 > prob_v2:
		print("V1 chosen: ", words[-1])
	elif prob_v1 == prob_v2:
		print("Probabilities equal, no chosen sentence")
	else:
		print("V2 chosen: ", words[-2])

#processing questions.txt  
questions = open(sys.argv[2], 'r')
n = 1
for line in questions:

	print("QUESTION", n, ":", line)
	n += 1

	#creating both versions of test sentence 
	words = re.findall(r'\w+', line.lower())
	sent_v1 = words[:-2]
	sent_v1[sent_v1.index('____')] = words[-1]
	sent_v2 = words[:-2]
	sent_v2[sent_v2.index('____')] = words[-2]

	#unigram model 
	print("Unigram model:")
	ug_prob_v1 = unigram_LM(sent_v1)
	ug_prob_v2 = unigram_LM(sent_v2)
	compareProb(ug_prob_v1, ug_prob_v2, words)

	#bigram model 
	print("Bigram model:")
	bg_prob_v1 = bigram_LM(sent_v1)
	bg_prob_v2 = bigram_LM(sent_v2)
	compareProb(bg_prob_v1, bg_prob_v2, words)

	#bigram model with add-1 smoothing 
	print("Bigram model with add-1 smoothing:")
	bg_s_prob_v1 = bigram_LM(sent_v1, smoothing=0.2)
	bg_s_prob_v2 = bigram_LM(sent_v2, smoothing=0.2)
	compareProb(bg_s_prob_v1, bg_s_prob_v2, words)

	print('')