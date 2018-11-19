import sys, re, os, random, nltk
import numpy as np
from collections import Counter 

#may need to uncomment the below if NLTK Data not already downloaded  
#nltk.download('averaged_perceptron_tagger')

class Dataset(object):
	def __init__(self, root_folder, orientation):
		self.root_folder = root_folder
		self.orientation = orientation
		self.path = "{}/txt_sentoken/{}".format(root_folder, orientation)
		self.files = os.listdir(self.path)
		self.files.sort()

	#function to specify training:test split
	def read_dataset(self, start, end):
		dataset = dict()
		for filename in self.files[start:end]:
			if self.orientation == 'pos':
				dataset[filename] = 1
			else:
				dataset[filename] = -1
		return dataset
	
class Features(object):
	def __init__(self, doc_vector):
		self.doc_vector = doc_vector

	def bagofwords(self):
		return Counter(self.doc_vector)

	def bigrams(self):
		return Counter(nltk.bigrams(self.doc_vector))

	# using NLTK POS tagger to find adjectives
	def adjectives(self):
		adj_vector = []
		pos_tags = nltk.pos_tag(self.doc_vector)
		for tags in pos_tags:
			if tags[1] == 'JJ':
				adj_vector.append(tags[0])
		return Counter(adj_vector)

def doc2vector(doc, orientation):
	# reads in each doc and returns a vector of words
	pathname = os.path.join("{}/txt_sentoken/{}".format(root_folder, orientation), doc)
	with open(pathname) as file:
		doc_vector = re.sub("[^\w']"," ",file.read()).split()
	return doc_vector

def build_featuredict(dataset, feature):
	# takes dataset and outputs dictionary with phi(x) for every doc 
	# {x : phi(x)}
	feature_dict = dict()
	for x  in dataset:
		if dataset[x] == 1: 
			doc_vector = doc2vector(x, 'pos')
		else:
			doc_vector = doc2vector(x, 'neg')

		features = Features(doc_vector)

		if feature == 'bagofwords':
			feature_dict[x] = features.bagofwords()
		elif feature == 'bigrams':
			feature_dict[x] = features.bigrams()
		else:
			feature_dict[x] = features.adjectives()
	return feature_dict

def build_wdict(feature_dict):
	#iterates through feature_dict and builds weight dictionary
	w_dict = dict()
	for x in feature_dict:
		for feature in feature_dict[x]:
			if feature not in w_dict:
				w_dict[feature] = 0
	return w_dict

def perceptron(d_train, w_dict, feature_dict, shuffle, maxIter, average):
	#updates weight dictionary according to training preferences

	keys = list(d_train.keys()) 
	if shuffle == True:
		keys = list(d_train.keys())
		random.shuffle(keys)
	else:
		keys = list(d_train.keys())

	c = 1 
	w_dict_list = []
	for i in range(0, maxIter):
		for x in keys:
			if y_predict(x, feature_dict, w_dict) != d_train[x]:
				for feature in feature_dict[x]:
					#potential to add learning rate into this calculation 
					w_dict[feature] = w_dict[feature] + (d_train[x]*feature_dict[x][feature])
			if average == True:
				w_dict_list.append(w_dict)
				c += 1
	if average == False:
		return w_dict
	else:		 
		#getting average weights 		
		for word in w_dict:
			word_total = 0 
			for w_dict in w_dict_list:
				word_total += w_dict[word]
			average = word_total/c
			w_dict[word] = average
		return w_dict 

def y_predict(x, feature_dict, w_dict):
	#predicts class by taking dot product of weight vector and feature vector 
	total = 0
	for feature in feature_dict[x]:
		if feature in w_dict:
			total += feature_dict[x][feature] * w_dict[feature]
		else:
			total += 0
	y_predict = np.sign(total)
	return y_predict

def train(d_train, feature, **kwargs):
	feature_dict = build_featuredict(d_train, feature)
	w_dict = build_wdict(feature_dict)
	w_dict = perceptron(d_train, w_dict, feature_dict, kwargs['shuffle'], kwargs['maxIter'], kwargs['average'])
	return w_dict
		
def test(d_test, w_dict, feature_dict):
	d_test_new = dict()
	for x in d_test:
		d_test_new[x] = y_predict(x, feature_dict, w_dict)
	return d_test_new	

def compare(d_test, d_test_new): 
	correct_guesses = len(d_test)
	total_guesses = len(d_test)
	for key in d_test:
		if d_test[key] != d_test_new[key]:
			correct_guesses -= 1
	return correct_guesses/total_guesses*100

def w_dict_analyse(w_dict):
	w_list = []
	for word in w_dict:
		w_list.append((word, w_dict[word]))
	w_list.sort(key=lambda x: x[1])
	negative_features = w_list[:10] 
	positive_features = w_list[-10:]
	print("positive features:", positive_features)
	print("negative features:", negative_features)


############################################################################################################	

random.seed(5)
root_folder = sys.argv[1]

#creating trainset and testset 
data_pos = Dataset(root_folder, 'pos')
data_neg = Dataset(root_folder, 'neg')
d_train = {**data_neg.read_dataset(0, 800), **data_pos.read_dataset(0, 800)}
d_test = {**data_neg.read_dataset(800, None), **data_pos.read_dataset(800, None)}

#BAGOFWORDS
print('FEATURE: BAG-OF-WORDS') 
#STANDARD PERCEPTRON
w_dict = train(d_train, 'bagofwords', shuffle=False, maxIter=1, average=False)
test_featuredict = build_featuredict(d_test, 'bagofwords')
d_test_new = test(d_test, w_dict, test_featuredict)
accuracy = compare(d_test, d_test_new)
print('STANDARD BINARY PERCEPTRON:', accuracy)

#WITH SHUFFLING 
w_dict = train(d_train, 'bagofwords', shuffle=True, maxIter=1, average=False)
test_featuredict = build_featuredict(d_test, 'bagofwords')
d_test_new = test(d_test, w_dict, test_featuredict)
accuracy = compare(d_test, d_test_new)
print('PERCEPTRON WITH SHUFFLING:', accuracy)

#WITH SHUFFLING AND AVERAGING
w_dict = train(d_train, 'bagofwords', shuffle=True, maxIter=1, average=True)
test_featuredict = build_featuredict(d_test, 'bagofwords')
d_test_new = test(d_test, w_dict, test_featuredict)
accuracy = compare(d_test, d_test_new)
print('PERCEPTRON WITH SHUFFLING & AVERAGING', accuracy)

print('TOP 10 POSITIVELY & NEGATIVELY WEIGHTED FEATURES')	
w_dict_analyse(w_dict)
print('')

#BIGRAMS
print('FEATURE: BIGRAMS') 
#WITH SHUFFLING, AVERAGING 
w_dict = train(d_train, 'bigrams', shuffle=True, maxIter=1, average=True)
test_featuredict = build_featuredict(d_test, 'bigrams')
d_test_new = test(d_test, w_dict, test_featuredict)
accuracy = compare(d_test, d_test_new)
print('PERCEPTRON WITH SHUFFLING AND AVERAGING:', accuracy)

print('TOP 10 POSITIVELY & NEGATIVELY WEIGHTED FEATURES')	
w_dict_analyse(w_dict)
print('')

#ADJECTIVES
print('FEATURE: ADJECTIVES')
#WITH SHUFFLING, AVERAGING 
w_dict = train(d_train, 'adjectives', shuffle=True, maxIter=1, average=True)
test_featuredict = build_featuredict(d_test, 'adjectives')
d_test_new = test(d_test, w_dict, test_featuredict)
accuracy = compare(d_test, d_test_new)
print('PERCEPTRON WITH SHUFFLING AND AVERAGING:', accuracy)

print('TOP 10 POSITIVELY & NEGATIVELY WEIGHTED FEATURES')	
w_dict_analyse(w_dict)