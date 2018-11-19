import sys, random
from collections import Counter 
from itertools import product 
from sklearn.metrics import f1_score

def train(classes, feature_list, **kwargs):
    d_train = Dataset(sys.argv[1]).read_data()
    feature_dict = build_featuredict(d_train, feature_list, classes)
    w_dict = build_wdict(feature_dict)
    w_dict = perceptron(d_train, w_dict, feature_dict, classes, feature_list, kwargs['shuffle'], kwargs['maxIter'], kwargs['average'])
    return w_dict

def test(w_dict, classes, feature_list):
    d_test = Dataset(sys.argv[2]).read_data()
    feature_dict = build_featuredict(d_test, feature_list, classes)
    d_test_predict = dict()
    for key in d_test:
        y_predict = y_argmax(key, feature_dict, w_dict, classes, d_test, feature_list)
        #using y_predict to represent d_test_predict as [('WORD','LABEL')]
        d_test_predict[key] = [(d_test[key][i][0], y_predict[i]) for i in range(len(d_test[key]))]
    f1_score = compare(d_test, d_test_predict)
    print('F1 score for', feature_list, ':', f1_score, '\n')
    w_dict_analyse(w_dict, classes)
    
class Dataset(object):
    def __init__(self, file):
        self.file = file
	#data represented as [('WORD','LABEL')]
    def read_data(self):
        dataset = dict()
        with open(self.file) as f:
            for i, line in enumerate(f): 
                x_y = line.split()
                halfway = int(len(x_y)/2)
                x, y = x_y[:halfway], x_y[halfway:]
                for n in range (halfway):
                    if i not in dataset:
                        dataset[i] = [(x[n], y[n])]
                    else:
                        dataset[i].append((x[n], y[n]))
        return dataset   

def build_featuredict(dataset, feature_list, classes):
    featuredict_list = []
    for feature in feature_list:
        feature_dict = dict()
        for key in dataset:
            features = Features(dataset[key], classes)
            feature_dict[key] = features.feature_labels(feature)
        featuredict_list.append(feature_dict)   
    #add feature dicts together when using more than one feature     
    if len(featuredict_list) > 1:
        feature_dict = add_featuredicts(featuredict_list)
    return feature_dict
         
class Features(object):
    def __init__(self, instance, classes):
        self.instance = instance
        self.classes = classes
    #feature counts represented as {feature_label: count}    
    def feature_labels(self, feature):
        label_counter = Counter()
        for c in self.classes:
            for i, word_label in enumerate(self.instance):
                position, prev = feat2loc(feature, self.instance, i)
                if prev == True:
                    if c == self.instance[i][1]:
                        if i > 0:
                            label_counter[position+"_"+c] += 1
                        else:
                            label_counter[str(None)+"_"+c] += 1
                    else:
                        if i > 0:
                            label_counter[position+"_"+c] += 0
                        else:
                            label_counter[str(None)+"_"+c] += 0   
                else:
                    if c == self.instance[i][1]:
                        label_counter[position+"_"+c] += 1
                    else:
                        label_counter[position+"_"+c] += 0
        return label_counter
    
def feat2loc(feature, listtoslice, i):
    #defines features to extract 
    if feature == 'word_label':
        position = listtoslice[i][0]
        prev = False
    elif feature =='prevlabel_label':
        position = listtoslice[i-1][1]
        prev = True
    elif feature =='suffix_label':
        position = listtoslice[i][0][len(listtoslice[i][0])-3:]
        prev = False
    elif feature == 'prevword_label':
        position = listtoslice[i-1][0]
        prev = True
    return position, prev

def add_featuredicts(featuredict_list):
    combined_featuredict = dict()
    for key in featuredict_list[0]:
        combined_featuredict[key] = Counter()
        for n in range(len(featuredict_list)):
            combined_featuredict[key].update(featuredict_list[n][key])
    return combined_featuredict 

def build_wdict(feature_dict):
    w_dict = dict()
    for key in feature_dict:
        for x in feature_dict[key]:
            if x not in w_dict:
                w_dict[x] = 0
    return w_dict

def perceptron(d_train, w_dict, feature_dict, classes, feature_list, shuffle, maxIter, average):
	#updates weight dictionary according to training preferences
    keys = list(d_train.keys()) 
    if shuffle == True:
        random.shuffle(keys)
    c = 1 
    w_dict_list = []
    for i in range(0, maxIter):
        for key in keys:
            y_predict = y_argmax(key, feature_dict, w_dict, classes, d_train, feature_list)
            y_actual = [t[1] for t in d_train[key]]
            if y_predict != y_actual: 
                for i in range(len(d_train[key])):
                    for feature in feature_list:   
                        for path in y_actual: 
                            location = str(feat2loc(feature, d_train[key], i)[0])+"_"+str(path)
                            w_dict[location] = w_dict[location] + feature_dict[key][location]
                        for path in y_predict:
                            location = str(feat2loc(feature, d_train[key], i)[0])+"_"+str(path)
                            w_dict[location] = w_dict[location] - feature_dict[key][location]     
            if average == True:
                w_dict_list.append(w_dict)
                c += 1
    if average == False:
        return w_dict
    else: #getting average weights 		 		
        for word in w_dict:
            word_total = 0 
            for w_dict in w_dict_list:
                word_total += w_dict[word]
            average = word_total/c
            w_dict[word] = average
        return w_dict 

def y_argmax(key, feature_dict, w_dict, classes, dataset, feature_list):
    #creating list of all possible paths 
    paths = product(set(classes), repeat=len(dataset[key]))
    paths_sorted = []
    for path in paths:
        paths_sorted.append(path)
    paths_sorted.sort() #sorting to ensure reproducability 
    max_result = float('-inf')
    for path in paths_sorted:
        total = 0
        for i in range(len(dataset[key])):
            for feature in feature_list:
                location = str(feat2loc(feature, dataset[key], i)[0])+"_"+str(path[i])
                if location in w_dict: 
                    total += w_dict[location] * feature_dict[key][location]
                else:
                    total += 0
            if total > max_result:
                max_result = total
                y_predict = list(path)
    return y_predict
   
def compare(d_test, d_test_predict):
    d_test = convert(d_test)
    d_test_predict = convert(d_test_predict)
    f1_micro = f1_score(d_test, d_test_predict, average='micro', labels=['ORG', 'MISC', 'PER', 'LOC'])
    return f1_micro

def convert(dataset):
    values = []
    for key, instance in dataset.items():
        for pair in instance:
            values.append(pair[1])
    return values   

def w_dict_analyse(w_dict, classes):
    for c in classes:
        f_list = []
        for feature in w_dict:
            if feature[-len(c):] == c:
                f_list.append((feature, w_dict[feature]))
        f_list.sort(key=lambda x: x[1])
        top_10 = f_list[-10:] 
        print('Top 10 feature for ',c, ' :', top_10, '\n')

############################################################################################################	
random.seed(5)
classes = ['O', 'PER', 'LOC', 'ORG', 'MISC']
feature_list = ['word_label', 'prevlabel_label'] 
w_dict = train(classes, feature_list, shuffle=True, maxIter=5, average=True)
test(w_dict, classes, feature_list)