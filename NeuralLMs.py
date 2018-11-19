# -*- coding: utf-8 -*-
# Author: Robert Guthrie
#edited for COM6513 NLP Lab 9 

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

######################################################################
# An Example: N-Gram Language Modeling
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

CONTEXT_SIZE = 2
EMBEDDING_DIM = 140

SOS = None #start of sentence
EOS = None #end of sentence 

training_set = [[SOS, "The", "mathematician", "ran",  ".", EOS], 
[SOS, "The", "mathematician", "ran", "to", "the", "store",  ".", EOS], 
[SOS, "The", "physicist", "ran", "to", "the", "store",  ".", EOS], 
[SOS, "The", "philosopher", "thought", "about", "it", ".", EOS], 
[SOS, "The", "mathematician", "solved", "the", "open", "problem", ".", EOS]]

# build a list of tuples.  Each tuple is ([ word_i-2, word_i-1 ], target word)
trigrams = []
for sentence in training_set:
	for i in range(len(sentence) - 2): 
		trigrams.append(([sentence[i], sentence[i + 1]], sentence[i + 2]))

#add words to vocabulary             
vocab = set()
for sentence in training_set:
	for word in sentence:
		vocab.add(word)

#mapping words to indexes 
word_to_ix = {word: i for i, word in enumerate(vocab)}
ix_to_word = {word_to_ix[w]: w for w in word_to_ix}

class NGramLanguageModeler(nn.Module):

	def __init__(self, vocab_size, embedding_dim, context_size):
		super(NGramLanguageModeler, self).__init__()
		self.embeddings = nn.Embedding(vocab_size, embedding_dim)
		self.linear1 = nn.Linear(context_size * embedding_dim, 128)
		self.linear2 = nn.Linear(128, vocab_size)

	def forward(self, inputs):
		embeds = self.embeddings(inputs).view((1, -1))
		out = F.relu(self.linear1(embeds))
		out = self.linear2(out)
		log_probs = F.log_softmax(out, dim=1)
		return log_probs

	#function to retrieve word embeddings to use in test 
	def get_embeddings(self, word_ix):
		word = autograd.Variable(torch.LongTensor([word_ix]))
		return self.embeddings(word).view(1, -1)

#training model 
losses = []
loss_function = nn.NLLLoss()
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(10):
    total_loss = torch.Tensor([0])
    for context, target in trigrams:

        # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
        # into integer indices and wrap them in variables)
        context_idxs = [word_to_ix[w] for w in context]
        context_var = autograd.Variable(torch.LongTensor(context_idxs))

        # Step 2. Recall that torch *accumulates* gradients. Before passing in a
        # new instance, you need to zero out the gradients from the old
        # instance
        model.zero_grad()

        # Step 3. Run the forward pass, getting log probabilities over next
        # words
        log_probs = model(context_var)

        # Step 4. Compute your loss function. (Again, Torch wants the target
        # word wrapped in a variable)
        loss = loss_function(log_probs, autograd.Variable(
            torch.LongTensor([word_to_ix[target]])))

        # Step 5. Do the backward pass and update the gradient
        loss.backward()
        optimizer.step()

#testing the model 
test_sentence = [SOS, "The", "mathematician", "ran", "to", "the", "store",  ".", EOS]

trigrams = []
for i in range(len(test_sentence) - 2): 
	trigrams.append(([sentence[i], sentence[i + 1]], sentence[i + 2]))

print('Output from test sentence:') 
for context, target in trigrams:
	context_idxs = [word_to_ix[w] for w in context]
	context_var = autograd.Variable(torch.LongTensor(context_idxs))	
	log_probs = model(context_var)
	predict_label = torch.max(log_probs,1)
	predict_word = ix_to_word[predict_label[1].data[0]]
	print(context, predict_word)


######################################################################
# Test: Gap Prediction 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

EMBEDDING_DIM = 300

print()

def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)

def cosine_similarity(word_A, word_B, model, word_to_ix):
	embedding_A = model.get_embeddings(word_to_ix[word_A])
	embedding_B = model.get_embeddings(word_to_ix[word_B])
	cos = nn.CosineSimilarity()
	return cos(embedding_A, embedding_B).data[0]

#train model 
losses = []
loss_function = nn.NLLLoss()
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.007)

for epoch in range(10):
    for context, target in trigrams:

        # Step 1. Prepare the inputs to be passed to the model 
        context_var = make_context_vector(context, word_to_ix)

        # Step 2. Recall that torch *accumulates* gradients. Before passing in a
        # new instance, you need to zero out the gradients from the old
        # instance
        model.zero_grad()

        # Step 3. Run the forward pass, getting log probabilities over next
        # words
        log_probs = model(context_var)

        # Step 4. Compute your loss function. (Again, Torch wants the target
        # word wrapped in a variable)
        loss = loss_function(log_probs, autograd.Variable(
            torch.LongTensor([word_to_ix[target]])))

        # Step 5. Do the backward pass and update the gradient
        loss.backward()
        optimizer.step()


#testing model
print("Predict between 'physicist' and 'philosopher' for the sentence") 
print("'The ______ solved the open problem.'") 

context = [SOS, "The"] 

context_vector = make_context_vector(context, word_to_ix)
log_probs = model(context_vector) 

#get probs for both options 'philospher' and 'physicist'
physicist_probs = log_probs[0][word_to_ix['physicist']].data[0]
philosopher_probs = log_probs[0][word_to_ix['philosopher']].data[0]

#choose highest probs 
if physicist_probs > philosopher_probs:
	predict_word = 'physicist'
else:
	predict_word = 'philosopher'

print('Prediction:', predict_word)
print()

#checing cosine similarity 
print('Checking cosine similarity:')
print('physicist & mathematician:', cosine_similarity('physicist', 'mathematician', model, word_to_ix))
print('philosopher & mathematician', cosine_similarity('philosopher', 'mathematician', model, word_to_ix))
 

