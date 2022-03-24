import random
import json
import pickle
import numpy as np
import pandas as pd
import os

import regex as re

# Disable TensorFlow debugging information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)



def find_element_key_is_value(list, key, value):
    for dict in list:
        if dict[key] == value:
            return dict # return the dictionnary inside the intents where the key (tag) is equal to the value (intent class)
    return None

class Chatbot():


    def __init__(self, intent_file, intent_methods =  None, model_name="Chatbot_model",default_response=None):

        self.intents = self.load_json_intents(intent_file) 
        self.intent_methods = intent_methods 
        self.model_name = model_name 
        self.default_response = "I don't understand. I am still learning." if default_response is None else default_response

        self.model = []
        self.words = [] 
        self.classes = []

        self.lemmatizer = WordNetLemmatizer() # create lemmatizer, it's used to pick up the root of the word. example: loved -> love

    @staticmethod
    def load_json_intents(intent_file):
        with open(intent_file) as data_file:
            intents = json.load(data_file)
        return intents

    def train_model(self):

        self.words = [] # list of words in the pattern
        self.classes = [] # classes are the intents
        documents = [] # list of tuples (words,intents)
        ignore_letters = ['!', '?', ',', '.'] # caracters to ignore

        for intent in self.intents['intents']: # loop through intents
            for pattern in intent['patterns']: # loop through patterns
                word = nltk.word_tokenize(pattern) # tokenize the pattern
                self.words.extend(word) # add to our words list
                documents.append((word, intent['tag'])) # add to documents the list of words and the intent
                if intent['tag'] not in self.classes: # add to our classes list if our intent is not in it
                    self.classes.append(intent['tag'])

        self.words = [self.lemmatizer.lemmatize(w.lower()) for w in self.words if w not in ignore_letters] # remove all the ignored letters
        self.words = sorted(list(set(self.words))) # remove duplicates

        self.classes = sorted(list(set(self.classes))) # remove duplicates


        training = [] 
        output_empty = [0] * len(self.classes) # create an empty list for our output

        for doc in documents: # loop through documents(list of tuples(words,intents))
            bag = []
            word_patterns = doc[0] # get the words of the current tuple
            word_patterns = [self.lemmatizer.lemmatize(word.lower()) for word in word_patterns] # lemmatize the words inside the word_patterns
            for word in self.words: # loop through words
                bag.append(1) if word in word_patterns else bag.append(0) # add 1 to bag if the word is in the word_patterns else add 0

            output_row = list(output_empty) 
            output_row[self.classes.index(doc[1])] = 1 # set the value of the output row to 1 for the current intent
            training.append([bag, output_row]) # add the bag and the output row to the training set

        random.shuffle(training) # shuffle the training set
        training = np.array(training, dtype = object) # convert to numpy array

        train_x = list(training[:, 0]) # get the first column of the training set
        train_y = list(training[:, 1]) # get the second column of the training set

        self.model = Sequential() # create model
        self.model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(len(train_y[0]), activation='softmax'))

        sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True) # create optimizer
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy']) # compile the model

        self.hist = self.model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1) # train the model

    def save_model(self, model_name=None):
        if model_name is None:
            self.model.save(f"{self.model_name}.h5", self.hist)
            pickle.dump(self.words, open(f'{self.model_name}_words.pkl', 'wb'))
            pickle.dump(self.classes, open(f'{self.model_name}_classes.pkl', 'wb'))
        else:
            self.model.save(f"{model_name}.h5", self.hist)
            pickle.dump(self.words, open(f'{model_name}_words.pkl', 'wb'))
            pickle.dump(self.classes, open(f'{model_name}_classes.pkl', 'wb'))

    def load_model(self, model_name=None):
        if model_name is None:
            self.words = pickle.load(open(f'{self.model_name}_words.pkl', 'rb'))
            self.classes = pickle.load(open(f'{self.model_name}_classes.pkl', 'rb'))
            self.model = load_model(f'{self.model_name}.h5')
        else:
            self.words = pickle.load(open(f'{model_name}_words.pkl', 'rb'))
            self.classes = pickle.load(open(f'{model_name}_classes.pkl', 'rb'))
            self.model = load_model(f'{model_name}.h5')

    def _clean_up_sentence(self, sentence): # clean up the sentence by tokenizing it to words and lemmatizing the words 
        sentence_words = nltk.word_tokenize(sentence) # tokenize the sentence into words
        sentence_words = [self.lemmatizer.lemmatize(word.lower()) for word in sentence_words] # lemmatize the words inside the sentence_words
        return sentence_words # return the sentence_words list 

    def _bag_of_words(self, sentence, words): # create the bag of words for the sentence
        sentence_words = self._clean_up_sentence(sentence)  # clean up the sentence
        bag = [0] * len(words) # create an empty list with the length of the words list
        for s in sentence_words: # loop through the sentence_words tokenized into words and lemmatized
            for i, word in enumerate(words): # loop through the words list
                if word == s: # if the word is equal to the word in the sentence_words list then 0 is replace by 1 in the bag list
                    bag[i] = 1
        return np.array(bag) # return the bag list in a numpy array format

    def _predict_class(self, sentence): # predict the class of the sentence

        p = self._bag_of_words(sentence, self.words) # create the bag of words for the sentence
        res = self.model.predict(np.array([p]))[0]  # predict the class of the sentence
        ERROR_THRESHOLD = 0.6
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD] # get the results that are greater than the error threshold
        print(results)
        results.sort(key=lambda x: x[1], reverse=True) # sort the results in descending order
                
        return [
            {
                'tag': self.classes[result[0]],
                'entities': find_element_key_is_value(list = self.intents['intents'],
                                                      key = 'tag',
                                                      value = self.classes[result[0]])['entities'],
                'probability': str(result[1])
            }
            for result in results
        ] # return the results in a list of dictionaries format where each dictionary contains the tag, entities and probability of the tag

    def _get_response(self, intents): # get the response of the chatbot
        res = None
        try: # try to get the response
            tag = intents[0]['tag'] # get the tag of the intent
            intents = self.intents['intents']
            for intent in intents: # loop through the intents
                if intent['tag']  == tag:
                    res =  random.choice(intent['responses']) # get a random response from the list of responses
                    break
        except IndexError: # if the response is not found
            tag = 'no_intent'
            res = self.default_response

        return  res # return the tag and the response

    @staticmethod 
    def _get_entities(sentence, intent): 
        entities_to_find = intent['entities'] # get the entities to find
        entities = dict() # create an empty dictionary

        for entity, regexes in entities_to_find.items(): # loop through the entities to find
            for regex in regexes: # loop through the regexes
                regex = re.compile(pattern = regex, flags = re.IGNORECASE) # compile the regex
                match = re.search(pattern = regex,string = sentence) # search the regex in the sentence
                if match: # if the match is found
                    entities[entity] = match.group(1) # add the entity and the match to the entities dictionary
                    break
        return entities # return the entities dictionary

    def request(self, message):

        intents = self._predict_class(sentence = message) # predict the class of the sentence
        print(intents)
        # if the intent is found and the intent method is found in the intent_methods dictionary (meaning it request with entities)
        if intents[0]['tag'] in self.intent_methods.keys(): 
            entities = self._get_entities(message,intents[0]) # get the entities
            print('function like artist .. :', self.intent_methods[intents[0]['tag']](**entities))
            return self.intent_methods[intents[0]['tag']](**entities) # return the tag and the appropriete response
        else: # else it means that we have a simple request (without any entity)
            return self._get_response(intents) # return the response of the chatbot
