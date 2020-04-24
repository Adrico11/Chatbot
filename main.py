import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow as tf
import random
import json
import pickle

"""1. IMPORT DATA"""

with open("intents.json") as file:
    data = json.load(file)

try:
    #we check if we already have some saved ready-to-use data in a pickle file
    with open("data.pickle", "rb") as f: #"rb" means read file "b"
        words, labels, training, output = pickle.load(f) #directly use the saved data

except: #We want to avoid having to preprocess the data every time we use train/use the model
        #We want to prerpocess the data once and then directly access this saved data for future operations
        #This is especially useful for large sets of data


    """2. PREPARE DATA"""

    words = [] #all stemmed words of the patterns in one big list ["How", "are", "you"...]
    labels = [] #all labels (= tags=
    docs_x = [] #list of lists of tokenized words from each pattern ([["How", "are", "you"],...])
    docs_y = [] #list of corresponding tags


    for intent in data["intents"]: #loop through all the dictionaries in the json file
        for pattern in intent["patterns"]: #loop through all the words in the pattern
            wrds = nltk.word_tokenize(pattern) #to tokenize : retourne une liste avec tous les mots séparés de pattern
            words.extend(wrds) #rajoute tous les mots de la liste wrds dans words en un coup
            docs_x.append(wrds)
            docs_y.append(intent["tag"]) #stock parallèlment les mots el leur tag respectif

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    #print(docs_x)
    #print(docs_y)

    """Stemming : bring each word to its root form (there? --> there, whats --> what...)
    #We do that to improve the model's accuracy (can apply it to more cases without special characters attached)
    Rremove duplicates, set all leters to lowercase to clean the data"""

    words = [stemmer.stem(w.lower()) for w in words if  w != "?"] #stem the words and lowercase
    words = sorted(list(set(words))) #set(words) removes duplicates
    print(words)
    labels = sorted(labels)

    """3. CREATE TRAINING DATA (INPUTS AND OUTPUTS)"""
    """Translate inputs and outputs into one hot encoded"""

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    #print(words)
    for x,doc in enumerate(docs_x): #index,element (doc is each expression in the docs_x list)
        bag = []
        #print(doc)
        wrds = [stemmer.stem(w) for w in doc] #stemmed version of the words in each doc expression
        print(wrds)
        for w in words:
            #we add a 1 if the word exists or a 0 if it does not
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
        print(bag)
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1 #put a 1 at the index of the label of the word in the labels list
        training.append(bag) #traning list of 0 and 1
        output.append(output_row) #output list of 0 and 1

    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:  # "wb" means write in file "b"
        pickle.dump((words, labels, training, output),f)#save these variables in file f

"""4. BUILD MODEL"""
"""OUR MODEL PREDICTS WHICH TAG TO TAKE A RESPONSE FROM AND ANSWER THE USER"""

tf.reset_default_graph()#reset the data graph

#tell the model what type of input to expect (in our case, all lists in training have the same length = len(words))
net = tflearn.input_data(shape=[None,len(training[0])])
net = tflearn.fully_connected(net, 8) #connect the network : one input layer and a hidden layer of 8 neurons
net = tflearn.fully_connected(net, 8) #connect a second hidden layer of 8 neurons
net = tflearn.fully_connected(net, len(output[0]),activation="softmax") #connect the output layer with a certain number of output neurons
#softmax shows us the probability of each output neuron (output 1 with a 0.9 probability, output 5 with a 0.1 probability)
net = tflearn.regression(net) #Type of model regression, classification, knn... ?)

model = tflearn.DNN(net) #effictively create model

"""5. TRAIN MODEL"""

#Select the training parameters (nb of epchs, batch_size...)
#show_metric is just to show the output in a nice way...

try:
    model.load("model.tflearn") #check if the model is already trained

except:
    #if not, train the model
    #model = tflearn.DNN(net)
    model.fit(training, output, n_epoch=180,batch_size=4,show_metric=True)
    model.save("model.tflearn")

"""6.MAKE PREDICTIONS (TEST MODEL)"""

#We first need to convert a sentence from a user into a format that the network can understand
#ie a bag of words (list of 0 or 1 depending if the word is in the list of known words

def bag_of_words(s, words): #s is the user's sentence
    """Convert user's sentence in a bag of words understandable by the network"""
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    for se in s_words:
        for i,w in enumerate(words):
            if w == se:
                bag[i]=1

    return numpy.array(bag)

def chat():
    """User-Chatbot interface"""
    print("Start talking with the bot (type quit to stop)")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit": #end conversation
            break
        results = model.predict([bag_of_words(inp,words)])[0] #gives a list of output probabilities
        results_index = numpy.argmax(results) #index of greatest nb of the list (= highest probability)
        tag = labels[results_index] #corresonding label

        if results[results_index] > 0.7:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
            print(random.choice(responses))
        else:
            print("I didn't get that, try again or ask a different question...")

chat()