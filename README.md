## Chatbot

### Goal of the project :
Code a very simple chatbot with a neural network (2 hidden layers of 8 neurons each)

### Contents of the files in this repo :
- main.py contains the actual programm
- intents.json contains the list of words/phrases used as training data sorted by tag (main idea of these words/phrases) and corresponding possible responses

### Main implementation steps :
#### 1. Try to import the data : 
Check if we already have some saved ready-to-use data in a pickle file
#### 2. Otherwise we have to Prepare the data (stored in the intents.json file) with the nltk library : 
Tokenize the word sequences (extract all the words in the sequence) and then apply stemming (bring each word to its root form (there? --> there, whats --> what...)) to further clean the data
#### 3. Create training data (inputs and outputs) : 
Use the bag of words technique to one hot encode each word and then create an input array (all the words a user can enter) and a corresponding output array (possible phrase tags)
#### 4. Build the model with tensorflow and tflearn
Set up a deep fully connected regression network which will be able to predict which tag to take a response from and answer the user according to his/her input
#### 5. Train and Test the model (after checking if it isn't already trained)
#### 6. Use the trained model to chat with a user 

