import pandas as pd
import datetime
import re
import spacy
from spacy.util import minibatch, compounding
import random


#lower casing
def all_lower(my_list):
    return [x.lower() for x in my_list]

# removing \n
def removing_char(lst):
    temp = []
    for x in lst:
        temp.append(x.strip())
    return temp

#separating into strings and creating tags
def single_value(lst):
    z = ''
    for line in lst:
        z = z + ',' + line
    temp = z.split(',')
    tag_list = []
    for x in temp:
        tag_list.append(x.strip())
    return set(tag_list)

data_frame  = pd.read_csv("your dataset path")
d={}

for col in data_frame['Name of categorical Columns For which you want to Create Custom NER']:
    d[col]=pd.unique(data_frame[col].dropna())
 
del data_frame

entity_labels={}

for key, value in d.items():
  entity_labels[key] = single_value(removing_char(all_lower(value)))
del d

# here entity_labels is your categorical Dictionary on which you want to train your NER



# Read the CSV file
df = pd.read_csv("training_data_for_ner.csv")

# here we are preparing the training data
train_data = []

# Loop over each row in the CSV file
for index, row in df.iterrows():
    # Extract the text and label for each entity
    for column, label in entity_labels.items():
        # Check if the value in the current column is not empty
        if row[column]:
            if column == 'Tags':
                try:
                    
                    start_end_index = []
                    start = 0
                    for i, word in enumerate(row[column].split(",")):
                        end = start + len(word)
                        start_end_index.append((start, end,column))
                        start = end + 1

                    train_data.append((row[column],{"entities":start_end_index}))
                except:
                    pass
            # Add the text, start index, end index, and label to the training data
            else:
                try:

                    train_data.append((row[column], {"entities": [(0, len(row[column]), column)]}))
                except:
                    pass


# load the pre-trained model
nlp = spacy.load("en_core_web_lg")

# create a new entity type in the model
ner = nlp.create_pipe("ner")
nlp.add_pipe(ner, last=True)
# ner.add_label("CUSTOM_LABEL")

# Define the number of iterations and batch size
n_iter = 100
batch_size = 32

# Get the names of the pipes in the blank model
pipe_names = [pipe_name for pipe_name in nlp.pipe_names]

# Add the entity recognizer to the blank model if it doesn't exist
if "ner" not in pipe_names:
    # ner = nlp.create_pipe("ner")
    ner = nlp.add_pipe("ner")
    # nlp.add_pipe(ner, last=True)
else:
    ner = nlp.get_pipe("ner")

# Add the entity labels to the entity recognizer


# Disable the other pipes in the pipeline
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
with nlp.disable_pipes(*other_pipes):
    # Define the optimizer and get the losses
    optimizer = nlp.begin_training()
    losses = {}

    # Train the model for the specified number of iterations
    for i in range(n_iter):
        # Shuffle the training data
        random.shuffle(train_data)
        # Create batches of the training data
        # batches = minibatch(train_data, size=compounding(batch_size, max_batch_size=128))
        batches = minibatch(train_data, size=compounding(start=2,stop=batch_size,compound=2))
        # Loop over the batches and update the model
        for batch in batches:
            texts, annotations = zip(*batch)
            # nlp.update(texts, annotations, sgd=optimizer, losses=losses)
            example = []
    # Update the model with iterating each text
            for i in range(len(texts)):
                doc = nlp.make_doc(texts[i])
                example.append(Example.from_dict(doc, annotations[i]))
            nlp.update(example, sgd=optimizer, losses=losses)

# Save the model to disk
nlp.to_disk("custom_ner_model")


# In order to test the entity recognition, we can use the following code:

doc = nlp("write your query to test your entity recognition here")
for ent in doc.ents:
    print(ent.text,ent.label_)

ents = [ent.text for ent in doc.ents]
print(ents)