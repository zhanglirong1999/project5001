import numpy as np
import pandas as pd
import string
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import tensorflow as tf

from transformers import DistilBertTokenizer
from transformers import TFDistilBertForSequenceClassification
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from transformers import  TFXLNetForSequenceClassification, BertTokenizer, RobertaTokenizer


def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'', text)


def remove_emoji(text):
    emoji_pattern = re.compile(
        '['
        u'\U0001F600-\U0001F64F'  # emoticons
        u'\U0001F300-\U0001F5FF'  # symbols & pictographs
        u'\U0001F680-\U0001F6FF'  # transport & map symbols
        u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
        u'\U00002702-\U000027B0'
        u'\U000024C2-\U0001F251'
        ']+',
        flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def remove_html(text):
    html = re.compile(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    return re.sub(html, '', text)


def remove_punct(text):
    table = str.maketrans('', '', string.punctuation)
    return text.translate(table)


df_train = pd.read_csv("/data/wwangbw/Lirong/project/5001/project/data/train.csv")
df_test = pd.read_csv("/data/wwangbw/Lirong/project/5001/project/data/test.csv")

# df_train['text_combo'] = df_train['location'].astype(str) + " : " + df_train['keyword'].astype(str) + " : " + df_train['text'].astype(str)
# df_test['text_combo'] = df_test['location'].astype(str) + " : " + df_test['keyword'].astype(str) + " : " + df_test['text'].astype(str)

df_train_subset = df_train[["text", "target"]].copy()
df_train_subset.rename(columns = {'text':'text'}, inplace = True)

# Applying helper functions

df_train_subset['text'] = df_train_subset['text'].apply(lambda x: remove_URL(x))
df_train_subset['text'] = df_train_subset['text'].apply(lambda x: remove_emoji(x))
df_train_subset['text'] = df_train_subset['text'].apply(lambda x: remove_html(x))
df_train_subset['text'] = df_train_subset['text'].apply(lambda x: remove_punct(x))

# This is TfidfVectorizer method
# tfidf = TfidfVectorizer()
# X = tfidf.fit_transform(df_train_subset['text']).toarray()
# print(X)
# X = torch.from_numpy(X)
# # X = df_train_subset['text']
# y = df_train_subset['target']

X_train, X_test = train_test_split(df_train_subset, test_size=0.2, random_state=0, stratify=df_train["target"])

MODEL_NAME = 'distilbert-base-uncased-finetuned-sst-2-english'
BATCH_SIZE = 16
#generally I got the best results already after epoch 1 
#but it is good to try and see with a few more
N_EPOCHS = 3
## use DistilBertTokenizer，BertTokenizer， RobertaTokenizer
# tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
# tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

train_encodings = tokenizer(list(X_train["text"]),
                            truncation=True, 
                            padding=True)

test_encodings = tokenizer(list(X_test["text"]),
                           truncation=True, 
                           padding=True)

train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings),
                                    list(X_train["target"].values)))

test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encodings),
                                    list(X_test["target"].values)))

MODEL_NAME = "xlnet-base-cased"
BATCH_SIZE = 16
N_EPOCHS = 2

model = TFXLNetForSequenceClassification.from_pretrained(MODEL_NAME)#chose the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=18e-6)#define the loss function 
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)#build the model

model.compile(optimizer=optimizer,
              loss=loss,
              metrics=['accuracy'])# train the model 

#checkpoint_filepath = './mycheckpoint'
#checkpoint_filepath = 'model-{epoch:02d}-{val_loss:.2f}.h5'
checkpoint_filepath = 'mycheckpoint_xlnet'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    verbose=1,
    save_best_only=True)

model.fit(train_dataset.shuffle(32).batch(BATCH_SIZE),
          epochs=N_EPOCHS,
          batch_size=BATCH_SIZE,
        #   callbacks=[model_checkpoint_callback],
          validation_data=test_dataset.shuffle(len(X_train)).batch(1))

predict = model.predict(X_test)



