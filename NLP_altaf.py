# %%
import pandas as pd
import os
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, Bidirectional
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import json
import datetime

# %%
df = pd.read_csv(
    "https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv"
)

# %%
# EDA
df.head()
# %%
print(df["text"][2])
# %%
# data cleaning
text = df["text"]  # features
category = df["category"]  # target

for index, data in enumerate(text):
    data = data.lower()
    data = re.sub("[^a-zA-Z]", " ", data)
    text[index] = data

# %%
# features selection
# nothing to select

# %%
# data preprocessing

# tokenization --> features

vocab_num = 5000
oov_token = "<OOV>"  # out of vocab

tokenizer = Tokenizer(num_words=vocab_num, oov_token=oov_token)

# training the tokenizer to learn the words
tokenizer.fit_on_texts(text)

# view the convert data
text_index = tokenizer.word_index
print(list(text_index.items())[0:10])

# to convert text into numbers
text = tokenizer.texts_to_sequences(text)

# %%
# padding --> features

maxlen = []

for i in text:
    maxlen.append(len(i))

maxlen = int(np.ceil(np.percentile(maxlen, 75)))

text = pad_sequences(text, maxlen=maxlen, padding="post", truncating="post")

# %%

# OHE --> target

ohe = OneHotEncoder(sparse=False)
category = ohe.fit_transform(np.expand_dims(category, axis=-1))

# %%
# train test split

text = np.expand_dims(text, axis=-1)

X_train, X_test, y_train, y_test = train_test_split(
    text, category, shuffle=True, train_size=0.7, random_state=123
)

# %%
# model development


nb_classes = len(np.unique(y_train, axis=0))
embedding_dims = 64

model = Sequential()
model.add(Embedding(vocab_num, embedding_dims))
model.add(Bidirectional(LSTM(embedding_dims)))
model.add(Dropout(0.3))
model.add(Dense(nb_classes, activation="softmax"))  # output layer
model.summary()

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
plot_model(model, show_shapes=True)

# %%
# tensorboard

log_dir = os.path.join(
    os.getcwd(), "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
)

tb_callback = TensorBoard(log_dir=log_dir)

# early stopping

es_callback = EarlyStopping(monitor="loss", patience=5)

hist = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    callbacks=[tb_callback],
)

# %%
# model analysis

print(hist.history.keys())
plt.figure()
plt.plot(hist.history["loss"])
plt.plot(hist.history["val_loss"])
plt.legend(["training loss", "validation loss"])
plt.show()

plt.figure()
plt.plot(hist.history["accuracy"])
plt.plot(hist.history["val_accuracy"])
plt.legend(["training Accuracy", "validation Accuracy"])
plt.show()

# %%

y_pred = model.predict(X_test)
y_true = y_test

y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_true, axis=1)

print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))
print(accuracy_score(y_true, y_pred))

# %%
# model deployment

model_path = os.path.join(os.getcwd(), "saved_model", "model.h5")
ohe_path = os.path.join(os.getcwd(), "saved_model", "ohe.pkl")
tokenizer_path = os.path.join(os.getcwd(), "saved_model", "tokenizer.json")

model.save(model_path)

with open(ohe_path, "wb") as f:
    pickle.dump(ohe, f)

token_json = tokenizer.to_json()
with open(tokenizer_path, "w") as json_file:
    json.dump(token_json, json_file)
