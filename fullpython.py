#! pip install tensorflow_datasets
import numpy as np
import pandas as pd
import os
from pathlib import Path
import glob
import json
import tensorflow as tf
import tensorflow_datasets as tfds
import nltk
import cv2
import matplotlib.pyplot as plt
import random
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.utils import plot_model
from tqdm.notebook import tqdm

print("Initializing variables and loading data paths...")
train_length = 30000
val_length = 5000
trainList = []

train_img_dir = Path(r"C:\Users\DHARANI KOLLI\Downloads\CLEVR_v1.0\CLEVR_v1.0\images\train")
val_img_dir = Path(r"C:\Users\DHARANI KOLLI\Downloads\CLEVR_v1.0\CLEVR_v1.0\images\val")

print("Reading training questions...")
with open(r"C:\Users\DHARANI KOLLI\Downloads\CLEVR_v1.0\CLEVR_v1.0\questions\CLEVR_train_questions.json") as f:
    data = json.load(f)
    for k in tqdm(range(train_length)):
        i = data['questions'][k]
        temp = []
        path = str(train_img_dir / i['image_filename'])
        temp.append(path)
        temp.append(i['question'])
        temp.append(i['answer'])
        trainList.append(temp)

print("Creating training DataFrame...")
del data
labels = ['Path', 'Question', 'Answer']
train_dataframe = pd.DataFrame.from_records(trainList, columns=labels)
del trainList

print("Reading validation questions...")
valList = []
with open(r"C:\Users\DHARANI KOLLI\Downloads\CLEVR_v1.0\CLEVR_v1.0\questions\CLEVR_val_questions.json") as f:
    data = json.load(f)
    for k in range(val_length):
        i = data['questions'][k]
        temp = []
        path = str(val_img_dir / i['image_filename'])
        temp.append(path)
        temp.append(i['question'])
        temp.append(i['answer'])
        valList.append(temp)

del data
print("Creating validation DataFrame...")
val_dataframe = pd.DataFrame.from_records(valList, columns=labels)
del valList

def visualize_data(dataframe, num_samples=5):
    print("Visualizing data samples...")
    samples = dataframe.tail(num_samples)
    for _, sample in samples.iterrows():
        image_path = sample['Path']
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)
        plt.axis('off')
        plt.show()
        print("Question:", sample['Question'])
        print("Answer:", sample['Answer'])
        print("-----------------------------------------")

print("Visualizing Training Data:")
visualize_data(train_dataframe)

print("Visualizing Validation Data:")
visualize_data(val_dataframe)

print("Creating vocabulary set...")
vocab_set = set()
tokenizer = tfds.deprecated.text.Tokenizer()
for i in pd.concat([train_dataframe['Question'], val_dataframe['Question'], train_dataframe['Answer'], val_dataframe['Answer']]):
    vocab_set.update(tokenizer.tokenize(i))

print("Initializing encoder...")
encoder = tfds.deprecated.text.TokenTextEncoder(vocab_set)

index = 14
print("Testing the Encoder with sample question:")
example_text = encoder.encode(train_dataframe['Question'][index])
print("Original Text =", train_dataframe['Question'][index])
print("After Encoding =", example_text)

BATCH_SIZE = 30
IMG_SIZE = (200, 200)

def encode_fn(text):
    return np.array(encoder.encode(text.numpy()))

def preprocess(ip, ans):
    img, ques = ip
    img = tf.io.read_file(img)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.math.divide(img, 255)
    ques = tf.py_function(encode_fn, inp=[ques], Tout=tf.int32)
    paddings = [[0, 50 - tf.shape(ques)[0]]]
    ques = tf.pad(ques, paddings, 'CONSTANT', constant_values=0)
    ques.set_shape([50])
    ans = tf.py_function(encode_fn, inp=[ans], Tout=tf.int32)
    ans.set_shape([1])
    return (img, ques), ans

def create_pipeline(dataframe):
    print("Creating data pipeline...")
    raw_df = tf.data.Dataset.from_tensor_slices(((dataframe['Path'], dataframe['Question']), dataframe['Answer']))
    df = raw_df.map(preprocess)
    df = df.batch(BATCH_SIZE)
    return df

print("Creating training and validation datasets...")
train_dataset = create_pipeline(train_dataframe)
validation_dataset = create_pipeline(val_dataframe)

print("Building model architecture using Functional API...")
CNN_Input = tf.keras.layers.Input(shape=(200, 200, 3), name='image_input')
print("Loading ResNet50 model...")
resnet = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_tensor=CNN_Input)
cnn_features = resnet.output
cnn_output = tf.keras.layers.GlobalAveragePooling2D()(cnn_features)

RNN_Input = tf.keras.layers.Input(shape=(50,), name='text_input')
x = tf.keras.layers.Embedding(len(vocab_set)+1, 256)(RNN_Input)
print("Building LSTM layers...")
x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True))(x)
x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True))(x)
rnn_output = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512, return_sequences=False))(x)

print("Concatenating image and text features...")
concat = tf.keras.layers.concatenate([cnn_output, rnn_output])
dense_out = tf.keras.layers.Dense(len(vocab_set), activation='softmax', name='output')(concat)

model = tf.keras.Model(inputs=[CNN_Input, RNN_Input], outputs=dense_out)
print("Model summary:")
model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy(),
                       tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)])

print("Saving model structure to file...")
plot_model(model, to_file='model_structure.png', show_shapes=True)

print("Starting model training...")
history = model.fit(train_dataset, validation_data=validation_dataset, epochs=3)
print("Training complete. Saving model...")
model.save('vqa.h5')

print("Saving encoder to file...")
with open('encoder.pkl', 'wb') as file:
    pickle.dump(encoder, file)

def get_answer(image_path, question):
    print(f"Loading image: {image_path}")
    img = cv2.imread(image_path)
    img = cv2.resize(img, (200, 200))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    print("Encoding question:", question)
    encoded_question = encoder.encode(question)

    padded_question = pad_sequences([encoded_question], maxlen=50, padding='post')
    print("Running model prediction...")
    prediction = model.predict([img, padded_question])
    predicted_answer_index = np.argmax(prediction)
    predicted_answer = encoder.decode([predicted_answer_index])
    print("Predicted answer:", predicted_answer)
    return predicted_answer

print("Testing answer prediction:")
print(get_answer(r"C:\Users\DHARANI KOLLI\Downloads\CLEVR_v1.0\CLEVR_v1.0\images\train\CLEVR_train_000000.png", 'How many other things are there of the same shape as the tiny cyan matte object?'))