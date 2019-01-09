"""
Author: Blanca Vazquez
UNAM / IIMAS
Goal: mortalidadICU
Database: MIMIC III
Tool: Multilayer Perceptron in Python
"""

from sklearn.model_selection import KFold
import tensorflow as tf
import pathlib
import numpy as np
import csv
import os, sys
from sklearn.model_selection import train_test_split
import os
import datetime
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_csv(filepath):
  filepath = pathlib.Path(filepath)
  print("Loading data: ", filepath)
  with filepath.open('r') as f:
    return list(csv.reader(f))

def load_data(filepath):
    data = load_csv(filepath)
    parsed = np.array(data, dtype='float')
    return parsed

# Hiperparámetros
learning_rate = 0.001
batch_size = 50
epochs = 10
stddev=0.1

#Cargamos set de datos (90%):
dataset_1 = load_data('../data/mortalidadICU/train_val.csv')
X_all = dataset_1[:,:134]
y_all = dataset_1[:,134]
#cargamos dataset de test
dataset_2 = load_data('../data/mortalidadICU/dataset_test.csv')
test_x = dataset_2[:,:134]
test_y = dataset_2[:,134]

#Vamos con One-hot-Encoding
num_labels = y_all.shape[0]
index_offset = np.arange(num_labels) * 2
y_ohe_all = np.zeros((num_labels, 2))
y_ohe_all.flat[np.array(index_offset) + y_all.ravel().astype(int)] = 1

num_labels = test_y.shape[0]
index_offset = np.arange(num_labels) * 2
y_ohe_2 = np.zeros((num_labels, 2))
y_ohe_2.flat[np.array(index_offset) + test_y.ravel().astype(int)] = 1

#dividir el conjunto de datos en trainig y validation set
train_x, val_x, train_y, val_y = train_test_split(X_all, y_ohe_all, train_size=0.8,test_size=0.2, random_state=42)
print("---------------------------------------------------------")
print("Tamaño del conjunto de entrenamiento",train_x.shape)
print("Tamaño del conjunto de validación",val_x.shape)
print("Tamaño del conjunto de prueba",test_x.shape)
print("---------------------------------------------------------")
# TF graph
x = tf.placeholder(tf.float32, [None, 134])
x_test = tf.placeholder(tf.float32, [None,134])
y = tf.placeholder(tf.float32, [None, 2])
init_state = tf.placeholder(tf.float32,[None, self.state_size], name='init_state')
keep_prob = tf.placeholder(tf.float32) #dropout
#parámetros
W1 = tf.Variable(tf.truncated_normal([134, 10], stddev=stddev), name='W1')
B1 = tf.Variable(tf.zeros([10]), name='B1')
W2 = tf.Variable(tf.truncated_normal([10, 400], stddev=stddev), name='W2')
B2 = tf.Variable(tf.zeros([400]), name='B2')
W3 = tf.Variable(tf.truncated_normal([400,2],stddev=stddev),name='W3')
B3 = tf.Variable(tf.zeros([2]),name='B3')


with tf.name_scope('Model'):
    Y1 = tf.nn.dropout(tf.nn.sigmoid(x @ W1 + B1),keep_prob) #capa 1
    Y2 = tf.nn.dropout(tf.nn.sigmoid(Y1 @ W2 + B2),keep_prob) # capa2
    L = Y2 @ W3 + B3 # capa 3
    pred = tf.nn.softmax(L, name="pred")
with tf.name_scope('Loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=L))
    tf.summary.scalar("loss", loss)
with tf.name_scope('optimizador'):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
with tf.name_scope('Accuracy'):
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
merged = tf.summary.merge_all()

def run_train(session, train_x, train_y, val_x, val_y):
  session.run(init)
  results = []
  train_writer = tf.summary.FileWriter('graficos/training')
  val_writer = tf.summary.FileWriter('graficos/validacion')
  for epoch in range(epochs):
      total_batch = int(train_x.shape[0] / batch_size)
      print(">>>>>>>>>>>>>>>>>>>> Epoca", epoch+1, "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
      for i in range(total_batch):
          batch_x = train_x[i*batch_size:(i+1)*batch_size]
          batch_y = train_y[i*batch_size:(i+1)*batch_size]
          opt, train_loss, summary = session.run([optimizer, loss, merged],feed_dict={x: batch_x, y: batch_y,keep_prob:0.5})
          train_writer.add_summary(summary, i)
          if i % 100 == 0:
              print("Iteracion: {0:3d}:\t train loss: {1:.2f}".format(i, train_loss)) #Display the batch loss
      #Ejecutamos validación después de cada época, y visualizamos accuracy
      loss_acc, acc_acc,summary = session.run([loss,accuracy,merged], feed_dict={x:val_x, y:val_y, keep_prob:1.0})
      val_writer.add_summary(summary, i)
      results.append(acc_acc)
      print("Validation accuracy", results[-1])
  return results

with tf.Session() as session:
    train_x, val_x, train_y, val_y = train_test_split(X_all, y_ohe_all, train_size=0.8,test_size=0.2, random_state=42)
    result = run_train(session, train_x, train_y,val_x, val_y)
    print("==============================================================")
    print("Validation acuraccy for all epoch: ", sum(result)/epochs)
    print ("Test accuracy: %f" % session.run(accuracy, feed_dict={x: test_x, y: y_ohe_2, keep_prob:1.0}))
