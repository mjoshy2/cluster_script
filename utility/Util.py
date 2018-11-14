import pandas as pd
import tensorflow as tf
from sklearn.cluster import SpectralClustering
from math import floor


# read csv file and return a pandas dataframe object of the csv file
def open_file(filename):
    return pd.read_csv(filename)


# function accepts a dataframe object and the label as a string to decompose classes
# separates data into two classes and reset index so there are no join errors later
def decompose_classes(dataframe, label):
    positive_class = dataframe[dataframe[label] == 1]
    positive_class = positive_class.reset_index(drop=True)
    negative_class = dataframe[dataframe[label] == 0]
    negative_class = negative_class.reset_index(drop=True)
    return positive_class, negative_class


# accepts a df object and an array of column names that are categorical variables
# and returns a df that is one hot encoded
def encode(series):
    return pd.get_dummies(series.astype(str))


# accepts a dataframe and clusters the data using the Spectral Clustering Algorithm
def cluster_data(dataframe):
    clf = SpectralClustering(n_clusters=3, assign_labels='kmeans')
    labels = clf.fit_predict(X=dataframe)
    return labels


# separates out the data by numerical, categorical and target values
def pre_process_data(dataframe, numerical_col, categorical_col, target_col):
    numerical_features = dataframe[numerical_col]
    categorical_features = dataframe[categorical_col]
    label = dataframe[target_col]
    return numerical_features, categorical_features, label


# splits the data into 4 parts, train_features, train_labels, test_features, test_labels
def split_data(train_size, train_x, train_y):
    train_cnt = floor(train_x.shape[0] * train_size)
    x_train = train_x.iloc[0:train_cnt].values
    y_train = train_y.iloc[0:train_cnt].values
    x_test = train_x.iloc[train_cnt:].values
    y_test = train_y.iloc[train_cnt:].values
    return x_train, y_train, x_test, y_test


def multilayer_perceptron(x, weights, biases, keep_prob):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_1 = tf.nn.dropout(layer_1, keep_prob)
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer
