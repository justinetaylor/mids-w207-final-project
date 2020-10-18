#!/usr/bin/env python
# coding: utf-8

# In[13]:


# This tells matplotlib not to try opening a new window for each plot.
get_ipython().run_line_magic('matplotlib', 'inline')

# Libraries for reading, cleaning and plotting the dataa
import numpy as np 
import pandas as pd 
import csv
import matplotlib.pyplot as plt


# Libraries for models 
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix


# In[2]:


# Read in training data 
training_data = []
with open('data/train.csv', newline='') as csvfile:
    train_data = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in train_data:
        training_data.append(row)
            
# Convert to a numpy array of type int (except for the label row)
training_data = np.array(training_data[1:]).astype(int)   

# Read in test data
testing_data = []
with open('data/test.csv', newline='') as csvfile:
    test_data = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in test_data:
        testing_data.append(row)

# The testing file is huge so only read in max_test_data
max_test_data = 30001
test_data = np.array(testing_data[1:max_test_data]).astype(int)        


# In[3]:


# Shuffle the input: create a random permutation of the integers between 0 and the number of data points and apply this
# permutation to X and Y.
# NOTE: Each time you run this cell, you'll re-shuffle the data, resulting in a different ordering.
shuffle = np.random.permutation(np.arange(training_data.shape[0]))
training_data = training_data[shuffle]

# Split training data (labeled) into 80% training and 20% dev) and skip over the id column (it doesn't add an information)
# Immediately cast train data as floats so we can normalize it later 
split_index = int(len(training_data) * 0.8)
train_data = training_data[:split_index, 1:-1].astype(np.float64)
train_labels = training_data[:split_index, -1]
dev_data = training_data[split_index:, 1:-1].astype(np.float64)
dev_labels = training_data[split_index:, -1]
test_data = test_data[:,1:]

# Retrieve the mean and standard deviation of each feature - axis=0 is for going along the columns, keepdims forces the dimensions to stay the same
# Only compute it for the first ten features (they're numeric - not one hot or categorical)
num_columns = 10
# To avoid dividing by 0, add 1e-10 to the standard deviation 
smoothing = 1e-10
# USe the mean and standard deviation of the training data
feature_mean = train_data[:,:num_columns].mean(axis=0, keepdims=True)
feature_std = train_data[:,:num_columns].std(axis=0, keepdims=True)
# Normalize all numeric features except wilderness type and soil type (one-hot) - first 10 columns
train_data[:,:num_columns] = train_data[:,:num_columns] - feature_mean
train_data[:,:num_columns] = np.divide(train_data[:,:num_columns], feature_std + smoothing)
# Normalize dev data as well (using training mean and standard deviation)
dev_data[:,:num_columns] = dev_data[:,:num_columns] - feature_mean
dev_data[:,:num_columns] = dev_data[:,:num_columns]/(feature_std + smoothing)


# In[4]:


# Explore and confirm the shape of the data
print("Training data shape: {0} Training labels shape: {1}".format(train_data.shape, train_labels.shape))
print("Dev data shape: {0} Dev labels shape: {1}".format(train_data.shape, train_labels.shape))
print("Test data shape: ", test_data.shape)
print("First training example: ", train_data[0], train_labels [0])
print("First dev example: ", dev_data[0,:], dev_labels [0])
print("First test example: ", test_data[0])


# In[5]:


# Try a random forest - before any data cleaning 
def RandomForest(num_trees):
    model = RandomForestClassifier(num_trees,max_depth=8)
    model.fit(train_data, train_labels)
    predictions = model.predict(dev_data)
    score = model.score(dev_data, dev_labels)
    print("Random Forest Performance for {0} trees: {1}".format(num_trees,score))
    print("Random Forest Confusion Matrix:\n")
    print(confusion_matrix(predictions, dev_labels))
    
    #plot_confusion_matrix
    plot_confusion_matrix(model, dev_data, dev_labels, values_format = "d")
    plt.title("{} Tree Random Forest Confusion Matrix:".format(num_trees))
    plt.plot()
    print('\n\n')
    
num_trees_list = [1,3,5,10,100]
for num_trees in num_trees_list:
    RandomForest(num_trees)


# In[11]:


# Try Naive Bayes - before any data cleaning 
def NB(alf):
    model = BernoulliNB(alpha = alf)
    model.fit(train_data, train_labels)
    predictions = model.predict(dev_data)
    score = model.score(dev_data, dev_labels)
    print("BernoulliNB for alph = {0}: accuracy = {1}".format(alf,score))
    
    #plot_confusion_matrix
    plot_confusion_matrix(model, dev_data, dev_labels, values_format = "d")
    plt.title("NB Confusion Matrix with alpha: {}".format(alf))
    plt.plot()
    print('\n\n')
    
# the alpha isn't actually making a difference 
# alphas_list = [0.00001,0.001, 0.01, 0.1, 1, 10]
alphas_list = [0.01]
for alpha in alphas_list:
    NB(alpha)


# In[16]:


# Try K Nearest Neighbors - before any data cleaning 
def KNN(kn):
    model = KNeighborsClassifier(n_neighbors = kn)
    model.fit(train_data, train_labels)
    predictions = model.predict(dev_data)
    score = model.score(dev_data, dev_labels)
    print("KNN {0} neighbors : accuracy = {1}".format(kn,score))
    
    #plot_confusion_matrix
    plot_confusion_matrix(model, dev_data, dev_labels, values_format = "d")
    plt.title("KNN Confusion Matrix with {} Neighbors".format(kn))
    plt.plot()
    print('\n\n')
    
# the alpha isn't actually making a difference 
neigh_list = [1,2,4, 7, 10]
# neigh_list = [5]
for neigh in neigh_list:
    KNN(neigh)


# ### End matter
# 
# #### Acknowledgements/Sources
# 
# * That helpful stack overflow post
#   * the url for it
# * Relevant Documentation
#   * KNeighborsClassifier
#     * https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
#   * Pretty Confusion Matrix
#     * https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_confusion_matrix.html
# * Soil information
#   * https://www.uidaho.edu/cals/soil-orders/aridisols
#   
# #### Backup Formats
# 
# *because sometimes you just want to look at the markdown real quick*

# In[ ]:


#Create a backup of the jupyter notebook in a format for where changes are easier to see.
get_ipython().system('jupyter nbconvert clear-cut-solution.ipynb --to="python" --output="backups/clear-cut-solution"')
get_ipython().system('jupyter nbconvert clear-cut-solution.ipynb --to markdown --output="backups/clear-cut-solution"')

# Also archiving this bad boy
get_ipython().system('jupyter nbconvert clear-cut-solution.ipynb --to html --output="backups/clear-cut-solution"')

