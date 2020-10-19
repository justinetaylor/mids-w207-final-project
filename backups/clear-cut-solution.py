#!/usr/bin/env python
# coding: utf-8

# # Forest Cover Type Prediction
# #### Team: Clear-Cut Solution: Kevin Martin, Yang Jing, Justine Schabel

# In[130]:


# This tells matplotlib not to try opening a new window for each plot.
get_ipython().run_line_magic('matplotlib', 'inline')

# Libraries for reading, cleaning and plotting the dataa
import numpy as np 
import pandas as pd 
import csv
import matplotlib.pyplot as plt
import seaborn as sns

# Libraries for models 
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler


# ## Data Engineering

# ### Load Data

# In[131]:


# Read in training data 
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv", nrows=30000)


# ### Initial Data Exploration
# 
# First, we split the training data into a training data set (80%) and development data set (20%). We will also have a large, separate test data set. 

# In[132]:


# Split training data (labeled) into 80% training and 20% dev) and randomly sample 
training_data = train_df.sample(frac=0.8)
dev_data_df = train_df.drop(training_data.index)

# Examine shape of both data sets
print(training_data.shape)
print(dev_data_df.shape)

# Briefly examine feature attributes for the training data 
training_data.describe()


# Now, we'll isolate and explore the distribution of soil types. 

# In[133]:


# Isolate soil type column names
soil_df = training_data[["Id",'Soil_Type1', 'Soil_Type2', 'Soil_Type3',
       'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8',
       'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12',
       'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16',
       'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',
       'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24',
       'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28',
       'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32',
       'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',
       'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40']]

# TODO: What are these doing? 
soil_df_unpivoted = soil_df.melt(id_vars="Id",var_name="soil_type",value_name="yes")
mask1 = soil_df_unpivoted["yes"] ==1
soil_df_unpivoted = soil_df_unpivoted[mask1]


# Examine the fequencies of soil types
soil_df_unpivoted["soil_type"].value_counts().to_frame()

# Histogram of soil types 
plt.figure(figsize=(24,6))
plt.hist(soil_df_unpivoted["soil_type"],bins=40)
plt.xticks(rotation=90)
plt.show()


# As we can see in the histogram above, there is an uneven distribution of occurances of soil types.

# In[134]:


sns.violinplot(x=training_data['Cover_Type'],y=training_data['Elevation'])
plt.show()


# Here, we can see there is a relationship between the cover type and elevation. 

# In[ ]:


sns.displot(training_data['Cover_Type'],rug=True)
plt.show()


# Here we can see that the training data has a somewhat uniform distribution of covertype and this tells us that our data set is balanced. 

# In[ ]:


# Explore correlations between features
train_corr=training_data.corr()
# Rank correlations with "cover type"
train_corr['Cover_Type'].abs().sort_values(ascending=False)
st_list = ['Soil_Type1', 'Soil_Type2', 'Soil_Type3',
       'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type8',
       'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12',
       'Soil_Type13', 'Soil_Type14', 'Soil_Type16','Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',
       'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24',
       'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28',
       'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32',
       'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',
       'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40']
# Visualize the distribution of soil type and "cover type"
fig, axes = plt.subplots(19,2,figsize=(24,120))
for i in range(len(st_list)):
    sns.violinplot(y=train_df['Cover_Type'],x=training_data[st_list[i]], ax=axes[i//2,i%2])
plt.show()


# Here we can examine the relationship between soil type and cover type for each soil type. # TODO: Discuss more

# Now, we'll isolate and explore the distribution of wilderness types. 

# In[ ]:


wilderness_list =['Wilderness_Area1','Wilderness_Area2','Wilderness_Area3','Wilderness_Area4']

# Visualize the distribution of wilderness area and "cover type"
fig, axes = plt.subplots(2,2,figsize=(24,12))
for i in range(4):
    sns.violinplot(y=training_data['Cover_Type'],x=training_data[wilderness_list[i]], ax=axes[i//2,i%2])
plt.show()


# ### Feature Engineering
# 
# Now we'll begin modifying both the train and dev data based on our exploratory data analysis. First, we'll drop soil types that don't exist in the training set. Then we will combine soil types 35, 38, 39 and 40 because they have a very similar distribution. 

# In[ ]:


# Remove soil type 7 and 15 due to no data
training_data.drop(columns=["Soil_Type7", "Soil_Type15"], inplace=True)
dev_data_df.drop(columns=["Soil_Type7", "Soil_Type15"], inplace=True)

# Remove soil type 19, 37, 34, 21, 27,36,9, 28,8,25 due to no limited data - TODO: should we be dropping these? 
training_data.drop(columns=["Soil_Type19", "Soil_Type37","Soil_Type34", "Soil_Type21","Soil_Type27", "Soil_Type36","Soil_Type9", "Soil_Type28","Soil_Type8", "Soil_Type25"], inplace=True)
dev_data_df.drop(columns=["Soil_Type19", "Soil_Type37","Soil_Type34", "Soil_Type21","Soil_Type27", "Soil_Type36","Soil_Type9", "Soil_Type28","Soil_Type8", "Soil_Type25"], inplace=True)

# Combine soil type 35,38,39, 40
training_data["soil_type35383940"] = training_data["Soil_Type38"] +  training_data["Soil_Type39"] + training_data["Soil_Type40"] +  training_data["Soil_Type35"]
training_data.drop(columns=["Soil_Type35","Soil_Type38", "Soil_Type39",'Soil_Type40'], inplace=True)
dev_data_df["soil_type35383940"] = dev_data_df["Soil_Type38"] +  dev_data_df["Soil_Type39"] + dev_data_df["Soil_Type40"] +  dev_data_df["Soil_Type35"]
dev_data_df.drop(columns=["Soil_Type35","Soil_Type38", "Soil_Type39",'Soil_Type40'], inplace=True)

# Check shape is as expected
print(training_data.shape)
print(dev_data_df.shape)


# Additionally, we will scale the training data to have a mean of 0 and a variance of 1. Then we will retrieve the original training mean and variance for each feature and use that to standardize the development data.

# In[ ]:


# Collect numeric feature column names - so we can easily access these columns when modifying them 
num_cols = ['Elevation', 'Aspect', 'Slope',
       'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
       'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon',
       'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points']

# Normalize features using the standard scaler [training data]
# scaler = StandardScaler()
# num_train_df = pd.DataFrame(scaler.fit_transform(training_data[num_cols]), columns=num_cols)
# num_train_df.head()
# training_data.drop(columns=num_cols, inplace=True)
# print(training_data.shape)
# temp = pd.concat([training_data, num_train_df], axis=1)
# print(temp.shape)
# temp2 = training_data.merge(num_train_df, left_index=True, right_index=True)
# print(temp2.shape)
# # Standardize dev data using the training data original mean and standard deviation
# dev_data_minus_mean = dev_data_df[num_cols].sub(scaler.mean_, axis='columns')
# dev_data_standardized = dev_data_minus_mean.div(scaler.var_**(1/2)+1e-05, axis='columns')
# # Drop original numeric columns (unstandardized)
# dev_data_df.drop(columns=num_cols, inplace=True)
# # Replace with standardized columns
# dev_data_df = dev_data_df.merge(dev_data_standardized, left_index=True, right_index=True, how='left')
# print(dev_data_df.shape)


# In[ ]:


"""
Normalize Manually:
- Subtract mean and divide by standard deviation (feature wise) using training data mean/std
- To avoid dividing by 0, add 1e-10 to the standard deviation
- Take the mean on axis 0 (feature wise)
- Reshape it to be a column vector
- Transpose it so we can subtract properly
""" 
train_mean = training_data[num_cols].mean(axis=0)
train_std = training_data[num_cols].std(axis=0)

# Examine mean and std of training data features
print("Mean by Feature:\n", train_mean)
print("\nStandard deviation by Features:\n", train_std)


# In[ ]:


train_mean = train_mean.values.reshape(-1,1).transpose()
train_std = train_std.values.reshape(-1,1).transpose()


# In[ ]:


# Double check shape
print(train_mean.shape, train_std.shape)


# In[ ]:


training_data[num_cols] = training_data[num_cols].sub(train_mean, axis=1)
training_data[num_cols] = training_data[num_cols].divide(train_std,axis=1)

dev_data_df[num_cols] = dev_data_df[num_cols].sub(train_mean, axis=1)
dev_data_df[num_cols] = dev_data_df[num_cols].divide(train_std,axis=1)


# In[ ]:


training_data.head()


# Next we will split the data and the labels. 

# In[ ]:


# Split into data and labels
train_data = training_data.drop(columns=["Cover_Type"])
train_labels = training_data["Cover_Type"]
dev_data = dev_data_df.drop(columns=["Cover_Type"])
dev_labels = dev_data_df["Cover_Type"]
test_data = test_df

# Double check the shape
print(train_data.shape)
print(dev_data.shape)


# In[ ]:


# Explore and confirm the shape of the data
print("Training data shape: {0} Training labels shape: {1}".format(train_data.shape, train_labels.shape))
print("Dev data shape: {0} Dev labels shape: {1}".format(dev_data.shape, dev_data.shape))
print("Test data shape: ", test_data.shape)


# In[ ]:


# Examine Training Data 
train_data.head()


# ## Models
# #### Random Forest

# In[ ]:


# Try a random forest - before any data cleaning 
def RandomForest(num_trees):
    model = RandomForestClassifier(num_trees,max_depth=8)
    model.fit(train_data, train_labels)
    predictions = model.predict(dev_data)
    score = model.score(dev_data, dev_labels)
    print("Random Forest Performance for {0} trees: {1}".format(num_trees,score))
    # Plot_confusion_matrix
    plot_confusion_matrix(model, dev_data, dev_labels, values_format = "d")
    plt.title("{} Tree Random Forest Confusion Matrix:".format(num_trees))
    plt.plot()
    
num_trees_list = [1,3,5,10,100]
for num_trees in num_trees_list:
    RandomForest(num_trees)


# #### Naive Bayes (Bernoulli)

# In[ ]:


# Try Naive Bayes - before any data cleaning 
def NB(alf):
    model = BernoulliNB(alpha = alf)
    model.fit(train_data, train_labels)
    predictions = model.predict(dev_data)
    score = model.score(dev_data, dev_labels)
    print("BernoulliNB for alph = {0}: accuracy = {1}".format(alf,score))
    # Plot Confusion Matrix
    plot_confusion_matrix(model, dev_data, dev_labels, values_format = "d")
    plt.title("NB Confusion Matrix with alpha: {}".format(alf))
    plt.plot()
    print('\n\n')
    
# the alpha isn't actually making a difference 
# alphas_list = [0.00001,0.001, 0.01, 0.1, 1, 10]
alphas_list = [0.01]
for alpha in alphas_list:
    NB(alpha)


# #### K-Nearest Neighbors

# In[ ]:


# Try K Nearest Neighbors - before any data cleaning 
def KNN(kn):
    model = KNeighborsClassifier(n_neighbors = kn)
    model.fit(train_data, train_labels)
    predictions = model.predict(dev_data)
    score = model.score(dev_data, dev_labels)
    print("KNN {0} neighbors : accuracy = {1}".format(kn,score))
    # Plot Confusion Matrix
    plot_confusion_matrix(model, dev_data, dev_labels, values_format = "d")
    plt.title("KNN Confusion Matrix with {} Neighbors".format(kn))
    plt.plot()
    
# The alpha isn't actually making a difference 
neigh_list = [1,2,4, 7, 10]
for neigh in neigh_list:
    KNN(neigh)


# #### Multi-layer Perceptron

# In[ ]:


# Try Multi-Layer Perceptron - before any data cleaning 
def MLP():
#    model = MLPClassifier(solver='adam', alpha=1e-6, hidden_layer_sizes=(100, ), random_state=0) .8257
#    model = MLPClassifier(solver='adam', alpha=1e-3, hidden_layer_sizes=(100, ), random_state=0)  .82969
#    model = MLPClassifier(solver='adam', alpha=1e-3, hidden_layer_sizes=(200, ), random_state=0) .837
#    model = MLPClassifier(solver='adam', alpha=1e-3, hidden_layer_sizes=(100, ), random_state=0, activation='tanh') .83068

    # Default activation is 'relu', random state lets us get the same result every time (so we can tune other parameters)
    # max_iter is 200 by default, but more helps. alpha is the regularization parameter. solver is 'adam' by default
    model = MLPClassifier(alpha=1e-3, hidden_layer_sizes=(100,), random_state=0, max_iter=300) 
    model.fit(train_data, train_labels) 
    score = model.score(dev_data, dev_labels)
    print("MLP accuracy = ",score)

    
MLP()


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
#   * Preprocessing
#     * https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html
# * Soil information
#   * https://www.uidaho.edu/cals/soil-orders/aridisols
#   
# #### Backup Formats
# 
# *because sometimes you just want to look at the markdown or whatever real quick*

# In[ ]:


#Create a backup of the jupyter notebook in a format for where changes are easier to see.
get_ipython().system('jupyter nbconvert clear-cut-solution.ipynb --to="python" --output="backups/clear-cut-solution"')
get_ipython().system('jupyter nbconvert clear-cut-solution.ipynb --to markdown --output="backups/clear-cut-solution"')

# Also archiving this bad boy
get_ipython().system('jupyter nbconvert clear-cut-solution.ipynb --to html --output="backups/clear-cut-solution"')


# In[ ]:





# In[ ]:




