#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/justinetaylor/mids-w207-final-project/blob/yang_branch/clear_cut_solution.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Forest Cover Type Prediction
# #### Team: Clear-Cut Solution: Kevin Martin, Yang Jing, Justine Schabel

# ## Initial Setup
# ### Import Required Libraries

# In[1]:


# This tells matplotlib not to try opening a new window for each plot.
get_ipython().run_line_magic('matplotlib', 'inline')

# Libraries for reading, cleaning and plotting the dataa
import numpy as np 
import pandas as pd 
import csv
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import seaborn as sns
import os
import re

# Libraries for models 
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import tensorflow as tf
tf.enable_eager_execution()


# In[2]:


# Mount the drive for file storage
# from google.colab import drive
# drive.mount('/content/drive')


# In[3]:


# os.chdir('/content/drive/My Drive/W207-Final-Project')


# ### Load Data

# In[4]:


# Read in training data 
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")


# ## Exploratory Data Analysis

# #### Basic Attributes

# First, we check the data attributes, quality and shape.

# In[5]:


# Examine shape 
print(train_df.shape)

# Briefly examine feature attributes for the training data 
train_df.describe()


# In[6]:


# Check data types
train_df.dtypes


# #### Verify Dataset Is Balanced

# In[7]:


# Visualize the distribution of labels, "Cover_Type"
plt.figure(figsize=(6,4))
sns.displot(train_df["Cover_Type"],rug=True)
plt.show()


# Here we can see that the training data has a somewhat uniform distribution of covertype and this tells us that our data set is balanced. 

# #### Check For Null Values

# In[8]:


# Check for NA values
# `.isna()` returns a df with bools the first `.sum()` returns series, second is int 
print("There are {} NA values in the training data".format(train_df.isna().sum().sum()))
print("There are {} NA values in the test data\n".format(train_df.isna().sum().sum()))
print("There are {} values in the training data".format(train_df.count()[0]))
print("There are {} values in the test data".format(test_df.count()[0]))


# There are no null values in the dataset. 

# #### Distributions of Numeric Columns

# In[9]:


# Collect numeric feature column names - so we can easily access these columns when modifying them 
num_cols = ['Elevation', 'Slope','Aspect',
       'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
       'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon',
       'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points']


# In[10]:


# Visualize the distribution of numerical columns
col_count = len(num_cols)
rows = col_count//2
fig, axes = plt.subplots(rows,2,figsize=(20,20))
for i in range(col_count):
    for j in range(2):
        # TODO: Can you explain the index manipulaions with a comment? 
        col= train_df[num_cols[j+2*(i//2)]]
        sns.histplot(col, ax=axes[i//2][j])
        axes[i//2][j].grid()


# Here we can see the distribution are skewed for a few variables, espcially in the "distance" related ones, such as "Horizontal_Diestance_To_Fire_points". A log-transformation may improve the model performance. Also, there are zeros in these variables, we need to add 1 before performing the log transofrmation.

# In[11]:


# Visualize the distribution of numerical columns with Cover Type
fig, axes = plt.subplots(rows,2,figsize=(20,20))
for i in range(col_count):
    for j in range(2):
        col= train_df[num_cols[j+2*(i//2)]]
        sns.violinplot(x=train_df['Cover_Type'], y= col, ax=axes[i//2][j])
        axes[i//2][j].grid()


# First, we can see there is a relationship between the cover type and elevation. The difference in the other fetures by cover type seem less significant. Cover type 1 and 2 share a lot of similar features. We need to find a way to magnify the signal between the 2 cover types. 

# We also see there is not much differences in the relationship between the cover type and Aspect. The Aspect is expressed in degrees, and 0 degree and 360 degree is the same thing but represented differently. This probably contributed to poor distinction among the lables. In feature engineering, we'll extract the sine and cosine values to normalize this feature.
# 

# #### Correlation

# In[12]:


# Rank correlations with "cover type"
# This was train_corr1=train_df1.corr(), but train_df1 isn't defined yet? - Maybe we can remove this, since we have the heatmap below?
train_corr1=train_df.corr()
train_corr1['Cover_Type'].abs().sort_values(ascending=False)[:31]


# In[13]:


# Explore correlations between numerical features
train_corr = train_df[num_cols].corr()

# Plot a heat map for correlations
ax = plt.figure(figsize=(8,8))
sns.heatmap(train_corr, xticklabels=train_corr.columns.values, yticklabels=train_corr.columns.values)
plt.show()


# From the above, "Hillshade_9am" has strong correlation with "Hillshade_3pm" and "Aspect". We may drop this feature to avoid multi-collinearity.

# #### Soil Types

# Now, we'll isolate and explore the distribution of soil types. 

# In[14]:


# Get a list of categorical column names
cat_cols = ['Soil_Type1', 'Soil_Type2', 'Soil_Type3',
       'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8',
       'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12',
       'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16',
       'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',
       'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24',
       'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28',
       'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32',
       'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',
       'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40','Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3',
       'Wilderness_Area4']

soil_cols = cat_cols.copy()
soil_cols.append("Cover_Type")
soil_df = train_df[soil_cols]

# Now we convert the soil type columns back into one column with values as the "soil type"
soil_df_unpivoted = soil_df.melt(id_vars="Cover_Type",var_name="soil_type",value_name="yes")
# Only keep rows of where the "soil type" is "yes"
mask1 = soil_df_unpivoted["yes"] == 1
soil_df_unpivoted = soil_df_unpivoted[mask1]


# In[15]:


# Visualize cover type VS soil type in a pivot table. 
df1 = soil_df_unpivoted.groupby(["Cover_Type","soil_type"], as_index=False).count()
df1 = df1.pivot("Cover_Type","soil_type","yes")
df1


# As we can see in the pivot table above, there are similar combinations of soil types for different "cover type". We'll combine the soil types that share same "cover types" to reduce dimensionality. Further, "cover type 1" and "cover type 2" , "cover type 3" and "cover type 6" share many overlapping features. To magnify the signal, we'll combine features as an extra feature where there is a difference between the 2 pairs of cover types.

# In[16]:


# Visualize the distribution of soil type and "cover type"
st_list = ['Soil_Type1', 'Soil_Type2', 'Soil_Type3',
       'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type8',
       'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12',
       'Soil_Type13', 'Soil_Type14', 'Soil_Type16','Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',
       'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24',
       'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28',
       'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32',
       'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',
       'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40']

fig, axes = plt.subplots(19,2,figsize=(24,120))
for i in range(len(st_list)):
    sns.violinplot(y=train_df['Cover_Type'],x=train_df[st_list[i]], ax=axes[i//2,i%2])
plt.show()


# Here we can examine the relationship between soil type and cover type for each soil type. # TODO: Discuss more

# #### Wilderness Types

# Now, we'll isolate and explore the distribution of wilderness types. 

# In[17]:


wilderness_list =['Wilderness_Area1','Wilderness_Area2','Wilderness_Area3','Wilderness_Area4']

# Visualize the distribution of wilderness area and "cover type"
fig, axes = plt.subplots(2,2,figsize=(24,12))
for i in range(4):
    sns.violinplot(y=train_df['Cover_Type'],x=train_df[wilderness_list[i]], ax=axes[i//2,i%2])
plt.show()


# ## Feature Engineering 

# #### Scale Hillshade
# 
# Now we'll normalize the "Hillsdale" variables by dividing them by 255. TODO: Can we explain why? We'll scale them all later? 

# In[18]:


fe1_cols = ['Hillshade_9am', 'Hillshade_Noon',
       'Hillshade_3pm']
train_df[fe1_cols] = train_df[fe1_cols]/255


# #### Create New Soil Types

# Now we'll create additional features to magnify the differences betweeen cover type1 and 2, and covery type3 and 6.

# In[19]:


# Create additional features to magnify the differences between cover type 1 and 2
# Combine soil type 2,18,25,3,36,6,8,28,34 and wildness area 4 as only cover type 2 appers under these features
train_df["type2stwa4"] = train_df["Soil_Type6"] + train_df["Wilderness_Area4"] +  train_df["Soil_Type2"]+ train_df["Soil_Type18"] +  train_df["Soil_Type25"] +  train_df["Soil_Type3"] + train_df["Soil_Type36"]+ train_df["Soil_Type8"] + train_df["Soil_Type34"]+ train_df["Soil_Type28"]

# Combine soil type 20, 23, 24, 31, 33 and 34 as only cover type 6 appears under these features but not cover type 3.
train_df["type6st"] = train_df["Soil_Type20"] + train_df["Soil_Type23"]+ train_df["Soil_Type24"] +  train_df["Soil_Type31"] + train_df["Soil_Type33"] +  train_df["Soil_Type34"]


# #### Drop Non-Existant Soil Types
# 
# Now we'll drop soil types that don't exist in the training set. Then we will combine soil types 35, 38, 39 and 40 because they have a very similar distribution. 

# In[20]:


# Remove soil type 7 and 15 due to no data
train_df.drop(columns=["Soil_Type7", "Soil_Type15"], inplace=True)

# Remove soil type 19, 37, 34, 21, 27,36,28,8,25 due to no limited data - TODO: should we be dropping these? 
train_df.drop(columns=["Soil_Type19", "Soil_Type37","Soil_Type34", "Soil_Type21","Soil_Type27", "Soil_Type36","Soil_Type28","Soil_Type8", "Soil_Type25"], inplace=True)


# #### Combine Similar Soil Types

# In[21]:


# Combine soil type 35,38,39, 40
train_df["soil_type35383940"] = train_df["Soil_Type38"] +  train_df["Soil_Type39"] + train_df["Soil_Type40"] +  train_df["Soil_Type35"]
# Combine soil type 10,11, 16, 17
train_df["st10111617"] = train_df["Soil_Type10"] + train_df["Soil_Type11"] + train_df["Soil_Type16"] + train_df["Soil_Type17"]
# Combine soil type 9, 12
train_df["st912"] = train_df["Soil_Type9"] + train_df["Soil_Type12"] 
# Combine soil type 31,33
train_df["st3133"] = train_df["Soil_Type31"] + train_df["Soil_Type33"]
# Combine soil type 23, 24
train_df["st2324"] = train_df["Soil_Type23"] + train_df["Soil_Type24"]
# Combine soil type 6 and wilderness area 4
train_df["st6w4"] = train_df["Soil_Type6"] + train_df["Wilderness_Area4"]

# train_df.drop(columns=["Soil_Type35","Soil_Type38", "Soil_Type39",'Soil_Type40','Soil_Type10','Soil_Type11','Soil_Type16','Soil_Type17','Soil_Type9','Soil_Type12','Soil_Type31','Soil_Type33','Soil_Type23','Soil_Type24','Soil_Type6','Wilderness_Area4'], inplace=True)

# Check shape is as expected
print(train_df.shape)


# #### Transform Aspect 
# 
# Now we'll transform the Asepct feature.
# TODO: Explain more

# In[22]:


# Convert aspect into sine and cosine values 
train_df["ap_ew"] = np.sin(train_df["Aspect"]/180*np.pi)
train_df["ap_ns"] = np.cos(train_df["Aspect"]/180*np.pi)

# Drop Aspect column
train_df.drop(columns= ["Aspect"], inplace=True)
              
# Check shape is as expected
print(train_df.shape)


# In[23]:


# Visualize cover type VS the cosine of Aspect degerees
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(12,4))
sns.violinplot(x=train_df['Cover_Type'],y=train_df['ap_ew'],ax=ax1)
sns.histplot(train_df['ap_ew'],ax=ax2)
plt.show()


# After the feature transformation, we see improved distinction in median values, espeically for cover type 6, where the median is notably higher than that of other cover types  and the distribution is concentrated around the median. 

# #### Log and Polynomial Transformations
# 
# Now we'll log transform the features related to the distances.

# In[24]:


# Complie a list of features to perform log transformation
fe4_cols = ['Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
       'Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points']

# Check the minimum value 
min_vertical_distance = train_df['Vertical_Distance_To_Hydrology'].min()
print("Vertical_Distance_To_Hydrology Minimum: ", min_vertical_distance)

# Add 147 to ensure no negative or 0 in the values 
train_df[fe4_cols] = train_df[fe4_cols] + 147

# Log transform
train_df[fe4_cols] = np.log(train_df[fe4_cols])

# Add a polynominal feature
train_df["elv_pwd"] = train_df["Elevation"]**2


# #### Drop Id Column

# In[25]:


# TODO: Can this be removed? We should drop id in the training data we actually use. 
# Make a copy of train_df for modelling
train_df1 = train_df.copy()

# Drop Id column as it is not a meaningful feature.
train_df.drop(columns=["Id"],inplace=True)
test_df.drop(columns=["Id"],inplace=True)


# #### Drop Hillshade_9am
# 
# Hillshade_9am has a strong correlation with Hillshade_3pm and Aspect. TODO: If we're only dropping Hillshade_9am here we can drop it directly  

# In[26]:


all_features = set(train_df.columns.to_list())

# Select features to drop. 
to_drop = set(['Hillshade_9am'])

sel_features = list(all_features - to_drop)
train_df =train_df[sel_features]


# #### Split Data into Train/Dev/Test
# 
# Then, we split the training data into a training data set (80%) and development data set (20%). We will also have a large, separate test data set. 

# In[27]:


# Split training data (labeled) into 80% training and 20% dev) and randomly sample 
training_data = train_df.sample(frac=0.8)
dev_data_df = train_df.drop(training_data.index)

# Examine shape of both data sets
print(training_data.shape)
print(dev_data_df.shape)

# Briefly examine feature attributes for the training data 
training_data.describe()


# In[28]:


# Split into data and labels
train_data = training_data.drop(columns=["Cover_Type"])
train_labels = training_data["Cover_Type"]
dev_data = dev_data_df.drop(columns=["Cover_Type"])
dev_labels = dev_data_df["Cover_Type"]
test_data = test_df

# Double check the shape
print(train_data.shape)
print(dev_data.shape)


# #### Scale Data
# Additionally, we will scale the training data to have a mean of 0 and a variance of 1. Then we will retrieve the original training mean and variance for each feature and use that to standardize the development data.

# In[29]:


#compile a list for columns for scaling
ss_cols = ['Elevation','Slope', 'Horizontal_Distance_To_Hydrology',
       'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
       'Horizontal_Distance_To_Fire_Points','elv_pwd']


# In[30]:


# Normalize features using the standard scaler [training data]
def scaler(ss="",cols=ss_cols):
    if ss == "minmax":
        scaler = MinMaxScaler()
    else :
        scaler = StandardScaler()
        model = scaler.fit(train_data[cols])
    train_data[cols] = model.transform(train_data[cols])
    # Normalize features using the standard scaler [dev data]
    dev_data[cols] = model.transform(dev_data[cols])

scaler()


# In[31]:


# Explore and confirm the shape of the data
print("Training data shape: {0} Training labels shape: {1}\n".format(train_data.shape, train_labels.shape))
print("Dev data shape: {0} Dev labels shape: {1}\n".format(dev_data.shape, dev_data.shape))
print("Test data shape: ", test_data.shape)


# ## Models
# #### Random Forest

# In[32]:


# Try a random forest - before any data cleaning 
def RandomForest(num_trees):
    model = RandomForestClassifier(num_trees,max_depth=8)
    model.fit(train_data, train_labels)
    predictions = model.predict(dev_data)
    score = model.score(dev_data, dev_labels)
    probabilities = model.predict_proba(dev_data)
    print("Random Forest Performance for {0} trees: {1}".format(num_trees,score))
    # Plot_confusion_matrix
    plot_confusion_matrix(model, dev_data, dev_labels, values_format = "d")
    plt.title("{} Tree Random Forest Confusion Matrix:".format(num_trees))
    plt.plot()
    mse_forest = mean_squared_error(dev_labels, predictions)
    print("Mean Squared Error: ", mse_forest)
    return score, probabilities
    
num_trees_list = [1,3,5,10,100]
random_forest_results = {}
for num_trees in num_trees_list:
    score, probabilities = RandomForest(num_trees)
    random_forest_results[score] = probabilities


# #### Naive Bayes (Bernoulli)

# In[33]:


# Try Naive Bayes - Bernoulli 
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


# In[34]:


# # Try Naive Bayes - multi-nominal
# def MNB(alf):
#     model = MultinomialNB(alpha = alf)
#     model.fit(train_data, train_labels)
#     predictions = model.predict(dev_data)
#     score = model.score(dev_data, dev_labels)
#     print("Multi NB for alph = {0}: accuracy = {1}".format(alf,score))
#     # Plot Confusion Matrix
#     plot_confusion_matrix(model, dev_data, dev_labels, values_format = "d")
#     plt.title("Multi NB Confusion Matrix with alpha: {}".format(alf))
#     plt.plot()
#     print('\n\n')
    
# # the alpha isn't actually making a difference 
# # alphas_list = [0.00001,0.001, 0.01, 0.1, 1, 10]
# alphas_list = [1.0]
# for alpha in alphas_list:
#     MNB(alpha)


# #### K-Nearest Neighbors

# In[35]:


# Try K Nearest Neighbors - before any data cleaning 
def KNN(kn):
    model = KNeighborsClassifier(n_neighbors = kn)
    model.fit(train_data, train_labels)
    predictions = model.predict(dev_data)
    score = model.score(dev_data, dev_labels)
    print("KNN {0} neighbors : accuracy = {1}".format(kn,score))
    probabilities = model.predict_proba(dev_data)
    # Plot Confusion Matrix
    plot_confusion_matrix(model, dev_data, dev_labels, values_format = "d")
    plt.title("KNN Confusion Matrix with {} Neighbors".format(kn))
    plt.plot()
    mse_knn = mean_squared_error(dev_labels, predictions)
    print("Mean Squared Error: ", mse_knn)
    return score, probabilities
    
# The alpha isn't actually making a difference 
neigh_list = [1,2,4, 7, 10]
knn_results = {}
for neigh in neigh_list:
    score, probabilities = KNN(neigh)
    knn_results[score] = probabilities 


# #### Multi-layer Perceptron

# In[36]:


# Try Multi-Layer Perceptron - before any data cleaning 
def MLP():
    #    model = MLPClassifier(solver='adam', alpha=1e-6, hidden_layer_sizes=(100, ), random_state=0) .8257
    #    model = MLPClassifier(solver='adam', alpha=1e-3, hidden_layer_sizes=(100, ), random_state=0)  .82969
    #    model = MLPClassifier(solver='adam', alpha=1e-3, hidden_layer_sizes=(200, ), random_state=0) .837
    #    model = MLPClassifier(solver='adam', alpha=1e-3, hidden_layer_sizes=(100, ), random_state=0, activation='tanh') .83068

    # Default activation is 'relu', random state lets us get the same result every time (so we can tune other parameters)
    # max_iter is 200 by default, but more helps. alpha is the regularization parameter. solver is 'adam' by default
    model = MLPClassifier(alpha=1e-3, hidden_layer_sizes=(200,), random_state=0, max_iter=300) 
    model.fit(train_data, train_labels) 
    predictions = model.predict(dev_data)
    score = model.score(dev_data, dev_labels)
    probabilities = model.predict_proba(dev_data)
    plot_confusion_matrix(model, dev_data, dev_labels, values_format = "d")
    plt.title("MLP Confusion Matrix")
    plt.plot()
    print("MLP accuracy = ",score)
    mse_nn = mean_squared_error(dev_labels, predictions)
    print("Mean Squared Error: ", mse_nn)
    return score, probabilities

mlp_results = {}
score, probabilities = MLP()
mlp_results[score] = probabilities 


# #### Logistic Regression

# In[37]:


# Logistic regression
def LR():
    model = LogisticRegression(random_state=0, multi_class='ovr',solver='lbfgs', max_iter = 300)
    model.fit(train_data, train_labels)
    score = model.score(dev_data,dev_labels)
    print("Logistic Regression accuracy = ",score)
LR()


# #### Neural Network with Tensorflow

# In[38]:





# In[40]:


tf.executing_eagerly()


# In[57]:


train_data[:1].shape
train_data.to_numpy().shape, train_labels.to_numpy().shape


# In[61]:


model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(52,)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

# Retrieve predictions 
predictions = model(train_data[:1].to_numpy())
# Convert logits to probabilities
tf.nn.softmax(predictions).numpy()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
model.fit(train_data.to_numpy(), train_labels.to_numpy(), epochs=5)


# In[63]:


model.evaluate(dev_data.to_numpy(),  dev_labels.to_numpy(), verbose=2)


# #### Ensemble
# 
# Here we will combine the three best performing models and implement a "voting" system to try to improve accuracy. 

# In[64]:


def Ensemble():
    # Find max score from each model. best_scores shape: (3,)
    best_scores = [max(mlp_results.keys()), max(knn_results.keys()),max(random_forest_results.keys())]
    # Find maximum probability for each example for each model. prediction_probabilities shape: (3024, 3)
    prediction_probabilities = [np.max(mlp_results[best_scores[0]],axis=1),np.max(knn_results[best_scores[1]],axis=1),np.max(random_forest_results[best_scores[2]],axis=1)]
    prediction_probabilities = np.transpose(np.array(prediction_probabilities))
    # Find highest predicted label. predicted_classes shape: (3024, 3)
    predicted_classes = [np.argmax(mlp_results[best_scores[0]],axis=1),np.argmax(knn_results[best_scores[1]],axis=1),np.argmax(random_forest_results[best_scores[2]],axis=1)]
    predicted_classes = np.transpose(np.array(predicted_classes))
    
    # Determine final predictions
    new_predictions = []
    # Keep track of instances in which the models disagree for insight 
    count = 0
    for i, row in enumerate(predicted_classes):
        # Count instances of each class in the predictions
        unique, counts = np.unique(row, return_counts=True)
        zipped = dict(zip(unique, counts))
        # Initialize Classification
        classification = 0
        # If there's only 1 unique value, all models agreed
        if len(unique) == 1:
            classification = unique[0]
        # Two out of three models agreed
        elif len(unique) == 2:
            count += 1
            classification = unique[np.argmax(counts)]
        # All three models disagree. Choose the label with the highest probability 
        else:
            count += 1
            classification = prediction_probabilities[i][0]
        # Assign the new prediction
        new_predictions.append(classification)
    print("Models disagreed on {0}/{1} dev examples.".format(count, dev_labels.shape[0]))
    return predicted_classes, np.array(new_predictions).astype(int)

predicted_classes, new_predictions = Ensemble()
mse_ensemble = mean_squared_error(dev_labels, new_predictions)
accuracy = accuracy_score(dev_labels, new_predictions)
print("Mean Squared Error: ", mse_ensemble)
print("Accuracy: ", accuracy)


# In[65]:


# Examine and Compare Histograms of Predictions
fig, axes = plt.subplots(2,2)
# Ensemble
axes[0,0].hist(new_predictions, bins=7,color = 'red') 
# MLP
axes[0,1].hist(predicted_classes[:,0], bins=7, color = 'orange') 
# KNN
axes[1,0].hist(predicted_classes[:,1], bins=7, color = 'green') 
# Random Forest
axes[1,1].hist(predicted_classes[:,2], bins=7, color = 'blue') 


# ### End matter
# 
# #### Acknowledgements/Sources
# 
# * That helpful stack overflow post
#   * https://stackoverflow.com/questions/28663856/how-to-count-the-occurrence-of-certain-item-in-an-ndarray
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
get_ipython().system('jupyter nbconvert clear_cut_solution.ipynb --to="python" --output="backups/clear-cut-solution"')
get_ipython().system('jupyter nbconvert clear_cut_solution.ipynb --to markdown --output="backups/clear-cut-solution"')

# Also archiving this bad boy
get_ipython().system('jupyter nbconvert clear_cut_solution.ipynb --to html --output="backups/clear-cut-solution"')

