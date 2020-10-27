#!/usr/bin/env python
# coding: utf-8

# # Forest Cover Type Prediction
# #### Team: Clear-Cut Solution: Kevin Martin, Yang Jing, Justine Schabel

# In[1]:


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


# ## Data Engineering

# ### Load Data

# In[2]:


# Read in training data 
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")


# ### Initial Data Exploration
# 
# First, we check the data attributes, quality and shape.

# In[3]:


# Examine shape 
print(train_df.shape)

# Briefly examine feature attributes for the training data 
train_df.describe()


# In[4]:


sns.distplot(train_df['Cover_Type'],rug=True)
plt.show()


# Here we can see that the training data has a somewhat uniform distribution of covertype and this tells us that our data set is balanced. 

# In[5]:


sns.violinplot(x=train_df['Cover_Type'],y=train_df['Elevation'])
plt.show()


# Here, we can see there is a relationship between the cover type and elevation. 

# In[6]:


# get NA values

print("There are {} NA values in the training data".format(train_df.isna().sum().sum()))
print("There are {} NA values in the test data".format(train_df.isna().sum().sum()))
    # `.isna()` returns a df with bools the first `.sum()` returns series, second is int 
print()
print("There are {} values in the training data".format(train_df.count()[0]))
print("There are {} values in the test data".format(test_df.count()[0]))


# No null values in the dataset. Also noted the "aspect" variable has a value between 0 and 359. This is expressed in degrees, compared to "true north". Will conver this ino sine(EW) and cosine(NS) values. 

# ### Feature Engineering 1
# Now we'll transform the "Aspect" into cosine and sine values to improve the representation of directions. 

# In[7]:


#split the aspect into a N/S and E/W unit vector
train_df["asp_ew"] = np.sin(train_df["Aspect"])
train_df["asp_ns"] = np.cos(train_df["Aspect"])


train_df["asp_ew"]**2 + train_df["asp_ns"]**2

train_df.hist(column = "asp_ew")
train_df.hist(column = "asp_ns")
train_df.hist(column = "Aspect")

df_circle = train_df[["asp_ew","asp_ns"]]
df_circle["jitter"] = np.random.rand(train_df.shape[0])*.6 +1
df_circle["asp_ew_jit"] = df_circle["asp_ew"] * df_circle["jitter"]
df_circle["asp_ns_jit"] = df_circle["asp_ns"] * df_circle["jitter"]

df_circle.plot.scatter("asp_ew_jit","asp_ns_jit",alpha=0.04)
df_circle.head()


# In[8]:


#drop Aspect column
train_df.drop(columns=["Aspect"], inplace=True)


# Now, we'll isolate and explore the distribution of soil types. 

# In[9]:


# Isolate soil type column names
soil_df = train_df[["Id",'Soil_Type1', 'Soil_Type2', 'Soil_Type3',
       'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8',
       'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12',
       'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16',
       'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',
       'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24',
       'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28',
       'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32',
       'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',
       'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40']]

# Now we convert the soil type columns back into one column with values as the "soil type"
soil_df_unpivoted = soil_df.melt(id_vars="Id",var_name="soil_type",value_name="yes")
mask1 = soil_df_unpivoted["yes"] ==1 #only keep rows of where the "soil type" is "yes"
soil_df_unpivoted = soil_df_unpivoted[mask1]


# Examine the fequencies of soil types
soil_df_unpivoted["soil_type"].value_counts().to_frame() 

# Histogram of soil types 
plt.figure(figsize=(24,6))
plt.hist(soil_df_unpivoted["soil_type"],bins=40)
plt.xticks(rotation=90)
plt.show()


# As we can see in the histogram above, there is an uneven distribution of occurances of soil types.

# In[10]:


# Explore correlations between features
train_corr=train_df.corr()
# Rank correlations with "cover type"
train_corr['Cover_Type'].abs().sort_values(ascending=False)


# In[11]:


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

# Now, we'll isolate and explore the distribution of wilderness types. 

# In[12]:


wilderness_list =['Wilderness_Area1','Wilderness_Area2','Wilderness_Area3','Wilderness_Area4']

# Visualize the distribution of wilderness area and "cover type"
fig, axes = plt.subplots(2,2,figsize=(24,12))
for i in range(4):
    sns.violinplot(y=train_df['Cover_Type'],x=train_df[wilderness_list[i]], ax=axes[i//2,i%2])
plt.show()


# ### Feature Engineering 2
# 
# I'm going to hold off on dropping any soil types and just transform them to the new type

# In[13]:


# # Remove soil type 7 and 15 due to no data
# train_df.drop(columns=["Soil_Type7", "Soil_Type15"], inplace=True)

# # Remove soil type 19, 37, 34, 21, 27,36,9, 28,8,25 due to no limited data - TODO: should we be dropping these? 
# train_df.drop(columns=["Soil_Type19", "Soil_Type37","Soil_Type34", "Soil_Type21","Soil_Type27", "Soil_Type36","Soil_Type9", "Soil_Type28","Soil_Type8", "Soil_Type25"], inplace=True)

# # Combine soil type 35,38,39, 40
# train_df["soil_type35383940"] = train_df["Soil_Type38"] +  train_df["Soil_Type39"] + train_df["Soil_Type40"] +  train_df["Soil_Type35"]
# train_df.drop(columns=["Soil_Type35","Soil_Type38", "Soil_Type39",'Soil_Type40'], inplace=True)

# # Check shape is as expected
# print(train_df.shape)


# In[14]:


#drop Id column as it is not a meaningful feature.
train_df.drop(columns=["Id"],inplace=True)
test_df.drop(columns=["Id"],inplace=True)


# ## Soil Type
# 
# Right now we have 40 composite types of soil. We are going to re-categorize those into their component parts with a 1 for if a component is present. 0 if they are not present. 
# 
# #### Adjustments to the data 
# 
# Looking throught the data, we see several areas where the data can be cleaned. The USDA which is the cabinet department which oversees the forest service, publishes a guide on Soil Taxonomy. (https://www.nrcs.usda.gov/wps/portal/nrcs/detail/soils/home/?cid=nrcs142p2_053577). This guide was very helpful in interpreting several areas where the data can be trimmed down. 
# 
# * Filler words
#   * "Family / "families" denotes a larger soil group such as Leighcan or Cryaquolls, but these words themselves don't actually carry any information about the group. 
#   * "complex" is another word that appears in the docuement but doesn't actually convey any information itself
#   * "typic" just means that a soil fits into the most general subgroup of the family listed. In the context of the soil types listed, it is useless because only a few soil types list it, and those that don't list it don't list a more specific subgroup that they are a part of. So, they would also generally be assumed to be typic (see guide to soil taxonomy linked above.)
# * Cryaquolis and aquolis types don't exist
#   * Looking through the guide to soil taxonomy, there is no such type as cryaquolis and aquolis. The simplest interpretation is that these are typos. Cryaquolls and aquolls do exist (see link above). I will change the words accordingly.
#   
# #### Possible future work
# 
# * See if the "very/extremely" qualifiers on stony should be removed. Do they mean the same thing?
# * See if changing "rubbly", "bouldery", "stoney" to all one indicator would be better. Do they all mean the same thing?

# In[30]:


#### Analyze frequqncy that imporant types of soil show up ####

import re  #regexes to work with strings
from sklearn.feature_extraction.text import CountVectorizer #count up occurences of words
import numpy as np

# pull in the text of the soil types
with open("km_EDA/soil_raw.txt") as f:
    s_raw = f.read()

# lowercase everything to make it easier to work with
s = s_raw.lower()

# take out punctuation and numbers.
pattern = re.compile(r"[\d\.,-]")
s = pattern.sub(" ",s)

# take out filler words "family","families","complex"
pattern = re.compile(r"(family|families|complex|typic)")
s = pattern.sub(" ",s)

# replace cryaquolis/aquolis (doesn't exist) with cryaquolls/aquolls
pattern = re.compile(r"aquolis")
s = pattern.sub("aquolls",s)

# the "unspecified" row doesn't contain any data
pattern = re.compile(r"unspecified in the usfs soil and elu survey")
s = pattern.sub(" ",s)

#replace the space in words separated by a single space with an underscore
pattern = re.compile(r"(\w+) (\w+)")
s = pattern.sub(r"\1_\2",s)



### COUNT AND TRANSFORM THE DATA ###
cv = CountVectorizer()
# create the counts matrix based on word occurences in our processed soil types
counts = cv.fit_transform(s.split("\n"))
# we can use the counts as a transformation matrix to convert to our refined categories
xform = counts.toarray()

## Explanation of xform
    # It turns out that multiplying our original soil matrix by xform using matrix mutiplication
    # will just give us a matrix that has been converted to the new feature space. 

# Grab out the new features (that are replacing s_01 thru s_40)
new_cats = cv.get_feature_names()
print("-- New Feature Names --")
print(new_cats,"\n")

# get original soil names
og_soil_col_names = [("Soil_Type{:d}".format(ii+1)) for ii in range(40)]

# get columns containing soil information from our dataframe.
soil_cols = np.array(train_df[og_soil_col_names])

# transform the soil features. Put them into a dataframe.
trans_soil = np.matmul(soil_cols,xform)
trans_soil_df = pd.DataFrame(data = trans_soil, columns = new_cats)
display(trans_soil_df)

# combine the new soil features with the existing freatures in a single df
df_new = train_df.drop(columns=og_soil_col_names)
df_new = pd.concat([df_new, trans_soil_df],axis=1)
df_new = df_new[[col for col in df_new if col not in ["Cover_Type"]]+["Cover_Type"]] #want cover type as last column
display(df_new)

# We'll just reassign the train_df here to be equal to df_new
train_df = df_new


# ### Additional Data Mungling
# 
# Then, we split the training data into a training data set (80%) and development data set (20%). We will also have a large, separate test data set. 

# In[31]:


# Split training data (labeled) into 80% training and 20% dev) and randomly sample 
training_data = train_df.sample(frac=0.8)
dev_data_df = train_df.drop(training_data.index)

# Examine shape of both data sets
print(training_data.shape)
print(dev_data_df.shape)

# Briefly examine feature attributes for the training data 
display(training_data.describe())
display(dev_data_df.describe())


# Additionally, we will scale the training data to have a mean of 0 and a variance of 1. Then we will retrieve the original training mean and variance for each feature and use that to standardize the development data.

# In[32]:


# Split into data and labels
train_data = training_data.drop(columns=["Cover_Type"])
train_labels = training_data["Cover_Type"]
dev_data = dev_data_df.drop(columns=["Cover_Type"])
dev_labels = dev_data_df["Cover_Type"]
test_data = test_df

# Double check the shape
print(train_data.shape)
print(dev_data.shape)


# In[33]:


train_data.columns


# In[34]:


# Collect numeric feature column names - so we can easily access these columns when modifying them 
num_cols = ['Elevation', 'Slope',
       'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
       'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon',
       'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points']

# Normalize features using the standard scaler [training data]
scaler = StandardScaler()
norm = scaler.fit(train_data[num_cols])
train_data[num_cols] = norm.transform(train_data[num_cols])
print(train_data.shape)
# Normalize features using the standard scaler [dev data]
dev_data[num_cols] = norm.transform(dev_data[num_cols])
print(dev_data.shape)


# In[35]:


# Double check shape
print(train_data.shape, dev_data.shape)


# In[36]:


# Explore and confirm the shape of the data
print("Training data shape: {0} Training labels shape: {1}".format(train_data.shape, train_labels.shape))
print("Dev data shape: {0} Dev labels shape: {1}".format(dev_data.shape, dev_data.shape))
print("Test data shape: ", test_data.shape)


# In[37]:


# Examine Training Data 
dev_data.head()


# ## Models
# #### Random Forest

# In[38]:


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

# In[39]:


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

# In[40]:


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

# In[41]:


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

# In[27]:


#Create a backup of the jupyter notebook in a format for where changes are easier to see.
get_ipython().system('jupyter nbconvert clear-cut-solution.ipynb --to="python" --output="backups/clear-cut-solution"')
get_ipython().system('jupyter nbconvert clear-cut-solution.ipynb --to markdown --output="backups/clear-cut-solution"')

# Also archiving this bad boy
get_ipython().system('jupyter nbconvert clear-cut-solution.ipynb --to html --output="backups/clear-cut-solution"')


# In[ ]:





# In[ ]:




