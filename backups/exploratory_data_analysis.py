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
import warnings
warnings.simplefilter("ignore")


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
df1 = df1.pivot("soil_type","Cover_Type","yes")
df1


# As we can see in the pivot table above, there are similar combinations of soil types for different "cover type". We'll combine the soil types that share same "cover types" to reduce dimensionality. Further, "cover type 1" and "cover type 2" , "cover type 3" and "cover type 6" share many overlapping features. To magnify the signal, we'll combine features as an extra feature where there is a difference between the 2 pairs of cover types.

# In[24]:


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

# In[18]:


#Create a backup of the jupyter notebook in a format for where changes are easier to see.
get_ipython().system('jupyter nbconvert exploratory_data_analysis.ipynb --to="python" --output="backups/exploratory_data_analysis"')
get_ipython().system('jupyter nbconvert exploratory_data_analysis.ipynb --to markdown --output="backups/exploratory_data_analysis"')

# Also archiving this bad boy
get_ipython().system('jupyter nbconvert exploratory_data_analysis.ipynb --to html --output="backups/exploratory_data_analysis"')

