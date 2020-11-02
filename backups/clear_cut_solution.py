#!/usr/bin/env python
# coding: utf-8

# # Forest Cover Type Prediction
# #### Team: Clear-Cut Solution: Kevin Martin, Yang Jing, Justine Schabel

# TODO: Introduce the project 

# ## Initial Setup
# #### Import Required Libraries

# In[31]:


# This tells matplotlib not to try opening a new window for each plot.
get_ipython().run_line_magic('matplotlib', 'inline')

# Libraries for reading, cleaning and plotting the dataa
import pandas as pd 
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import seaborn as sns

# Feature Engineering was written by the team 
import feature_engineering as fe
# Models was written by the team 
import models 

from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

# So we can reload packages without restarting the kernel
import importlib


# In[32]:


# If you update the feature_engineering package, run this line so it updates without needing to restart the kernel
importlib.reload(fe)


# In[33]:


# If you update the models package, run this line so it updates without needing to restart the kernel
importlib.reload(models)


# #### Load Data

# In[34]:


# Read in training data 
train_df = pd.read_csv("data/train.csv")


# ## Feature Engineering 
# The following transformations were made in the function below. 
# 
# #### Transform Hillshade
# Now we'll normalize the "Hillsdale" variables by dividing them by 255. The hillshade variables contain index of shades with a value between 0 and 255. 
# 
# #### Create new soil types
# Now we'll create additional features to magnify the differences betweeen cover type1 and 2, and covery type3 and 6.
# 
# #### Combine soil types 
# 
# #### Drop rare or non-existant soil types 
# Now we'll drop soil types that don't exist in the training set. Then we will combine soil types 35, 38, 39 and 40 because they have a very similar distribution. 
# 
# #### Create new features based on soil type descriptions 
# TODO: Explain how we split up soil descriptions into different features to account for overlap.
# 
# #### Transform Aspect
# TODO: Explain aspect problem and solution
# 
# #### Log transformations
# TODO: Now we'll log transform the features related to the distances. (explain why)
# 
# #### Add polynomial features
# TODO: Explain why we're making Elevation polynomial
# 
# #### Drop irrelevant or problematic features
# - We'll drop "Id" because it does not provide any meaning in the classifications.
# - We'll drop "Hillshade_9am" because it has a high correlation with "Aspect" and "Hillshade_3pm".
# - TODO: We'll also drop "Vertical_Distance_To_Hydrology" because _.
# 

# In[35]:


def manipulate_data(data):
    """ 
    This function applys transformations on the input data set
    
    Parameters: 
        data (dataframe): n_examples x m_features (int64) dataframe 
    """
    data = fe.scale_hillside(data)
    
    # Soil Combination One (based on distributions)
    data = fe.combine_environment_features(data)
    data = fe.drop_unseen_soil_types(data)  
    data = fe.combine_soil_types(data)
      
    
    # Soil Combination Two (based on descriptions)
    # data = fe.set_soil_type_by_attributes(data)
    
    data = fe.transform_aspect(data)
    
    features_to_log = ['Horizontal_Distance_To_Hydrology',
           'Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points']
    data = fe.log_features(data, features_to_log)
    
    features_to_square = ["Elevation"]
    data = fe.add_polynomial_features(data, features_to_square)

    # These are already being dropped by now? 
    features_to_drop = ["Id","Hillshade_9am","Vertical_Distance_To_Hydrology"]
    data = fe.drop_features(data, features_to_drop)
    
    return data

train_df = manipulate_data(train_df)


# #### Examine transformed data

# In[36]:


train_df.describe()


# Now that the data is transformed, we can also visualize the new aspect features. 

# In[37]:


# Visualize cover type VS the cosine of Aspect degerees
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(12,4))
sns.violinplot(x=train_df['Cover_Type'],y=train_df['ap_ew'],ax=ax1)
sns.histplot(train_df['ap_ew'],ax=ax2)
plt.show()


# After the feature transformation, we see improved distinction in median values, espeically for cover type 6, where the median is notably higher than that of other cover types and the distribution is concentrated around the median.

# #### Split data into train/dev
# 
# Then, we split the training data into a training data set (80%) and development data set (20%). We will also have a large, separate test data set. 

# In[38]:


train_data, train_labels, dev_data, dev_labels = fe.split_data(train_df)


# #### Scale the data to have a mean of 0 and a variance of 1.

# In[39]:


standardize_features = ['Elevation','Slope', 'Horizontal_Distance_To_Hydrology',
       'Horizontal_Distance_To_Roadways',
       'Horizontal_Distance_To_Fire_Points','Elevation_squared']
train_data, train_scaler = fe.scale_training_data(standardize_features, train_data, scaler_type="standard")
dev_data = fe.scale_non_training_data(standardize_features, dev_data, train_scaler)


# #### Explore and confirm the shape of the data

# In[40]:


print("Training data shape: {0} Training labels shape: {1}\n".format(train_data.shape, train_labels.shape))
print("Dev data shape: {0} Dev labels shape: {1}\n".format(dev_data.shape, dev_labels.shape))


# ## Models

# #### Random Forest

# In[41]:


num_trees_list = [1,3,5,10,100]
random_forest_models = []
random_forest_results = {}
for num_trees in num_trees_list:
    score, probabilities, random_forest_model = models.random_forest(num_trees, train_data, train_labels, dev_data, dev_labels)
    random_forest_results[score] = probabilities
    random_forest_models.append(random_forest_model)


# #### K-Nearest Neighbors

# In[42]:


neighbor_list = [1,2,4, 7, 10]
knn_models = []
knn_results = {}
for neighbor in neighbor_list:
    score, probabilities, knn_model = models.k_nearest_neighbors(neighbor,train_data, train_labels, dev_data, dev_labels)
    knn_results[score] = probabilities
    knn_models.append(knn_model)
    


# #### Multi-Layer Perceptron

# In[43]:


mlp_results = {}
score, probabilities,mlp_model = models.multi_layer_perceptron(train_data, train_labels, dev_data, dev_labels)
mlp_results[score] = probabilities 


# #### Logistic Regression

# In[44]:


models.logistic_regression(train_data, train_labels, dev_data, dev_labels)


# #### Neural Network 

# In[45]:


models.neural_network(train_data, train_labels, dev_data, dev_labels)


# #### Ensemble
# Here we will combine the three best performing models and implement a "voting" system to try to improve accuracy.

# In[46]:


predicted_classes, new_predictions = models.ensemble(mlp_results,knn_results,random_forest_results, dev_labels)
mse_ensemble = mean_squared_error(dev_labels, new_predictions)
accuracy = accuracy_score(dev_labels, new_predictions)
print("Mean Squared Error: ", mse_ensemble)
print("Accuracy: ", accuracy)


# #### Examine and Compare Histograms of Predictions

# In[47]:


fig, axes = plt.subplots(2,2)
# Ensemble
axes[0,0].hist(new_predictions, bins=7,color = 'red') 
# MLP
axes[0,1].hist(predicted_classes[:,0], bins=7, color = 'orange') 
# KNN
axes[1,0].hist(predicted_classes[:,1], bins=7, color = 'green') 
# Random Forest
axes[1,1].hist(predicted_classes[:,2], bins=7, color = 'blue') 


# ## Test Results
# #### Read in test data

# In[48]:


# Read in training data 
test_data = pd.read_csv("data/test.csv")
# Preserve testing df ID for submission purpose
test_df_ID = test_data["Id"]


# #### Apply the same transformations

# In[49]:


test_data = manipulate_data(test_data)
test_data = fe.scale_non_training_data(standardize_features, test_data, train_scaler)


# In[ ]:


random_forest_predictions = random_forest_models[-1].predict(test_data)
knn_predictions = knn_models[0].predict(test_data)
mlp_predictions = mlp_model.predict(test_data)


# #### Generate Submission File

# In[ ]:


def gen_submission(y_pred,model):
    result = pd.DataFrame.from_dict(dict(zip(test_df_ID.to_list(),y_pred)), orient='index', columns=["Cover_Type"])
    result.to_csv(f"submissions/submission{model}.csv",index_label="Id")

gen_submission(random_forest_predictions, model="RandomForest")
gen_submission(knn_predictions, model="KNN")
gen_submission(mlp_predictions, model="MLP")


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
get_ipython().system('jupyter nbconvert clear_cut_solution.ipynb --to="python" --output="backups/clear_cut_solution"')
get_ipython().system('jupyter nbconvert clear_cut_solution.ipynb --to markdown --output="backups/clear_cut_solution"')

# Also archiving this bad boy
get_ipython().system('jupyter nbconvert clear_cut_solution.ipynb --to html --output="backups/clear_cut_solution"')

