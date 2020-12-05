# Forest Cover Type Prediction
#### Team: Clear-Cut Solution: Kevin Martin, Yang Jing, Justine Schabel

## Introduction

This project was from a Kaggle competion closed in 2014. 
[Link to Data page on Kaggle](https://www.kaggle.com/c/forest-cover-type-prediction/data)

The objective is to classify trees from four wilderness areas located in the Roosevelt National Forest of northern Colorado. Each observation is a 30m x 30m patch. The "labels" are 7 "cover types". They are: 

1. Spruce/Fir
2. Lodgepole Pine
3. Ponderosa Pine
4. Cottonwood/Willow
5. Aspen
6. Douglas-fir
7. Krummholz

The training set (15120 observations) contains both features and the labels. The test set contains only the features. Compared to the training set, the test set is very large with 565892 observations. 

There are 54 features, include 40 soil types (binary), 4 wilderness ares (binary), and 10 numerical features, such as Elevation, slope, aspects and distances to certain points.

We quickly noticed that the training data size is less than 3% of the testign data. We might not have enough diversites in the training data to predict the labels for testign data. We also noticed that 100% accuracy was achieved in this Kaggle competion previously, but that appeared to due to a "cheat" already known to the Kaggle team. As such, our objective is not to obsess over the accuracy of the "testing data", but to explore which classficiation algorithm and/or process works best for the dataset. In addition, some "cover types" appear to be easy to confuse, such as "spruce/fir" and "lodgpole pine". We confirmed this suspicsion by performing exploratory data analysis first. Our EDA is documented in the "explanatory_data_analysis" notebook. To imporve model performance, we need to find a way to bettr differentiate between type 1 and 2. 

Due to the perceived higher confusion rate between type 1 ad type 2 trees from the EDA, and between type 3 and 6 trees, we performed a separate training on these 2 subsets. We also performed other feature engineering steps, which include scaling numerical features, power-transfomration of "elevation" and dropping some less frequent soil types or "distance" related features. In addition, dFor better organizaiton of the code, we saved all the feature engineering functions under a separate module, called "feature_engineering.py".

In addition, to make the computing environment portable and enable collaboration among team members, we created a docker container based on the image "jupyter/tensorflow-notebook". Details about the container is documented in the "readme.md" file. 

This noteboook documents the model development, evluation and final testing process. In each section, we included a flowchart to summarize the feature engineering applied and modeling processes. We experimented with different algorithms, including KNN, Random Forest, Logistic Regression, Adaboost, Gradient Boosting(XGBoost), and neural networks(MLP). Without separate training, the accuracy for "dev data" from both MLP and XGBoost was already in 84%-86% range. With separately trained sub-models, the accuracy for "dev data" from both MLP and XGBoost models was above 90%. However, when applied to the "testing data", the accuracy dropped to 74%. We suspect that it's mainly due to the small training data size,and the model is overfitting. To include more observations in the training data, we used 90/10 split instead of 80/20. And, to counter the impact of overfitting, we used "max_depth=1" in training the Random Forest model. While this decreased the Randome Forest model accuracy for the dev data to 40%, it works well in our manual ensemble to serve as a disserting vote. Our final model is based on a manual ensemble where we take the majority vote of the predictions from the top 5 performing algorithms (KNN, Random Forest, Adaboost, XGBoost and MLP) for this use case. Where there is a tie, we took the prediction from the XGboost model. This manual ensemble model is documented in the "model.py" file.  In the end, we achieved over 94% accuracy on the dev data and over 76% accuracy on the testing data. 


## Initial Setup
#### Import Required Libraries


```python
# Surpress warning messages
import warnings
warnings.filterwarnings("ignore")

# This tells matplotlib not to try opening a new window for each plot.
%matplotlib inline

# Libraries for reading, cleaning and plotting the dataa
import pandas as pd 
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import seaborn as sns
import numpy as np

# Feature Engineering was written by the team 
import feature_engineering as fe

# Models was written by the team 
import models 

from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import plot_confusion_matrix
import xgboost as xg


# So we can reload packages without restarting the kernel
import importlib
```


```python
# If you update the feature_engineering package, run this line so it updates without needing to restart the kernel
importlib.reload(fe)
```




    <module 'feature_engineering' from '/home/jupyter/mids-w207-final-project/feature_engineering.py'>




```python
# Run this cell to intal xgboost if you are using the container.
!pip install xgboost
```

    Requirement already satisfied: xgboost in /opt/conda/lib/python3.7/site-packages (1.2.1)
    Requirement already satisfied: numpy in /opt/conda/lib/python3.7/site-packages (from xgboost) (1.18.5)
    Requirement already satisfied: scipy in /opt/conda/lib/python3.7/site-packages (from xgboost) (1.5.3)


#### Load Data


```python
# Read in training data 
train_df = pd.read_csv("data/train.csv")
```

## Feature Engineering 

Overall Data Pipeline
<img src="data/data_pipeline.png">


The following transformations were made in the function below. 

#### Transform Hillshade
Now we'll normalize the "Hillsdale" variables by dividing them by 255. The hillshade variables contain index of shades with a value between 0 and 255. 

#### Create new soil types
Now we'll create additional features to magnify the differences betweeen cover type1 and 2, and covery type3 and 6.

#### Combine soil types 

#### Drop rare or non-existant soil types 
Now we'll drop soil types that don't exist in the training set. Then we will combine soil types 35, 38, 39 and 40 because they have a very similar distribution. 

#### Create new features based on soil type descriptions 
TODO: Explain how we split up soil descriptions into different features to account for overlap.

#### Transform Aspect
TODO: Explain aspect problem and solution

#### Log transformations
Per EDA, we noticed the distribution of the "distance" related variables are skewed. Now we'll log transform the features related to the distances to make the distribution smoother, and thus decrease the variances of the predictions.

#### Add polynomial features
Per EDA, Elevation is a numerical variable and there is a clearer distinciton in Elevation among the dependenet variable, cover type. To imporve the signal, we sqaured Elevation. 

#### Drop irrelevant or problematic features
- We'll drop "Id" because it does not provide any meaning in the classifications.
- We'll drop "Hillshade_9am" because it has a high correlation with "Aspect" and "Hillshade_3pm".
- We'll also drop "Vertical_Distance_To_Hydrology" because it does not show much distinction among the "Cover Types" and has a very skewed distribution, with negative values in some cases. The variable offers little insight and there might be data issues in this variable. 



```python
# Apply (data independent) feature engineering to entire dataset 
train_df  = fe.manipulate_data(train_df)
# Examine transformed data
train_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Elevation</th>
      <th>Slope</th>
      <th>Horizontal_Distance_To_Hydrology</th>
      <th>Horizontal_Distance_To_Roadways</th>
      <th>Hillshade_Noon</th>
      <th>Hillshade_3pm</th>
      <th>Horizontal_Distance_To_Fire_Points</th>
      <th>Wilderness_Area1</th>
      <th>Wilderness_Area2</th>
      <th>Wilderness_Area3</th>
      <th>...</th>
      <th>Cover_Type</th>
      <th>soil_type35383940</th>
      <th>st10111617</th>
      <th>st912</th>
      <th>st3133</th>
      <th>st2324</th>
      <th>st6w4</th>
      <th>ap_ew</th>
      <th>ap_ns</th>
      <th>Elevation_squared</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2596</td>
      <td>3</td>
      <td>5.556828</td>
      <td>6.236370</td>
      <td>0.909804</td>
      <td>0.580392</td>
      <td>8.745125</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.777146</td>
      <td>0.629320</td>
      <td>6739216</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2590</td>
      <td>2</td>
      <td>5.361292</td>
      <td>5.968708</td>
      <td>0.921569</td>
      <td>0.592157</td>
      <td>8.736489</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.829038</td>
      <td>0.559193</td>
      <td>6708100</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2804</td>
      <td>9</td>
      <td>5.594711</td>
      <td>8.064951</td>
      <td>0.933333</td>
      <td>0.529412</td>
      <td>8.719644</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.656059</td>
      <td>-0.754710</td>
      <td>7862416</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2785</td>
      <td>18</td>
      <td>5.493061</td>
      <td>8.036250</td>
      <td>0.933333</td>
      <td>0.478431</td>
      <td>8.734238</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.422618</td>
      <td>-0.906308</td>
      <td>7756225</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2595</td>
      <td>2</td>
      <td>5.036953</td>
      <td>5.971262</td>
      <td>0.917647</td>
      <td>0.588235</td>
      <td>8.727940</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.707107</td>
      <td>0.707107</td>
      <td>6734025</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 55 columns</p>
</div>



Now that the data is transformed, we can also visualize the new aspect features. 


```python
# Visualize cover type VS the cosine of Aspect degerees
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(12,4))
sns.violinplot(x=train_df['Cover_Type'],y=train_df['ap_ns'],ax=ax1)
sns.histplot(train_df['ap_ns'],ax=ax2)
plt.show()
```


![png](backups/clear_cut_solution_files/backups/clear_cut_solution_12_0.png)


After the feature transformation, we see improved distinction in median values, espeically for cover type 6, where the median is notably higher than that of other cover types and the distribution is concentrated around the median.

#### Segment data

Based on closer examination of our model performance, we found that the models consistently confused cover types 1 and 2 and covertypes 3 and 6. So we decided to break up our model into one primary model (outputs 12 (for 1 or 2), 36 (for outputs 3 or 6), 4, 5 or 7) and two secondary models (one for distinguishing 1 or 2 and the other for distinguishing 3 or 6). 


```python
# Split trainning data into subsets to improve model accurracy with cover type 1&2, and 3&6.
train_df_original, train_df_12_36_4_5_7, train_df_12, train_df_36 = fe.subset_data(train_df)

# Apply specific transformations for cover types one and two and three and six
train_df_12 = fe.combine_environment_features_ct12(train_df_12)
train_df_36 = fe.combine_environment_features_ct36(train_df_36)
```

#### Split data into train/dev

Then, we split the training data into a training data set (90%) and development data set (10%). We will also have a large, separate test data set. 


```python
# Only (randomly) sample indicies once for the entire dataset 
# This will be very important for comparing the output to the original dev labels 
train_indicies_all_data = list(train_df_original.sample(frac=0.9).index)

# Split each dataset into 80% train and 20% dev by randomly sampling indicies 
train_data_original, train_labels_original, dev_data_original, dev_labels_original = fe.split_data(train_df_original,train_indicies_all_data)
train_data_12_36_4_5_7, train_labels_12_36_4_5_7, dev_data_12_36_4_5_7, dev_labels_12_36_4_5_7 = fe.split_data(train_df_12_36_4_5_7,train_indicies_all_data)
train_data_cover_type_12, train_labels_cover_type_12, dev_data_cover_type_12, dev_labels_cover_type_12  = fe.split_data(train_df_12,list(train_df_12.sample(frac=0.8).index))
train_data_cover_type_36, train_labels_cover_type_36, dev_data_cover_type_36, dev_labels_cover_type_36  = fe.split_data(train_df_36,list(train_df_36.sample(frac=0.8).index))
```

#### Scale the data to have a mean of 0 and a variance of 1.


```python
standardize_features = ['Elevation','Slope', 'Horizontal_Distance_To_Hydrology',
       'Horizontal_Distance_To_Roadways',
       'Horizontal_Distance_To_Fire_Points','Elevation_squared']

# Retrieve scaler *once* for the original training data so we don't overfit the smaller datasets 
train_data_original, train_data_original_scaler = fe.scale_training_data(standardize_features, train_data_original, scaler_type="standard")
dev_data_original = fe.scale_non_training_data(standardize_features, dev_data_original, train_data_original_scaler)

train_data_12_36_4_5_7 = fe.scale_non_training_data(standardize_features, train_data_12_36_4_5_7, train_data_original_scaler)
dev_data_12_36_4_5_7 = fe.scale_non_training_data(standardize_features, dev_data_12_36_4_5_7, train_data_original_scaler)

train_data_cover_type_12 = fe.scale_non_training_data(standardize_features, train_data_cover_type_12, train_data_original_scaler)
dev_data_cover_type_12 = fe.scale_non_training_data(standardize_features, dev_data_cover_type_12, train_data_original_scaler)

train_data_cover_type_36 = fe.scale_non_training_data(standardize_features, train_data_cover_type_36, train_data_original_scaler)
dev_data_cover_type_36 = fe.scale_non_training_data(standardize_features, dev_data_cover_type_36, train_data_original_scaler)
```

#### Explore and confirm the shape of the data


```python
print("Original Data with Labels 1,2,3,4,5,6,7")
print("Training data shape: {0} Training labels shape: {1}".format(train_data_original.shape, train_labels_original.shape))
print("Dev data shape: {0} Dev labels shape: {1}\n".format(dev_data_original.shape, dev_labels_original.shape))

print("Data with Labels 12,36,4,5,7")
print("Training data shape: {0} Training labels shape: {1}".format(train_data_12_36_4_5_7.shape, train_labels_12_36_4_5_7.shape))
print("Dev data shape: {0} Dev labels shape: {1}\n".format(dev_data_12_36_4_5_7.shape, dev_labels_12_36_4_5_7.shape))

print("Data with Labels 1,2")
print("Training data shape: {0} Training labels shape: {1}".format(train_data_cover_type_12.shape, train_labels_cover_type_12.shape))
print("Dev data shape: {0} Dev labels shape: {1}\n".format(dev_data_cover_type_12.shape, dev_labels_cover_type_12.shape))

print("Data with Labels 3,6")
print("Training data shape: {0} Training labels shape: {1}".format(train_data_cover_type_36.shape, train_labels_cover_type_36.shape))
print("Dev data shape: {0} Dev labels shape: {1}\n".format(dev_data_cover_type_36.shape, dev_labels_cover_type_36.shape))
```

    Original Data with Labels 1,2,3,4,5,6,7
    Training data shape: (13608, 54) Training labels shape: (13608,)
    Dev data shape: (1512, 54) Dev labels shape: (1512,)
    
    Data with Labels 12,36,4,5,7
    Training data shape: (13608, 54) Training labels shape: (13608,)
    Dev data shape: (1512, 54) Dev labels shape: (1512,)
    
    Data with Labels 1,2
    Training data shape: (3456, 56) Training labels shape: (3456,)
    Dev data shape: (864, 56) Dev labels shape: (864,)
    
    Data with Labels 3,6
    Training data shape: (3456, 55) Training labels shape: (3456,)
    Dev data shape: (864, 55) Dev labels shape: (864,)
    


## Models

Fit Random Forest, K Nearest Neighbors and Multilayer Perceptron models to the training data
<img src="data/training_models.png">

#### Random Forest


```python
from sklearn.ensemble import RandomForestClassifier

# Set model parameters
num_trees = 100
max_depth = 1

# Fit a Random Forest Model on the original dataset
forest_model_original = RandomForestClassifier(num_trees, max_depth=max_depth,max_features="auto",random_state=1)
forest_model_original.fit(train_data_original, train_labels_original)

# Fit a Random Forest Model to data with labels 12, 36, 4, 5, 7
forest_model_all_data = RandomForestClassifier(num_trees, max_depth=max_depth,max_features="auto",random_state=1)
forest_model_all_data.fit(train_data_12_36_4_5_7, train_labels_12_36_4_5_7)

# Fit a Random Forest Model to differentiate cover types one and two
forest_model_cover_type_12 = RandomForestClassifier(num_trees, max_depth=max_depth,max_features="auto",random_state=1)
forest_model_cover_type_12.fit(train_data_cover_type_12, train_labels_cover_type_12)

# Fit a Random Forest Model to differentiate cover types three and six
forest_model_cover_type_36 = RandomForestClassifier(num_trees, max_depth=max_depth,max_features="auto",random_state=1)
forest_model_cover_type_36.fit(train_data_cover_type_36, train_labels_cover_type_36)
```




    RandomForestClassifier(max_depth=1, random_state=1)



#### K-Nearest Neighbors


```python
from sklearn.neighbors import KNeighborsClassifier

# Set model parameters
num_neighbors = 1

# Fit a KNN  Model on the original dataset
knn_original = KNeighborsClassifier(num_neighbors)
knn_original.fit(train_data_original,train_labels_original)

# Fit a KNN Model to data with labels 12, 36, 4, 5, 7
knn_model_all_data = KNeighborsClassifier(num_neighbors)
knn_model_all_data.fit(train_data_12_36_4_5_7, train_labels_12_36_4_5_7)

# Fit a KNN Model to differentiate cover types one and two
knn_model_cover_type_12 = KNeighborsClassifier(num_neighbors)
knn_model_cover_type_12.fit(train_data_cover_type_12, train_labels_cover_type_12)

# Fit a KNN Model to differentiate cover types three and six
knn_model_cover_type_36 = KNeighborsClassifier(num_neighbors)
knn_model_cover_type_36.fit(train_data_cover_type_36, train_labels_cover_type_36)
```

#### Multi-Layer Perceptron


```python
from sklearn.neural_network import MLPClassifier

# Set model parameters
alpha = 1e-3
hidden_layer_sizes = (200,)
random_state = 1
max_iter = 200

# Fit a MLP  Model on the original dataset
mlp_model_original = MLPClassifier(alpha=alpha,hidden_layer_sizes=hidden_layer_sizes,random_state=random_state,max_iter=max_iter)
mlp_model_original.fit(train_data_original, train_labels_original)

# Fit a MLP Model to data with labels 12, 36, 4, 5, 7
mlp_model_all_data = MLPClassifier(alpha=alpha,hidden_layer_sizes=hidden_layer_sizes,random_state=random_state,max_iter=max_iter)
mlp_model_all_data.fit(train_data_12_36_4_5_7, train_labels_12_36_4_5_7)

# Fit a MLP Model to differentiate cover types one and two
mlp_model_cover_type_12 = MLPClassifier(alpha=alpha,hidden_layer_sizes=hidden_layer_sizes,random_state=random_state,max_iter=max_iter)
mlp_model_cover_type_12.fit(train_data_cover_type_12, train_labels_cover_type_12)

# Fit a MLP Model to differentiate cover types three and six
mlp_model_cover_type_36 = MLPClassifier(alpha=alpha,hidden_layer_sizes=hidden_layer_sizes,random_state=random_state,max_iter=max_iter)
mlp_model_cover_type_36.fit(train_data_cover_type_36, train_labels_cover_type_36)
```




    MLPClassifier(alpha=0.001, hidden_layer_sizes=(200,), random_state=1)



#### Adaboost 


```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

# Fit an Adaboost Model to all of the training data
adaboost_original = AdaBoostClassifier(DecisionTreeClassifier(max_depth=8),n_estimators=200,random_state=1,)
adaboost_original.fit(train_data_original,train_labels_original)

# Fit a Adaboost Model to data with labels 12, 36, 4, 5, 7
adaboost_all_data = AdaBoostClassifier(DecisionTreeClassifier(max_depth=8),n_estimators=200,random_state=1,)
adaboost_all_data.fit(train_data_12_36_4_5_7, train_labels_12_36_4_5_7)

# Fit a Adaboost Model to differentiate cover types one and two
adaboost_cover_type_12 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=8),n_estimators=200,random_state=1,)
adaboost_cover_type_12.fit(train_data_cover_type_12.to_numpy(), train_labels_cover_type_12)

# Fit a Adaboost Model to differentiate cover types three and six
adaboost_cover_type_36 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=8),n_estimators=200,random_state=1,)
adaboost_cover_type_36.fit(train_data_cover_type_36.to_numpy(), train_labels_cover_type_36)
```




    AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=8),
                       n_estimators=200, random_state=1)



#### Xgboost 


```python
from xgboost import XGBClassifier

# Fit a XG Model to all of the training data
xgb_model_original = XGBClassifier(n_estimators=200,max_depth=8,random_state=1)
xgb_model_original.fit(train_data_original,train_labels_original)

# Fit a XGBoost Model to data with labels 12, 36, 4, 5, 7
xgb_model_all_data = XGBClassifier(n_estimators=200,max_depth=8,random_state=1)
xgb_model_all_data.fit(train_data_12_36_4_5_7, train_labels_12_36_4_5_7)

# Fit a XGBoost Model to differentiate cover types one and two
xgb_model_cover_type_12 = XGBClassifier(n_estimators=200,max_depth=8,random_state=1)
xgb_model_cover_type_12.fit(train_data_cover_type_12.to_numpy(), train_labels_cover_type_12)

# Fit a XGBoost Model to differentiate cover types three and six
xgb_model_cover_type_36 = XGBClassifier(n_estimators=200,max_depth=8)
xgb_model_cover_type_36.fit(train_data_cover_type_36.to_numpy(), train_labels_cover_type_36)
```




    XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                  colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
                  importance_type='gain', interaction_constraints='',
                  learning_rate=0.300000012, max_delta_step=0, max_depth=8,
                  min_child_weight=1, missing=nan, monotone_constraints='()',
                  n_estimators=200, n_jobs=0, num_parallel_tree=1, random_state=0,
                  reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
                  tree_method='exact', validate_parameters=1, verbosity=None)



## Evaluation

Predictions from each model are producted accordingly (for both dev and test)
<img src="data/predictions.png">

#### Evaluate Random Forest Model 


```python
# Evaluate random forest model based on original data (no subsetting)
forest_original_score = forest_model_original.score(dev_data_original,dev_labels_original)
print("Random Forest Perforamance (against original dev labels) - Original data: ",forest_original_score)

# Evaluate random forest model on subset data 12, 36, 4, 5, 7
forest_predictions_all_data = forest_model_all_data.predict(dev_data_12_36_4_5_7)
print("Random Forest Perforamance (against 12/36 dev labels) - Model 1: ", accuracy_score(np.array(dev_labels_12_36_4_5_7), forest_predictions_all_data))

# Retrieve examples where the model predicted 12 or 36
forest_predicted_12_indicies = np.where(forest_predictions_all_data==12)
forest_predicted_36_indicies = np.where(forest_predictions_all_data==36)

# Use specific sub-models to differentiate between 1 and 2 -and- 3 and 6
forest_data_12 = fe.combine_environment_features_ct12(dev_data_12_36_4_5_7.iloc[forest_predicted_12_indicies])
forest_predictions_1_2 = forest_model_cover_type_12.predict(forest_data_12)

forest_data_36 = fe.combine_environment_features_ct36(dev_data_12_36_4_5_7.iloc[forest_predicted_36_indicies])
forest_predictions_3_6 = forest_model_cover_type_36.predict(forest_data_36)

# Update those 12 or 36 labels to be 1,2,3,6
forest_predictions_all_data[forest_predicted_12_indicies] = forest_predictions_1_2
forest_predictions_all_data[forest_predicted_36_indicies] = forest_predictions_3_6

print("Random Forest Perforamance Against Real Dev Labels - Overall Score: ",  accuracy_score(np.array(dev_labels_original).reshape(-1, 1), forest_predictions_all_data.reshape(-1, 1)))
```

    Random Forest Perforamance (against original dev labels) - Original data:  0.5119047619047619
    Random Forest Perforamance (against 12/36 dev labels) - Model 1:  0.5383597883597884
    Random Forest Perforamance Against Real Dev Labels - Overall Score:  0.38492063492063494


If set max_depth =8, without subsetting, the accuracy on the dev data is 74%-75%. With separte model training and consolidating the results, accuracy on dev data has improved to 79%-80%. However, this leads to overfitting. We keep "max_depth=1" to achieve better fitting on the "testing data".

#### Evaluate KNN Model


```python
# Evaluate knn model based on original data (no subsetting)
knn_original_score = knn_original.score(dev_data_original,dev_labels_original)
print("KNN Perforamance (against original dev labels) - Original data: ",knn_original_score)

knn_predictions_all_data = knn_model_all_data.predict(dev_data_12_36_4_5_7)
print("KNN Perforamance (against 12/36 dev labels) - Model 1: ", accuracy_score(np.array(dev_labels_12_36_4_5_7), knn_predictions_all_data))

# Retrieve examples where the model predicted 12 or 36
knn_predicted_12_indicies = np.where(knn_predictions_all_data==12)
knn_predicted_36_indicies = np.where(knn_predictions_all_data==36)

# Use specific sub-models to differentiate between 1 and 2 -and- 3 and 6
knn_data_12 = fe.combine_environment_features_ct12(dev_data_12_36_4_5_7.iloc[knn_predicted_12_indicies])
knn_predictions_1_2 = knn_model_cover_type_12.predict(knn_data_12)
knn_data_36 = fe.combine_environment_features_ct36(dev_data_12_36_4_5_7.iloc[knn_predicted_36_indicies])
knn_predictions_3_6 = knn_model_cover_type_36.predict(knn_data_36)

# Update those 12 or 36 labels to be 1,2,3,6
knn_predictions_all_data[knn_predicted_12_indicies] = knn_predictions_1_2
knn_predictions_all_data[knn_predicted_36_indicies] = knn_predictions_3_6

print("KNN Perforamance Against Real Dev Labels - Overall Score: ",  accuracy_score(np.array(dev_labels_original).reshape(-1, 1), knn_predictions_all_data.reshape(-1, 1)))
```

    KNN Perforamance (against original dev labels) - Original data:  0.8366402116402116
    KNN Perforamance (against 12/36 dev labels) - Model 1:  0.9186507936507936
    KNN Perforamance Against Real Dev Labels - Overall Score:  0.9034391534391535


#### Evaluate MLP Model


```python
# Evaluate mlp model based on original data (no subsetting)
print("MLP on original data accuracy:", mlp_model_original.score(dev_data_original,dev_labels_original))

# Plot Confusion Matrix
plot_confusion_matrix(mlp_model_original, dev_data_original, dev_labels_original, values_format = "d")
plt.title("MLP with Original Data Confusion Matrix")
plt.plot()

mlp_predictions_all_data = mlp_model_all_data.predict(dev_data_12_36_4_5_7)
print("MLP Perforamance (against 12/36 dev labels)- Model 1: ", accuracy_score(np.array(dev_labels_12_36_4_5_7), mlp_predictions_all_data))

# Retrieve examples where the model predicted 12 or 36
mlp_predicted_12_indicies = np.where(mlp_predictions_all_data==12)
mlp_predicted_36_indicies = np.where(mlp_predictions_all_data==36)

# Use specific sub-models to differentiate between 1 and 2 -and- 3 and 6
mlp_data_12 = fe.combine_environment_features_ct12(dev_data_12_36_4_5_7.iloc[mlp_predicted_12_indicies])
mlp_predictions_1_2 = mlp_model_cover_type_12.predict(mlp_data_12)
mlp_data_36 = fe.combine_environment_features_ct36(dev_data_12_36_4_5_7.iloc[mlp_predicted_36_indicies])
mlp_predictions_3_6 = mlp_model_cover_type_36.predict(mlp_data_36)

# Update those 12 or 36 labels to be 1,2,3,6
mlp_predictions_all_data[mlp_predicted_12_indicies] = mlp_predictions_1_2
mlp_predictions_all_data[mlp_predicted_36_indicies] = mlp_predictions_3_6

print("MLP Perforamance Against Real Dev Labels - Overall Score: ",  accuracy_score(np.array(dev_labels_original).reshape(-1, 1), mlp_predictions_all_data.reshape(-1, 1)))
```

    MLP on original data accuracy: 0.8386243386243386
    MLP Perforamance (against 12/36 dev labels)- Model 1:  0.9259259259259259
    MLP Perforamance Against Real Dev Labels - Overall Score:  0.8571428571428571



![png](backups/clear_cut_solution_files/backups/clear_cut_solution_39_1.png)


Without training sub-models for type 1 and 2, type 3 and 6, the accuracy of the MLP model on dev data is 83.7%. With separate sub-model training, accuracy on the dev data improved to 91.9%. 

#### Evaluate Adaboost Model


```python
# First, evaluate Adaboost model on the original data (no subsetting)
adaboost_score = adaboost_original.score(dev_data_original,dev_labels_original)
print("Adaboost with original data accuracy:", adaboost_score)

# Plot Confusion Matrix
plot_confusion_matrix(adaboost_original, dev_data_original, dev_labels_original, values_format = "d")
plt.title("Adaboost Confusion Matrix")
plt.plot()

# Evaluate XGBoost model with the submodels.
adaboost_predictions_all_data = adaboost_all_data.predict(dev_data_12_36_4_5_7)
print("Adaboost Perforamance (against 12/36 dev labels) - Model 1: ", accuracy_score(np.array(dev_labels_12_36_4_5_7), adaboost_predictions_all_data))

# Retrieve examples where the model predicted 12 or 36
adaboost_predicted_12_indicies = np.where(adaboost_predictions_all_data==12)
adaboost_predicted_36_indicies = np.where(adaboost_predictions_all_data==36)

# Use specific sub-models to differentiate between 1 and 2 -and- 3 and 6
adaboost_data_12 = fe.combine_environment_features_ct12(dev_data_12_36_4_5_7.iloc[adaboost_predicted_12_indicies])
adaboost_predictions_1_2 = adaboost_cover_type_12.predict(adaboost_data_12)
adaboost_data_36 = fe.combine_environment_features_ct36(dev_data_12_36_4_5_7.iloc[adaboost_predicted_36_indicies])
adaboost_predictions_3_6 = adaboost_cover_type_36.predict(adaboost_data_36)

# Update those 12 or 36 labels to be 1,2,3,6
adaboost_predictions_all_data[adaboost_predicted_12_indicies] = adaboost_predictions_1_2
adaboost_predictions_all_data[adaboost_predicted_36_indicies] = adaboost_predictions_3_6

print("Adaboost Perforamance Against Real Dev Labels - Overall Score: ",  accuracy_score(np.array(dev_labels_original).reshape(-1, 1), adaboost_predictions_all_data.reshape(-1, 1)))
```

    Adaboost with original data accuracy: 0.8247354497354498
    Adaboost Perforamance (against 12/36 dev labels) - Model 1:  0.9365079365079365
    Adaboost Perforamance Against Real Dev Labels - Overall Score:  0.919973544973545



![png](backups/clear_cut_solution_files/backups/clear_cut_solution_42_1.png)


We only trained and evaluated an adaboost model on the original data, to determine how it compares to the XGboost model. As the accuracy is 5% lower than that of the XGBoost model, we'll use XGBoost for our final model.

#### Evaluate Xgboost Model


```python
# First, evaluate XGBoost model on the original data (no subsetting)
print("XGBoost with original data accuracy:", xgb_model_original.score(dev_data_original,dev_labels_original))

# Plot Confusion Matrix
plot_confusion_matrix(xgb_model_original, dev_data_original, dev_labels_original, values_format = "d")
plt.title("XGBoost Confusion Matrix")
plt.plot()

# Evaluate XGBoost model with the submodels.
xgb_predictions_all_data = xgb_model_all_data.predict(dev_data_12_36_4_5_7)
print("Xgboost Perforamance (against 12/36 dev labels) - Model 1: ", accuracy_score(np.array(dev_labels_12_36_4_5_7), xgb_predictions_all_data))

# Retrieve examples where the model predicted 12 or 36
xgboost_predicted_12_indicies = np.where(xgb_predictions_all_data==12)
xgboost_predicted_36_indicies = np.where(xgb_predictions_all_data==36)

# Use specific sub-models to differentiate between 1 and 2 -and- 3 and 6
xgboost_data_12 = fe.combine_environment_features_ct12(dev_data_12_36_4_5_7.iloc[xgboost_predicted_12_indicies])
xgboost_predictions_1_2 = xgb_model_cover_type_12.predict(xgboost_data_12.to_numpy())
xgboost_data_36 = fe.combine_environment_features_ct36(dev_data_12_36_4_5_7.iloc[xgboost_predicted_36_indicies])
xgboost_predictions_3_6 = xgb_model_cover_type_36.predict(xgboost_data_36.to_numpy())

# Update those 12 or 36 labels to be 1,2,3,6
xgb_predictions_all_data[xgboost_predicted_12_indicies] = xgboost_predictions_1_2
xgb_predictions_all_data[xgboost_predicted_36_indicies] = xgboost_predictions_3_6

print("Xgboost Perforamance Against Real Dev Labels - Overall Score: ",  accuracy_score(np.array(dev_labels_original).reshape(-1, 1), xgb_predictions_all_data.reshape(-1, 1)))
```

    XGBoost with original data accuracy: 0.8690476190476191
    Xgboost Perforamance (against 12/36 dev labels) - Model 1:  0.9404761904761905
    Xgboost Perforamance Against Real Dev Labels - Overall Score:  0.9285714285714286



![png](backups/clear_cut_solution_files/backups/clear_cut_solution_45_1.png)


Without subsetting, xgboost algorithm works very well for the dev data, showing an accuracy above 86%, which is 3% better than the 2nd best performer, or the MLP model. With subsetting and training submodels for type 1 and 2, and 3 and 6, the accuracy on the dev data has imrpoved to over 94%. However, when applied to the "testing data" in Kaggle, the accuracy dropped back to 75%. Still 1% higher than that from MLP model, but the difference is small. Also, there is small decrease in model accuracy in "testing data" when we train one model compared to training separate models and consolidate.


#### Ensemble
Here we will combine the three best performing models and implement a "voting" system to try to improve accuracy.
<img src="data/ensemble.png">


```python
# An ensemble of knn, random forest, mlp, xgboost and adaboost results
new_predictions = models.ensemble(
    knn_predictions_all_data ,
    forest_predictions_all_data, 
    mlp_predictions_all_data,
    xgb_predictions_all_data,
    adaboost_predictions_all_data)
accuracy = accuracy_score(dev_labels_original, new_predictions)
print("Ensemble Accuracy: ", accuracy)
```

    Models disagreed on 1026/1512 examples.
    Ensemble Accuracy:  0.9252645502645502


### Test Results


```python
# Read in testing data 
test_data = pd.read_csv("data/test.csv")

# Preserve testing df ID for submission purpose
test_df_id = test_data["Id"]

# Apply Same Transformations 
test_data = fe.manipulate_data(test_data)
test_data = fe.scale_non_training_data(standardize_features, test_data, train_data_original_scaler)

# Verify shape
print("Testing data shape: ", test_data.shape)
```

    Testing data shape:  (565892, 54)


#### Test Random Forest


```python
# Inference on the entire original dataset 
test_forest_predictions_original = forest_model_all_data.predict(test_data)
forest_result = pd.DataFrame.from_dict(dict(zip(test_df_id.to_list(),test_forest_predictions_original)), orient='index', columns=["Cover_Type"])
forest_result.to_csv(f"submissions/forest_submission.csv",index_label="Id")

# Retrieve examples where the model predicted 12 or 36
test_forest_predicted_12_indicies = np.where(test_forest_predictions_original==12)
test_forest_predicted_36_indicies = np.where(test_forest_predictions_original==36)

# Use specific sub-models to differentiate between 1 and 2 -and- 3 and 6
test_forest_data_12 = fe.combine_environment_features_ct12(test_data.iloc[test_forest_predicted_12_indicies])
test_forest_predictions_1_2 = forest_model_cover_type_12.predict(test_forest_data_12)

# Use specific sub-models to differentiate between 1 and 2 -and- 3 and 6
test_forest_data_36 = fe.combine_environment_features_ct36(test_data.iloc[test_forest_predicted_36_indicies])
test_forest_predictions_3_6 = forest_model_cover_type_36.predict(test_forest_data_36)

# Update those 12 or 36 labels to be 1,2,3,6
test_forest_predictions_original[test_forest_predicted_12_indicies] = test_forest_predictions_1_2
test_forest_predictions_original[test_forest_predicted_36_indicies] = test_forest_predictions_3_6
```

#### Test KNN


```python
# Inference on the entire original dataset 
test_knn_predictions_original = knn_model_all_data.predict(test_data)
knn_result = pd.DataFrame.from_dict(dict(zip(test_df_id.to_list(),test_knn_predictions_original)), orient='index', columns=["Cover_Type"])
knn_result.to_csv(f"submissions/knn_submission.csv",index_label="Id")

# Retrieve examples where the model predicted 12 or 36
test_knn_predicted_12_indicies = np.where(test_knn_predictions_original==12)
test_knn_predicted_36_indicies = np.where(test_knn_predictions_original==36)

# Use specific sub-models to differentiate between 1 and 2 -and- 3 and 6
test_knn_data_12 = fe.combine_environment_features_ct12(test_data.iloc[test_knn_predicted_12_indicies])
test_knn_predictions_1_2 = knn_model_cover_type_12.predict(test_knn_data_12)

test_knn_data_36 = fe.combine_environment_features_ct36(test_data.iloc[test_knn_predicted_36_indicies])
test_knn_predictions_3_6 = knn_model_cover_type_36.predict(test_knn_data_36)

# Update those 12 or 36 labels to be 1,2,3,6
test_knn_predictions_original[test_knn_predicted_12_indicies] = test_knn_predictions_1_2
test_knn_predictions_original[test_knn_predicted_36_indicies] = test_knn_predictions_3_6
```

#### Test MLP


```python
# Inference on the entire original dataset 
test_mlp_predictions_original = mlp_model_all_data.predict(test_data)
mlp_result = pd.DataFrame.from_dict(dict(zip(test_df_id.to_list(),test_mlp_predictions_original)), orient='index', columns=["Cover_Type"])
mlp_result.to_csv(f"submissions/mlp_submission.csv",index_label="Id")

# Retrieve examples where the model predicted 12 or 36
test_mlp_predicted_12_indicies = np.where(test_mlp_predictions_original==12)
test_mlp_predicted_36_indicies = np.where(test_mlp_predictions_original==36)

# Use specific sub-models to differentiate between 1 and 2 -and- 3 and 6
test_mlp_data_12 = fe.combine_environment_features_ct12(test_data.iloc[test_mlp_predicted_12_indicies])
test_mlp_predictions_1_2 = mlp_model_cover_type_12.predict(test_mlp_data_12)

test_mlp_data_36 = fe.combine_environment_features_ct36(test_data.iloc[test_mlp_predicted_36_indicies])
test_mlp_predictions_3_6 = mlp_model_cover_type_36.predict(test_mlp_data_36)

# Update those 12 or 36 labels to be 1,2,3,6
test_mlp_predictions_original[test_mlp_predicted_12_indicies] = test_mlp_predictions_1_2
test_mlp_predictions_original[test_mlp_predicted_36_indicies] = test_mlp_predictions_3_6
```

#### Test Adaboost


```python
# Inference on the entire original dataset 
test_adaboost_predictions_original = adaboost_all_data.predict(test_data)
adaboost_result = pd.DataFrame.from_dict(dict(zip(test_df_id.to_list(),test_adaboost_predictions_original)), orient='index', columns=["Cover_Type"])
adaboost_result.to_csv(f"submissions/mlp_submission.csv",index_label="Id")

# Retrieve examples where the model predicted 12 or 36
test_adaboost_predicted_12_indicies = np.where(test_adaboost_predictions_original==12)
test_adaboost_predicted_36_indicies = np.where(test_adaboost_predictions_original==36)

# Use specific sub-models to differentiate between 1 and 2 -and- 3 and 6
test_adaboost_data_12 = fe.combine_environment_features_ct12(test_data.iloc[test_adaboost_predicted_12_indicies])
print(test_adaboost_data_12.shape)
test_adaboost_predictions_1_2 = adaboost_cover_type_12.predict(test_adaboost_data_12.to_numpy())

test_adaboost_data_36 = fe.combine_environment_features_ct36(test_data.iloc[test_adaboost_predicted_36_indicies])
test_adaboost_predictions_3_6 = adaboost_cover_type_36.predict(test_adaboost_data_36)

# Update those 12 or 36 labels to be 1,2,3,6
test_adaboost_predictions_original[test_adaboost_predicted_12_indicies] = test_adaboost_predictions_1_2
test_adaboost_predictions_original[test_adaboost_predicted_36_indicies] = test_adaboost_predictions_3_6
```

    (450013, 56)


#### Test XGBoost Ensemble


```python
# Test xgboost model with original data (no subsetting) and generate submission file
test_xgb_predictions_original = xgb_model_original.predict(test_data)
xg_results = pd.DataFrame.from_dict(dict(zip(test_df_id.to_list(),test_xgb_predictions_original)), orient='index', columns=["Cover_Type"])
xg_results.to_csv(f"submissions/xgb_submission.csv",index_label="Id")

test_xgb_predictions = xgb_model_all_data.predict(test_data)

# Retrieve examples where the model predicted 12 or 36
test_xgb_predicted_12_indicies = np.where(test_xgb_predictions_original==12)
test_xgb_predicted_36_indicies = np.where(test_xgb_predictions_original==36)

# Use specific sub-models to differentiate between 1 and 2 -and- 3 and 6
test_xgb_data_12 = fe.combine_environment_features_ct12(test_data.iloc[test_xgb_predicted_12_indicies])
test_xgb_predictions_1_2 = xgb_model_cover_type_12.predict(test_xgb_data_12.to_numpy())
test_xgb_data_36 = fe.combine_environment_features_ct36(test_data.iloc[test_xgb_predicted_36_indicies])
test_xgb_predictions_3_6 = xgb_model_cover_type_36.predict(test_xgb_data_36.to_numpy())

# Update those 12 or 36 labels to be 1,2,3,6
test_xgb_predictions_original[test_xgb_predicted_12_indicies] = test_xgb_predictions_1_2
test_xgb_predictions_original[test_xgb_predicted_36_indicies] = test_xgb_predictions_3_6

pd.DataFrame.from_dict(dict(zip(test_df_id.to_list(),test_xgb_predictions_original)), orient='index', columns=["Cover_Type"]).to_csv(f"submissions/xgb_submission_submodels.csv",index_label="Id")
```

#### Test Ensemble


```python
new_predictions = models.ensemble(test_knn_predictions_original,
                                  test_forest_predictions_original, 
                                  test_mlp_predictions_original,
                                  test_xg_predictions_original,
                                  test_ada_predictions_original)
```

#### Generate Submission File


```python
result = pd.DataFrame.from_dict(dict(zip(test_df_id.to_list(),new_predictions)), orient='index', columns=["Cover_Type"])
result.to_csv(f"submissions/cosolidated_submission.csv",index_label="Id")
```

### End matter

#### Acknowledgements/Sources

* That helpful stack overflow post
  * https://stackoverflow.com/questions/28663856/how-to-count-the-occurrence-of-certain-item-in-an-ndarray
* Relevant Documentation
  * KNeighborsClassifier
    * https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
  * Pretty Confusion Matrix
    * https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_confusion_matrix.html
  * Preprocessing
    * https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html
* Soil information
  * https://www.uidaho.edu/cals/soil-orders/aridisols
* Adaboost Tuning
    * https://medium.com/@chaudhurysrijani/tuning-of-adaboost-with-computational-complexity-8727d01a9d20
  
#### Backup Formats

*because sometimes you just want to look at the markdown or whatever real quick*


```python
#Create a backup of the jupyter notebook in a format for where changes are easier to see.
!jupyter nbconvert clear_cut_solution.ipynb --to="python" --output="backups/clear_cut_solution"
!jupyter nbconvert clear_cut_solution.ipynb --to markdown --output="backups/clear_cut_solution"

# Also archiving this bad boy
!jupyter nbconvert clear_cut_solution.ipynb --to html --output="backups/clear_cut_solution"
```


```python

```
