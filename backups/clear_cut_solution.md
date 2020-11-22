# Forest Cover Type Prediction
#### Team: Clear-Cut Solution: Kevin Martin, Yang Jing, Justine Schabel

**Todo** write a (better) intro    ;-]

This noteboook documents the development of a predictive model to determine the type of forest cover within national forests in the state of Colorado given various geographic location characteristics. 

We quickly determined that the most difficult cover types to distinguish betweeen are type 1 and 2. We focused particular attention on developing a model that could pick the 2 of them apart.

## Initial Setup
#### Import Required Libraries


```python
#surpress warning messages
import warnings
warnings.filterwarnings("ignore")

# This tells matplotlib not to try opening a new window for each plot.
%matplotlib inline

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
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import plot_confusion_matrix

# So we can reload packages without restarting the kernel
import importlib

import numpy as np
```


```python
# If you update the feature_engineering package, run this line so it updates without needing to restart the kernel
importlib.reload(fe)
```




    <module 'feature_engineering' from '/home/jupyter/mids-w207-final-project/feature_engineering.py'>



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
      <th>Soil_Type35</th>
      <th>Soil_Type36</th>
      <th>Soil_Type37</th>
      <th>Soil_Type38</th>
      <th>Soil_Type39</th>
      <th>Soil_Type40</th>
      <th>Cover_Type</th>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
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
sns.violinplot(x=train_df['Cover_Type'],y=train_df['ap_ew'],ax=ax1)
sns.histplot(train_df['ap_ew'],ax=ax2)
plt.show()
```


![png](backups/clear_cut_solution_files/backups/clear_cut_solution_10_0.png)


After the feature transformation, we see improved distinction in median values, espeically for cover type 6, where the median is notably higher than that of other cover types and the distribution is concentrated around the median.

#### Segment data

Based on closer examination of our model performance, we found that the models consistently confused cover types 1 and 2 and covertypes 3 and 6. So we decided to break up our model into one primary model (outputs 12 (for 1 or 2), 36 (for outputs 3 or 6), 4, 5 or 7) and two secondary models (one for distinguishing 1 or 2 and the other for distinguishing 3 or 6). 


```python
# Split trainning data into subsets to improve model accurracy with cover type 1&2, and 3&6.
train_df_original, train_df_12_36_4_5_7, train_df_12, train_df_36 = fe.subset_data(train_df)
```

#### Split data into train/dev

Then, we split the training data into a training data set (80%) and development data set (20%). We will also have a large, separate test data set. 


```python
# Only (randomly) sample indicies once for the entire dataset 
# This will be very important for comparing the output to the original dev labels 
train_indicies_all_data = list(train_df_original.sample(frac=0.8).index)

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

print("Data with Labels 1,3")
print("Training data shape: {0} Training labels shape: {1}".format(train_data_cover_type_36.shape, train_labels_cover_type_36.shape))
print("Dev data shape: {0} Dev labels shape: {1}\n".format(dev_data_cover_type_36.shape, dev_labels_cover_type_36.shape))
```

    Original Data with Labels 1,2,3,4,5,6,7
    Training data shape: (12096, 54) Training labels shape: (12096,)
    Dev data shape: (3024, 54) Dev labels shape: (3024,)
    
    Data with Labels 12,36,4,5,7
    Training data shape: (12096, 54) Training labels shape: (12096,)
    Dev data shape: (3024, 54) Dev labels shape: (3024,)
    
    Data with Labels 1,2
    Training data shape: (3456, 54) Training labels shape: (3456,)
    Dev data shape: (864, 54) Dev labels shape: (864,)
    
    Data with Labels 1,3
    Training data shape: (3456, 54) Training labels shape: (3456,)
    Dev data shape: (864, 54) Dev labels shape: (864,)
    


## Models

Fit Random Forest, K Nearest Neighbors and Multilayer Perceptron models to the training data
<img src="data/training_models.png">

#### Random Forest


```python
from sklearn.ensemble import RandomForestClassifier

# Set model parameters
num_trees = 100
max_depth = 8

# Fit a Random Forest Model to all of the training data
forest_model_all_data = RandomForestClassifier(num_trees, max_depth=max_depth)
forest_model_all_data.fit(train_data_12_36_4_5_7, train_labels_12_36_4_5_7)

# # Fit a Random Forest Model to differentiate cover types one and two
forest_model_cover_type_12 = RandomForestClassifier(num_trees, max_depth=max_depth)
forest_model_cover_type_12.fit(train_data_cover_type_12, train_labels_cover_type_12)

# # Fit a Random Forest Model to differentiate cover types three and six
forest_model_cover_type_36 = RandomForestClassifier(num_trees, max_depth=max_depth)
forest_model_cover_type_36.fit(train_data_cover_type_36, train_labels_cover_type_36)
```




    RandomForestClassifier(max_depth=8)



#### K-Nearest Neighbors


```python
from sklearn.neighbors import KNeighborsClassifier

# Set model parameters
num_neighbors = 1

# Fit a KNN Model to all of the training data
knn_model_all_data = KNeighborsClassifier(num_neighbors)
knn_model_all_data.fit(train_data_12_36_4_5_7, train_labels_12_36_4_5_7)

# Fit a KNN Model to differentiate cover types one and two
knn_model_cover_type_12 = KNeighborsClassifier(num_neighbors)
knn_model_cover_type_12.fit(train_data_cover_type_12, train_labels_cover_type_12)

# Fit a KNN Model to differentiate cover types three and six
knn_model_cover_type_36 = KNeighborsClassifier(num_neighbors)
knn_model_cover_type_36.fit(train_data_cover_type_36, train_labels_cover_type_36)
```




    KNeighborsClassifier(n_neighbors=1)



#### Multi-Layer Perceptron


```python
from sklearn.neural_network import MLPClassifier

# Set model parameters
alpha = 1e-3
hidden_layer_sizes = (200,)
random_state = 0
max_iter = 200

# Fit a KNN Model to all of the training data
mlp_model_all_data = MLPClassifier(alpha=alpha,hidden_layer_sizes=hidden_layer_sizes,random_state=random_state,max_iter=max_iter)
mlp_model_all_data.fit(train_data_12_36_4_5_7, train_labels_12_36_4_5_7)

# # Fit a KNN Model to differentiate cover types one and two
mlp_model_cover_type_12 = MLPClassifier(alpha=alpha,hidden_layer_sizes=hidden_layer_sizes,random_state=random_state,max_iter=max_iter)
mlp_model_cover_type_12.fit(train_data_cover_type_12, train_labels_cover_type_12)

# # Fit a KNN Model to differentiate cover types three and six
mlp_model_cover_type_36 = MLPClassifier(alpha=alpha,hidden_layer_sizes=hidden_layer_sizes,random_state=random_state,max_iter=max_iter)
mlp_model_cover_type_36.fit(train_data_cover_type_36, train_labels_cover_type_36)
```




    MLPClassifier(alpha=0.001, hidden_layer_sizes=(200,), random_state=0)



## Evaluation

Predictions from each model are producted accordingly (for both dev and test)
<img src="data/predictions.png">

#### Evaluate Random Forest Model 


```python
forest_predictions_all_data = forest_model_all_data.predict(dev_data_12_36_4_5_7)
print("Random Forest Perforamance (against 12/36 dev labels) - Model 1: ", accuracy_score(np.array(dev_labels_12_36_4_5_7), forest_predictions_all_data))

# Retrieve examples where the model predicted 12 or 36
forest_predicted_12_indicies = np.where(forest_predictions_all_data==12)
forest_predicted_36_indicies = np.where(forest_predictions_all_data==36)

# Use specific sub-models to differentiate between 1 and 2 -and- 3 and 6
forest_predictions_1_2 = forest_model_cover_type_12.predict(np.array(dev_data_12_36_4_5_7)[forest_predicted_12_indicies])
forest_predictions_3_6 = forest_model_cover_type_36.predict(np.array(dev_data_12_36_4_5_7)[forest_predicted_36_indicies])

# Update those 12 or 36 labels to be 1,2,3,6
forest_predictions_all_data[forest_predicted_12_indicies] = forest_predictions_1_2
forest_predictions_all_data[forest_predicted_36_indicies] = forest_predictions_3_6

print("Random Forest Perforamance Against Real Dev Labels - Overall Score: ",  accuracy_score(np.array(dev_labels_original).reshape(-1, 1), forest_predictions_all_data.reshape(-1, 1)))
```

    Random Forest Perforamance (against 12/36 dev labels) - Model 1:  0.8792989417989417
    Random Forest Perforamance Against Real Dev Labels - Overall Score:  0.7913359788359788


#### Evaluate KNN Model


```python
knn_predictions_all_data = knn_model_all_data.predict(dev_data_12_36_4_5_7)
print("KNN Perforamance (against 12/36 dev labels) - Model 1: ", accuracy_score(np.array(dev_labels_12_36_4_5_7), knn_predictions_all_data))

# Retrieve examples where the model predicted 12 or 36
knn_predicted_12_indicies = np.where(knn_predictions_all_data==12)
knn_predicted_36_indicies = np.where(knn_predictions_all_data==36)

# Use specific sub-models to differentiate between 1 and 2 -and- 3 and 6
knn_predictions_1_2 = knn_model_cover_type_12.predict(np.array(dev_data_12_36_4_5_7)[knn_predicted_12_indicies])
knn_predictions_3_6 = knn_model_cover_type_36.predict(np.array(dev_data_12_36_4_5_7)[knn_predicted_36_indicies])

# Update those 12 or 36 labels to be 1,2,3,6
knn_predictions_all_data[knn_predicted_12_indicies] = knn_predictions_1_2
knn_predictions_all_data[knn_predicted_36_indicies] = knn_predictions_3_6

print("KNN Perforamance Against Real Dev Labels - Overall Score: ",  accuracy_score(np.array(dev_labels_original).reshape(-1, 1), knn_predictions_all_data.reshape(-1, 1)))
```

    KNN Perforamance (against 12/36 dev labels) - Model 1:  0.9166666666666666
    KNN Perforamance Against Real Dev Labels - Overall Score:  0.9001322751322751


#### Evaluate MLP Model


```python
mlp_predictions_all_data = mlp_model_all_data.predict(dev_data_12_36_4_5_7)
print("MLP Perforamance (against 12/36 dev labels)- Model 1: ", accuracy_score(np.array(dev_labels_12_36_4_5_7), mlp_predictions_all_data))

# Retrieve examples where the model predicted 12 or 36
mlp_predicted_12_indicies = np.where(mlp_predictions_all_data==12)
mlp_predicted_36_indicies = np.where(mlp_predictions_all_data==36)

# Use specific sub-models to differentiate between 1 and 2 -and- 3 and 6
mlp_predictions_1_2 = knn_model_cover_type_12.predict(np.array(dev_data_12_36_4_5_7)[mlp_predicted_12_indicies])
mlp_predictions_3_6 = knn_model_cover_type_36.predict(np.array(dev_data_12_36_4_5_7)[mlp_predicted_36_indicies])

# Update those 12 or 36 labels to be 1,2,3,6
mlp_predictions_all_data[mlp_predicted_12_indicies] = mlp_predictions_1_2
mlp_predictions_all_data[mlp_predicted_36_indicies] = mlp_predictions_3_6

print("MLP Perforamance Against Real Dev Labels - Overall Score: ",  accuracy_score(np.array(dev_labels_original).reshape(-1, 1), mlp_predictions_all_data.reshape(-1, 1)))
```

    MLP Perforamance (against 12/36 dev labels)- Model 1:  0.939484126984127
    MLP Perforamance Against Real Dev Labels - Overall Score:  0.9232804232804233


#### Ensemble
Here we will combine the three best performing models and implement a "voting" system to try to improve accuracy.
<img src="data/ensemble.png">


```python
new_predictions = models.ensemble(forest_predictions_all_data, knn_predictions_all_data, mlp_predictions_all_data)
accuracy = accuracy_score(dev_labels_original, new_predictions)
print("Ensemble Accuracy: ", accuracy)
```

    Models disagreed on 718/3024 examples.
    Ensemble Accuracy:  0.9222883597883598


### Test Results


```python
# Read in testing data 
test_data = pd.read_csv("data/test.csv")

# Preserve testing df ID for submission purpose
test_df_id = test_data["Id"]
```


```python
# Apply Same Transformations 
test_data = fe.manipulate_data(test_data)
test_data = fe.scale_non_training_data(standardize_features, test_data, train_data_original_scaler)
```


```python
# Verify shape
print("Testing data shape: ", test_data.shape)
```

    Testing data shape:  (565892, 54)


#### Test Random Forest


```python
test_forest_predictions = forest_model_all_data.predict(test_data)

# Retrieve examples where the model predicted 12 or 36
test_forest_predicted_12_indicies = np.where(test_forest_predictions==12)
test_forest_predicted_36_indicies = np.where(test_forest_predictions==36)

# Use specific sub-models to differentiate between 1 and 2 -and- 3 and 6
test_forest_predictions_1_2 = forest_model_cover_type_12.predict(np.array(test_data)[test_forest_predicted_12_indicies])
test_forest_predictions_3_6 = forest_model_cover_type_36.predict(np.array(test_data)[test_forest_predicted_36_indicies])

# Update those 12 or 36 labels to be 1,2,3,6
test_forest_predictions[test_forest_predicted_12_indicies] = test_forest_predictions_1_2
test_forest_predictions[test_forest_predicted_36_indicies] = test_forest_predictions_3_6
```

#### Test KNN


```python
test_knn_predictions = knn_model_all_data.predict(test_data)

# Retrieve examples where the model predicted 12 or 36
test_knn_predicted_12_indicies = np.where(test_knn_predictions==12)
test_knn_predicted_36_indicies = np.where(test_knn_predictions==36)

# Use specific sub-models to differentiate between 1 and 2 -and- 3 and 6
test_knn_predictions_1_2 = knn_model_cover_type_12.predict(np.array(test_data)[test_knn_predicted_12_indicies])
test_knn_predictions_3_6 = knn_model_cover_type_36.predict(np.array(test_data)[test_knn_predicted_36_indicies])

# Update those 12 or 36 labels to be 1,2,3,6
test_knn_predictions[test_knn_predicted_12_indicies] = test_knn_predictions_1_2
test_knn_predictions[test_knn_predicted_36_indicies] = test_knn_predictions_3_6
```

#### Test MLP


```python
test_mlp_predictions = mlp_model_all_data.predict(test_data)

# Retrieve examples where the model predicted 12 or 36
test_mlp_predicted_12_indicies = np.where(test_mlp_predictions==12)
test_mlp_predicted_36_indicies = np.where(test_mlp_predictions==36)

# Use specific sub-models to differentiate between 1 and 2 -and- 3 and 6
test_mlp_predictions_1_2 = mlp_model_cover_type_12.predict(np.array(test_data)[test_mlp_predicted_12_indicies])
test_mlp_predictions_3_6 = mlp_model_cover_type_36.predict(np.array(test_data)[test_mlp_predicted_36_indicies])

# Update those 12 or 36 labels to be 1,2,3,6
test_mlp_predictions[test_mlp_predicted_12_indicies] = test_mlp_predictions_1_2
test_mlp_predictions[test_mlp_predicted_36_indicies] = test_mlp_predictions_3_6
```

#### Test Ensemble


```python
new_predictions = models.ensemble(test_forest_predictions, test_knn_predictions, test_mlp_predictions)
```

#### Generate Submission File


```python
result = pd.DataFrame.from_dict(dict(zip(test_df_id.to_list(),new_predictions)), orient='index', columns=["Cover_Type"])
result.to_csv(f"submissions/cobsolidated_submission.csv",index_label="Id")
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
