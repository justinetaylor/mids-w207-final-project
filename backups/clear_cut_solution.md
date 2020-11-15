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


```


```python
# If you update the feature_engineering package, run this line so it updates without needing to restart the kernel
importlib.reload(fe)
```




    <module 'feature_engineering' from '/home/jovyan/work/feature_engineering.py'>



#### Load Data


```python
# Read in training data 
train_df = pd.read_csv("data/train.csv")

# Read in training data 
test_data = pd.read_csv("data/test.csv")
# Preserve testing df ID for submission purpose
test_df_ID = test_data["Id"]

#make a copy for testing subsets
test_data1 = test_data.copy() 
```

## Feature Engineering 
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
#subset trainning data into subsets to improve model accurracy with cover type 1&2, and 3&6.
train_df, train_df_12, train_df_36 = fe.subset_data(train_df)
```


```python
def manipulate_data(data):
    """ 
    This function applys transformations on the input data set
    
    Parameters: 
        data (dataframe): n_examples x m_features (int64) dataframe 
    """
    data = fe.scale_hillside(data)
    
    
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
train_df_12 = manipulate_data(train_df_12)
train_df_36 = manipulate_data(train_df_36)
```


```python
def manipulate_ct12(data):
    """ 
    This function applys additional transformations on the input data set for cover type 1 and 2
    
    Parameters: 
        data (dataframe): n_examples x m_features (int64) dataframe 
    """
        
    # Soil Combination One (based on distributions)
    data = fe.combine_environment_features_ct12(data)
    data = fe.drop_unseen_soil_types(data)  
    data = fe.combine_soil_types(data)
      
    
    # Soil Combination Two (based on descriptions)
    # data = fe.set_soil_type_by_attributes(data)
    
    return data

train_df_12 = manipulate_ct12(train_df_12)

```


```python
def manipulate_ct36(data):
    """ 
    This function applys additional transformations on the input data set for cover type 1 and 2
    
    Parameters: 
        data (dataframe): n_examples x m_features (int64) dataframe 
    """
        
    # Soil Combination One (based on distributions)
    data = fe.combine_environment_features_ct36(data)
    data = fe.drop_unseen_soil_types(data)  
    data = fe.combine_soil_types(data)
      
    
    # Soil Combination Two (based on descriptions)
    # data = fe.set_soil_type_by_attributes(data)
    
    return data

train_df_36 = manipulate_ct36(train_df_36)
```

#### Examine transformed data


```python
train_df.describe()
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
      <th>count</th>
      <td>15120.000000</td>
      <td>15120.000000</td>
      <td>15120.000000</td>
      <td>15120.000000</td>
      <td>15120.000000</td>
      <td>15120.000000</td>
      <td>15120.000000</td>
      <td>15120.000000</td>
      <td>15120.000000</td>
      <td>15120.000000</td>
      <td>...</td>
      <td>15120.000000</td>
      <td>15120.000000</td>
      <td>15120.000000</td>
      <td>15120.000000</td>
      <td>15120.000000</td>
      <td>15120.000000</td>
      <td>15120.000000</td>
      <td>15120.000000</td>
      <td>15120.000000</td>
      <td>1.512000e+04</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2749.322553</td>
      <td>16.501587</td>
      <td>4.645584</td>
      <td>7.124224</td>
      <td>0.858689</td>
      <td>0.529773</td>
      <td>7.058263</td>
      <td>0.237897</td>
      <td>0.033003</td>
      <td>0.419907</td>
      <td>...</td>
      <td>0.006746</td>
      <td>0.000661</td>
      <td>0.002249</td>
      <td>0.048148</td>
      <td>0.043452</td>
      <td>0.030357</td>
      <td>16.000000</td>
      <td>0.226527</td>
      <td>0.146622</td>
      <td>7.733218e+06</td>
    </tr>
    <tr>
      <th>std</th>
      <td>417.678187</td>
      <td>8.453927</td>
      <td>1.805867</td>
      <td>0.875852</td>
      <td>0.089419</td>
      <td>0.179981</td>
      <td>0.776280</td>
      <td>0.425810</td>
      <td>0.178649</td>
      <td>0.493560</td>
      <td>...</td>
      <td>0.081859</td>
      <td>0.025710</td>
      <td>0.047368</td>
      <td>0.214086</td>
      <td>0.203880</td>
      <td>0.171574</td>
      <td>12.972927</td>
      <td>0.676366</td>
      <td>0.685404</td>
      <td>2.315859e+06</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1863.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.388235</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>3.470769e+06</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2376.000000</td>
      <td>10.000000</td>
      <td>4.219508</td>
      <td>6.639876</td>
      <td>0.811765</td>
      <td>0.415686</td>
      <td>6.594413</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5.000000</td>
      <td>-0.390731</td>
      <td>-0.500000</td>
      <td>5.645376e+06</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2752.000000</td>
      <td>15.000000</td>
      <td>5.198497</td>
      <td>7.183112</td>
      <td>0.874510</td>
      <td>0.541176</td>
      <td>7.136483</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>12.000000</td>
      <td>0.406737</td>
      <td>0.275637</td>
      <td>7.573504e+06</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3104.000000</td>
      <td>22.000000</td>
      <td>5.802118</td>
      <td>7.727976</td>
      <td>0.921569</td>
      <td>0.654902</td>
      <td>7.595513</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>36.000000</td>
      <td>0.866025</td>
      <td>0.809017</td>
      <td>9.634816e+06</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3849.000000</td>
      <td>52.000000</td>
      <td>7.203406</td>
      <td>8.837971</td>
      <td>0.996078</td>
      <td>0.972549</td>
      <td>8.852808</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>36.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.481480e+07</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 55 columns</p>
</div>



Now that the data is transformed, we can also visualize the new aspect features. 


```python
# Visualize cover type VS the cosine of Aspect degerees
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(12,4))
sns.violinplot(x=train_df['Cover_Type'],y=train_df['ap_ew'],ax=ax1)
sns.histplot(train_df['ap_ew'],ax=ax2)
plt.show()
```


    
![png](backups/clear_cut_solution_files/backups/clear_cut_solution_15_0.png)
    


After the feature transformation, we see improved distinction in median values, espeically for cover type 6, where the median is notably higher than that of other cover types and the distribution is concentrated around the median.

#### Split data into train/dev

Then, we split the training data into a training data set (80%) and development data set (20%). We will also have a large, separate test data set. 


```python
train_data, train_labels, dev_data, dev_labels = fe.split_data(train_df)
train_data12, train_labels12, dev_data12, dev_labels12  = fe.split_data(train_df_12)
train_data36, train_labels36, dev_data36, dev_labels36  = fe.split_data(train_df_36)
```

#### Scale the data to have a mean of 0 and a variance of 1.


```python
standardize_features = ['Elevation','Slope', 'Horizontal_Distance_To_Hydrology',
       'Horizontal_Distance_To_Roadways',
       'Horizontal_Distance_To_Fire_Points','Elevation_squared']
train_data, train_scaler = fe.scale_training_data(standardize_features, train_data, scaler_type="standard")
dev_data = fe.scale_non_training_data(standardize_features, dev_data, train_scaler)

#generate scaling models for separate subsets 
train_data12, train_12_scaler = fe.scale_training_data(standardize_features, train_data12, scaler_type="standard")
train_data36, train_36_scaler = fe.scale_training_data(standardize_features, train_data36, scaler_type="standard")

dev_data12 = fe.scale_non_training_data(standardize_features, dev_data12, train_12_scaler)
dev_data36 = fe.scale_non_training_data(standardize_features, dev_data36, train_36_scaler)

```

#### Explore and confirm the shape of the data


```python
print("Training data shape: {0} Training labels shape: {1}\n".format(train_data.shape, train_labels.shape))
print("Dev data shape: {0} Dev labels shape: {1}\n".format(dev_data.shape, dev_labels.shape))
```

    Training data shape: (12096, 54) Training labels shape: (12096,)
    
    Dev data shape: (3024, 54) Dev labels shape: (3024,)
    


## Models

#### Random Forest


```python
# num_trees_list = [1,3,5,10,100]
num_trees_list = [100]
random_forest_models = []
random_forest_results = {}
for num_trees in num_trees_list:
    score, probabilities, random_forest_model = models.random_forest(num_trees, train_data, train_labels, dev_data, dev_labels)
    random_forest_results[score] = probabilities
    random_forest_models.append(random_forest_model)
```

    Random Forest Performance for 100 trees: 0.8849206349206349
    Mean Squared Error:  48.31117724867725



    
![png](backups/clear_cut_solution_files/backups/clear_cut_solution_25_1.png)
    


#### K-Nearest Neighbors


```python
# neighbor_list = [1,2,4, 7, 10]
neighbor_list = [1]
knn_models = []
knn_results = {}
for neighbor in neighbor_list:
    score, probabilities, knn_model = models.k_nearest_neighbors(neighbor,train_data, train_labels, dev_data, dev_labels)
    knn_results[score] = probabilities
    knn_models.append(knn_model)
    
```

    KNN 1 neighbors : accuracy = 0.9222883597883598
    Mean Squared Error:  30.93452380952381



    
![png](backups/clear_cut_solution_files/backups/clear_cut_solution_27_1.png)
    


#### Multi-Layer Perceptron


```python
mlp_results = {}
score, probabilities,mlp_model = models.multi_layer_perceptron(train_data, train_labels, dev_data, dev_labels)
mlp_results[score] = probabilities 
```

    MLP accuracy =  0.9411375661375662
    Mean Squared Error:  23.007275132275133



    
![png](backups/clear_cut_solution_files/backups/clear_cut_solution_29_1.png)
    


#### Generate Subset test df for testing results

#### Apply the same transformations


```python
test_data = manipulate_data(test_data)
test_data = fe.scale_non_training_data(standardize_features, test_data, train_scaler)
```


```python
#generate subset df for spearte testing.
y_pred = mlp_model.predict(test_data)
def gen_subset_test(y_pred):
    result = pd.DataFrame.from_dict(dict(zip(test_df_ID.to_list(),y_pred)), orient='index', columns=["Cover_Type"])
    type12_id = result[result.Cover_Type==12].index.to_list()
    type36_id = result[result.Cover_Type==36].index.to_list()
    test_12 = test_data1[test_data1.Id.isin(type12_id)]
    test_36 = test_data1[test_data1.Id.isin(type36_id)]
    test_12_id = test_12.Id
    test_36_id = test_36.Id
    return test_12, test_36, result, test_12_id, test_36_id
test_12, test_36, result, test_12_id, test_36_id = gen_subset_test(y_pred)
```


```python
#transofrm and scale the subset data with the respective scalers
test_12 = manipulate_data(test_12)
test_36 = manipulate_data(test_36)
test_12 = fe.scale_non_training_data(standardize_features, test_12, train_12_scaler)
test_36 = fe.scale_non_training_data(standardize_features, test_36, train_36_scaler)

#additional feature engineering for each subset
test_12 = manipulate_ct12(test_12)
test_36 = manipulate_ct36(test_36)
```


```python
#run the mlp model for subset of cover type 1 and 2
def subset_12():
    """This funciton trains a MLP model specifically on cover type 1 and 2 datasets"""
    model = MLPClassifier(alpha=1e-3, hidden_layer_sizes=(200,), random_state=0, max_iter=200) 
    model.fit(train_data12, train_labels12) 
    predictions = model.predict(dev_data12)
    ypred12 = model.predict(test_12)
    res12 = pd.DataFrame.from_dict(dict(zip(test_12_id.to_list(),ypred12)), orient='index', columns=["Cover_Type"])
    score = model.score(dev_data12, dev_labels12)
    probabilities = model.predict_proba(dev_data12)
    plot_confusion_matrix(model, dev_data12, dev_labels12, values_format = "d")
    plt.title("CT 1_2 Confusion Matrix")
    plt.plot()
    print("CT 1_2 accuracy = ",score)
    mse_nn = mean_squared_error(dev_labels12, predictions)
    print("Mean Squared Error: ", mse_nn)
           
    return score, probabilities, res12
score12, prob12, res12 = subset_12()
```

    CT 1_2 accuracy =  0.8090277777777778
    Mean Squared Error:  0.1909722222222222



    
![png](backups/clear_cut_solution_files/backups/clear_cut_solution_35_1.png)
    



```python
def subset_36():
    """This funciton trains a MLP model specifically on cover type 3 and 6 datasets"""
    model = MLPClassifier(alpha=1e-3, hidden_layer_sizes=(200,), random_state=0, max_iter=200) 
    model.fit(train_data36, train_labels36) 
    predictions = model.predict(dev_data36)
    ypred36 = model.predict(test_36)
    res36 = pd.DataFrame.from_dict(dict(zip(test_36_id.to_list(),ypred36)), orient='index', columns=["Cover_Type"])
    score = model.score(dev_data36, dev_labels36)
    probabilities = model.predict_proba(dev_data36)
    plot_confusion_matrix(model, dev_data36, dev_labels36, values_format = "d")
    plt.title("MLP Confusion Matrix")
    plt.plot()
    print("MLP accuracy = ",score)
    mse_nn = mean_squared_error(dev_labels36, predictions)
    print("Mean Squared Error: ", mse_nn)
           
    return score, probabilities, res36, ypred36
score36, prob36, res36, ypred36 = subset_36()
```

    MLP accuracy =  0.8530092592592593
    Mean Squared Error:  1.3229166666666667



    
![png](backups/clear_cut_solution_files/backups/clear_cut_solution_36_1.png)
    


#### Logistic Regression


```python
models.logistic_regression(train_data, train_labels, dev_data, dev_labels)
```

    Logistic Regression accuracy =  0.8197751322751323


#### Neural Network 


```python
# models.neural_network(train_data, train_labels, dev_data, dev_labels)
```

#### Ensemble
Here we will combine the three best performing models and implement a "voting" system to try to improve accuracy.


```python
predicted_classes, new_predictions = models.ensemble(mlp_results,knn_results,random_forest_results, dev_labels)
mse_ensemble = mean_squared_error(dev_labels, new_predictions)
accuracy = accuracy_score(dev_labels, new_predictions)
print("Mean Squared Error: ", mse_ensemble)
print("Accuracy: ", accuracy)
```

    Models disagreed on 2982/3024 dev examples.
    Mean Squared Error:  315.0671296296296
    Accuracy:  0.0006613756613756613


#### Examine and Compare Histograms of Predictions


```python
fig, axes = plt.subplots(2,2)
# Ensemble
axes[0,0].hist(new_predictions, bins=7,color = 'red') 
# MLP
axes[0,1].hist(predicted_classes[:,0], bins=7, color = 'orange') 
# KNN
axes[1,0].hist(predicted_classes[:,1], bins=7, color = 'green') 
# Random Forest
axes[1,1].hist(predicted_classes[:,2], bins=7, color = 'blue') 
```




    (array([439., 438.,   0., 429.,   0., 816., 902.]),
     array([0.        , 0.57142857, 1.14285714, 1.71428571, 2.28571429,
            2.85714286, 3.42857143, 4.        ]),
     <BarContainer object of 7 artists>)




    
![png](backups/clear_cut_solution_files/backups/clear_cut_solution_44_1.png)
    


### Test Results


```python
#generate predictions for test data
# random_forest_predictions = random_forest_models[-1].predict(test_data)
# knn_predictions = knn_models[0].predict(test_data)
# mlp_predictions = mlp_model.predict(test_data)
final_file = result[result.Cover_Type.isin([4,5,7])].append(res36).append(res12)
```

#### Generate Submission File


```python
def gen_submission(file):
#     result = pd.DataFrame.from_dict(dict(zip(test_df_ID.to_list(),y_pred)), orient='index', columns=["Cover_Type"])
    file.to_csv(f"submissions/cobsolidated_submission.csv",index_label="Id")

# gen_submission(random_forest_predictions, model="RandomForest")
# gen_submission(knn_predictions, model="KNN")
gen_submission(final_file)
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

    [NbConvertApp] Converting notebook clear_cut_solution.ipynb to python
    [NbConvertApp] Writing 15351 bytes to backups/clear_cut_solution.py
    [NbConvertApp] Converting notebook clear_cut_solution.ipynb to markdown
    [NbConvertApp] Support files will be in backups/clear_cut_solution_files/
    [NbConvertApp] Making directory backups/clear_cut_solution_files/backups
    [NbConvertApp] Making directory backups/clear_cut_solution_files/backups
    [NbConvertApp] Making directory backups/clear_cut_solution_files/backups
    [NbConvertApp] Making directory backups/clear_cut_solution_files/backups
    [NbConvertApp] Making directory backups/clear_cut_solution_files/backups
    [NbConvertApp] Making directory backups/clear_cut_solution_files/backups
    [NbConvertApp] Writing 23852 bytes to backups/clear_cut_solution.md
    [NbConvertApp] Converting notebook clear_cut_solution.ipynb to html
    [NbConvertApp] Writing 786754 bytes to backups/clear_cut_solution.html



```python

```
