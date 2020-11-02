# Forest Cover Type Prediction
#### Team: Clear-Cut Solution: Kevin Martin, Yang Jing, Justine Schabel

TODO: Introduce the project 

## Initial Setup
#### Import Required Libraries


```python
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

# So we can reload packages without restarting the kernel
import importlib
```


```python
# If you update the feature_engineering package, run this line so it updates without needing to restart the kernel
importlib.reload(fe)
```




    <module 'feature_engineering' from '/home/jupyter/mids-w207-final-project/feature_engineering.py'>




```python
# If you update the models package, run this line so it updates without needing to restart the kernel
importlib.reload(models)
```




    <module 'models' from '/home/jupyter/mids-w207-final-project/models.py'>



#### Load Data


```python
# Read in training data 
train_df = pd.read_csv("data/train.csv")
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
TODO: Now we'll log transform the features related to the distances. (explain why)

#### Add polynomial features
TODO: Explain why we're making Elevation polynomial

#### Drop irrelevant or problematic features
- We'll drop "Id" because it does not provide any meaning in the classifications.
- We'll drop "Hillshade_9am" because it has a high correlation with "Aspect" and "Hillshade_3pm".
- TODO: We'll also drop "Vertical_Distance_To_Hydrology" because _.



```python
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
      <th>type6st</th>
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
      <td>0.140410</td>
      <td>0.128704</td>
      <td>0.216534</td>
      <td>0.015675</td>
      <td>0.062698</td>
      <td>0.067063</td>
      <td>0.352183</td>
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
      <td>0.347423</td>
      <td>0.334883</td>
      <td>0.411896</td>
      <td>0.124217</td>
      <td>0.242428</td>
      <td>0.250140</td>
      <td>0.560491</td>
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
      <td>0.000000</td>
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
      <td>0.000000</td>
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
      <td>0.000000</td>
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
      <td>1.000000</td>
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
      <td>2.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.481480e+07</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 53 columns</p>
</div>



Now that the data is transformed, we can also visualize the new aspect features. 


```python
# Visualize cover type VS the cosine of Aspect degerees
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(12,4))
sns.violinplot(x=train_df['Cover_Type'],y=train_df['ap_ew'],ax=ax1)
sns.histplot(train_df['ap_ew'],ax=ax2)
plt.show()
```


![png](backups/clear_cut_solution_files/backups/clear_cut_solution_13_0.png)


After the feature transformation, we see improved distinction in median values, espeically for cover type 6, where the median is notably higher than that of other cover types and the distribution is concentrated around the median.

#### Split data into train/dev

Then, we split the training data into a training data set (80%) and development data set (20%). We will also have a large, separate test data set. 


```python
train_data, train_labels, dev_data, dev_labels = fe.split_data(train_df)
```

#### Scale the data to have a mean of 0 and a variance of 1.


```python
standardize_features = ['Elevation','Slope', 'Horizontal_Distance_To_Hydrology',
       'Horizontal_Distance_To_Roadways',
       'Horizontal_Distance_To_Fire_Points','Elevation_squared']
train_data, train_scaler = fe.scale_training_data(standardize_features, train_data, scaler_type="standard")
dev_data = fe.scale_non_training_data(standardize_features, dev_data, train_scaler)
```

#### Explore and confirm the shape of the data


```python
print("Training data shape: {0} Training labels shape: {1}\n".format(train_data.shape, train_labels.shape))
print("Dev data shape: {0} Dev labels shape: {1}\n".format(dev_data.shape, dev_labels.shape))
```

    Training data shape: (12096, 52) Training labels shape: (12096,)
    
    Dev data shape: (3024, 52) Dev labels shape: (3024,)
    


## Models

#### Random Forest


```python
num_trees_list = [1,3,5,10,100]
random_forest_models = []
random_forest_results = {}
for num_trees in num_trees_list:
    score, probabilities, random_forest_model = models.random_forest(num_trees, train_data, train_labels, dev_data, dev_labels)
    random_forest_results[score] = probabilities
    random_forest_models.append(random_forest_model)
```

    Random Forest Performance for 1 trees: 0.6696428571428571
    Mean Squared Error:  2.7880291005291005
    Random Forest Performance for 3 trees: 0.7311507936507936
    Mean Squared Error:  2.2943121693121693
    Random Forest Performance for 5 trees: 0.7218915343915344
    Mean Squared Error:  2.1686507936507935
    Random Forest Performance for 10 trees: 0.7476851851851852
    Mean Squared Error:  2.203042328042328
    Random Forest Performance for 100 trees: 0.7609126984126984
    Mean Squared Error:  1.8994708994708995



![png](backups/clear_cut_solution_files/backups/clear_cut_solution_23_1.png)



![png](backups/clear_cut_solution_files/backups/clear_cut_solution_23_2.png)



![png](backups/clear_cut_solution_files/backups/clear_cut_solution_23_3.png)



![png](backups/clear_cut_solution_files/backups/clear_cut_solution_23_4.png)



![png](backups/clear_cut_solution_files/backups/clear_cut_solution_23_5.png)


#### K-Nearest Neighbors


```python
neighbor_list = [1,2,4, 7, 10]
knn_models = []
knn_results = {}
for neighbor in neighbor_list:
    score, probabilities, knn_model = models.k_nearest_neighbors(neighbor,train_data, train_labels, dev_data, dev_labels)
    knn_results[score] = probabilities
    knn_models.append(knn_model)
    
```

    KNN 1 neighbors : accuracy = 0.8356481481481481
    Mean Squared Error:  1.3859126984126984
    KNN 2 neighbors : accuracy = 0.8128306878306878
    Mean Squared Error:  1.443121693121693
    KNN 4 neighbors : accuracy = 0.8237433862433863
    Mean Squared Error:  1.410383597883598
    KNN 7 neighbors : accuracy = 0.8115079365079365
    Mean Squared Error:  1.6335978835978835
    KNN 10 neighbors : accuracy = 0.810515873015873
    Mean Squared Error:  1.619047619047619



![png](backups/clear_cut_solution_files/backups/clear_cut_solution_25_1.png)



![png](backups/clear_cut_solution_files/backups/clear_cut_solution_25_2.png)



![png](backups/clear_cut_solution_files/backups/clear_cut_solution_25_3.png)



![png](backups/clear_cut_solution_files/backups/clear_cut_solution_25_4.png)



![png](backups/clear_cut_solution_files/backups/clear_cut_solution_25_5.png)


#### Multi-Layer Perceptron


```python
mlp_results = {}
score, probabilities,mlp_model = models.multi_layer_perceptron(train_data, train_labels, dev_data, dev_labels)
mlp_results[score] = probabilities 
```

    /opt/conda/lib/python3.7/site-packages/sklearn/neural_network/_multilayer_perceptron.py:585: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
      % self.max_iter, ConvergenceWarning)


    MLP accuracy =  0.8452380952380952
    Mean Squared Error:  1.3316798941798942



![png](backups/clear_cut_solution_files/backups/clear_cut_solution_27_2.png)


#### Logistic Regression


```python
models.logistic_regression(train_data, train_labels, dev_data, dev_labels)
```

    Logistic Regression accuracy =  0.7063492063492064


#### Neural Network 


```python
models.neural_network(train_data, train_labels, dev_data, dev_labels)
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

    Models disagreed on 2968/3024 dev examples.
    Mean Squared Error:  2.484457671957672
    Accuracy:  0.779431216931217


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




    (array([496., 241., 415., 506., 565., 361., 440.]),
     array([0.        , 0.85714286, 1.71428571, 2.57142857, 3.42857143,
            4.28571429, 5.14285714, 6.        ]),
     <BarContainer object of 7 artists>)




![png](backups/clear_cut_solution_files/backups/clear_cut_solution_35_1.png)


## Test Results
#### Read in test data


```python
# Read in training data 
test_data = pd.read_csv("data/test.csv")
# Preserve testing df ID for submission purpose
test_df_ID = test_data["Id"]
```

#### Apply the same transformations


```python
test_data = manipulate_data(test_data)
test_data = fe.scale_non_training_data(standardize_features, test_data, train_scaler)
```


```python
random_forest_predictions = random_forest_models[-1].predict(test_data)
knn_predictions = knn_models[0].predict(test_data)
mlp_predictions = mlp_model.predict(test_data)
```

#### Generate Submission File


```python
def gen_submission(y_pred,model):
    result = pd.DataFrame.from_dict(dict(zip(test_df_ID.to_list(),y_pred)), orient='index', columns=["Cover_Type"])
    result.to_csv(f"submissions/submission{model}.csv",index_label="Id")

gen_submission(random_forest_predictions, model="RandomForest")
gen_submission(knn_predictions, model="KNN")
gen_submission(mlp_predictions, model="MLP")
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
