<a href="https://colab.research.google.com/github/justinetaylor/mids-w207-final-project/blob/yang_branch/clear_cut_solution.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Forest Cover Type Prediction
#### Team: Clear-Cut Solution: Kevin Martin, Yang Jing, Justine Schabel

## Initial Setup
### Import Required Libraries


```python
# This tells matplotlib not to try opening a new window for each plot.
%matplotlib inline

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
```


```python
# Mount the drive for file storage
# from google.colab import drive
# drive.mount('/content/drive')
```


```python
# os.chdir('/content/drive/My Drive/W207-Final-Project')
```

### Load Data


```python
# Read in training data 
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")
```

## Exploratory Data Analysis

#### Basic Attributes

First, we check the data attributes, quality and shape.


```python
# Examine shape 
print(train_df.shape)

# Briefly examine feature attributes for the training data 
train_df.describe()
```

    (15120, 56)





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
      <th>Id</th>
      <th>Elevation</th>
      <th>Aspect</th>
      <th>Slope</th>
      <th>Horizontal_Distance_To_Hydrology</th>
      <th>Vertical_Distance_To_Hydrology</th>
      <th>Horizontal_Distance_To_Roadways</th>
      <th>Hillshade_9am</th>
      <th>Hillshade_Noon</th>
      <th>Hillshade_3pm</th>
      <th>...</th>
      <th>Soil_Type32</th>
      <th>Soil_Type33</th>
      <th>Soil_Type34</th>
      <th>Soil_Type35</th>
      <th>Soil_Type36</th>
      <th>Soil_Type37</th>
      <th>Soil_Type38</th>
      <th>Soil_Type39</th>
      <th>Soil_Type40</th>
      <th>Cover_Type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>15120.00000</td>
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
      <td>15120.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>7560.50000</td>
      <td>2749.322553</td>
      <td>156.676653</td>
      <td>16.501587</td>
      <td>227.195701</td>
      <td>51.076521</td>
      <td>1714.023214</td>
      <td>212.704299</td>
      <td>218.965608</td>
      <td>135.091997</td>
      <td>...</td>
      <td>0.045635</td>
      <td>0.040741</td>
      <td>0.001455</td>
      <td>0.006746</td>
      <td>0.000661</td>
      <td>0.002249</td>
      <td>0.048148</td>
      <td>0.043452</td>
      <td>0.030357</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>4364.91237</td>
      <td>417.678187</td>
      <td>110.085801</td>
      <td>8.453927</td>
      <td>210.075296</td>
      <td>61.239406</td>
      <td>1325.066358</td>
      <td>30.561287</td>
      <td>22.801966</td>
      <td>45.895189</td>
      <td>...</td>
      <td>0.208699</td>
      <td>0.197696</td>
      <td>0.038118</td>
      <td>0.081859</td>
      <td>0.025710</td>
      <td>0.047368</td>
      <td>0.214086</td>
      <td>0.203880</td>
      <td>0.171574</td>
      <td>2.000066</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.00000</td>
      <td>1863.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-146.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>99.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3780.75000</td>
      <td>2376.000000</td>
      <td>65.000000</td>
      <td>10.000000</td>
      <td>67.000000</td>
      <td>5.000000</td>
      <td>764.000000</td>
      <td>196.000000</td>
      <td>207.000000</td>
      <td>106.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>7560.50000</td>
      <td>2752.000000</td>
      <td>126.000000</td>
      <td>15.000000</td>
      <td>180.000000</td>
      <td>32.000000</td>
      <td>1316.000000</td>
      <td>220.000000</td>
      <td>223.000000</td>
      <td>138.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>11340.25000</td>
      <td>3104.000000</td>
      <td>261.000000</td>
      <td>22.000000</td>
      <td>330.000000</td>
      <td>79.000000</td>
      <td>2270.000000</td>
      <td>235.000000</td>
      <td>235.000000</td>
      <td>167.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>15120.00000</td>
      <td>3849.000000</td>
      <td>360.000000</td>
      <td>52.000000</td>
      <td>1343.000000</td>
      <td>554.000000</td>
      <td>6890.000000</td>
      <td>254.000000</td>
      <td>254.000000</td>
      <td>248.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>7.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 56 columns</p>
</div>




```python
# Check data types
train_df.dtypes
```




    Id                                    int64
    Elevation                             int64
    Aspect                                int64
    Slope                                 int64
    Horizontal_Distance_To_Hydrology      int64
    Vertical_Distance_To_Hydrology        int64
    Horizontal_Distance_To_Roadways       int64
    Hillshade_9am                         int64
    Hillshade_Noon                        int64
    Hillshade_3pm                         int64
    Horizontal_Distance_To_Fire_Points    int64
    Wilderness_Area1                      int64
    Wilderness_Area2                      int64
    Wilderness_Area3                      int64
    Wilderness_Area4                      int64
    Soil_Type1                            int64
    Soil_Type2                            int64
    Soil_Type3                            int64
    Soil_Type4                            int64
    Soil_Type5                            int64
    Soil_Type6                            int64
    Soil_Type7                            int64
    Soil_Type8                            int64
    Soil_Type9                            int64
    Soil_Type10                           int64
    Soil_Type11                           int64
    Soil_Type12                           int64
    Soil_Type13                           int64
    Soil_Type14                           int64
    Soil_Type15                           int64
    Soil_Type16                           int64
    Soil_Type17                           int64
    Soil_Type18                           int64
    Soil_Type19                           int64
    Soil_Type20                           int64
    Soil_Type21                           int64
    Soil_Type22                           int64
    Soil_Type23                           int64
    Soil_Type24                           int64
    Soil_Type25                           int64
    Soil_Type26                           int64
    Soil_Type27                           int64
    Soil_Type28                           int64
    Soil_Type29                           int64
    Soil_Type30                           int64
    Soil_Type31                           int64
    Soil_Type32                           int64
    Soil_Type33                           int64
    Soil_Type34                           int64
    Soil_Type35                           int64
    Soil_Type36                           int64
    Soil_Type37                           int64
    Soil_Type38                           int64
    Soil_Type39                           int64
    Soil_Type40                           int64
    Cover_Type                            int64
    dtype: object



#### Verify Dataset Is Balanced


```python
# Visualize the distribution of labels, "Cover_Type"
plt.figure(figsize=(6,4))
sns.displot(train_df["Cover_Type"],rug=True)
plt.show()
```


    <Figure size 432x288 with 0 Axes>



![png](backups/clear-cut-solution_files/backups/clear-cut-solution_14_1.png)


Here we can see that the training data has a somewhat uniform distribution of covertype and this tells us that our data set is balanced. 

#### Check For Null Values


```python
# Check for NA values
# `.isna()` returns a df with bools the first `.sum()` returns series, second is int 
print("There are {} NA values in the training data".format(train_df.isna().sum().sum()))
print("There are {} NA values in the test data\n".format(train_df.isna().sum().sum()))
print("There are {} values in the training data".format(train_df.count()[0]))
print("There are {} values in the test data".format(test_df.count()[0]))
```

    There are 0 NA values in the training data
    There are 0 NA values in the test data
    
    There are 15120 values in the training data
    There are 565892 values in the test data


There are no null values in the dataset. 

#### Distributions of Numeric Columns


```python
# Collect numeric feature column names - so we can easily access these columns when modifying them 
num_cols = ['Elevation', 'Slope','Aspect',
       'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
       'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon',
       'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points']
```


```python
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
```


![png](backups/clear-cut-solution_files/backups/clear-cut-solution_21_0.png)


Here we can see the distribution are skewed for a few variables, espcially in the "distance" related ones, such as "Horizontal_Diestance_To_Fire_points". A log-transformation may improve the model performance. Also, there are zeros in these variables, we need to add 1 before performing the log transofrmation.


```python
# Visualize the distribution of numerical columns with Cover Type
fig, axes = plt.subplots(rows,2,figsize=(20,20))
for i in range(col_count):
    for j in range(2):
        col= train_df[num_cols[j+2*(i//2)]]
        sns.violinplot(x=train_df['Cover_Type'], y= col, ax=axes[i//2][j])
        axes[i//2][j].grid()
```


![png](backups/clear-cut-solution_files/backups/clear-cut-solution_23_0.png)


First, we can see there is a relationship between the cover type and elevation. The difference in the other fetures by cover type seem less significant. Cover type 1 and 2 share a lot of similar features. We need to find a way to magnify the signal between the 2 cover types. 

We also see there is not much differences in the relationship between the cover type and Aspect. The Aspect is expressed in degrees, and 0 degree and 360 degree is the same thing but represented differently. This probably contributed to poor distinction among the lables. In feature engineering, we'll extract the sine and cosine values to normalize this feature.


#### Correlation


```python
# Rank correlations with "cover type"
# This was train_corr1=train_df1.corr(), but train_df1 isn't defined yet? - Maybe we can remove this, since we have the heatmap below?
train_corr1=train_df.corr()
train_corr1['Cover_Type'].abs().sort_values(ascending=False)[:31]
```




    Cover_Type                            1.000000
    Soil_Type38                           0.257810
    Soil_Type39                           0.240384
    Wilderness_Area1                      0.230117
    Soil_Type29                           0.218564
    Soil_Type40                           0.205851
    Soil_Type22                           0.195993
    Soil_Type23                           0.158762
    Soil_Type32                           0.132312
    Soil_Type12                           0.129985
    Soil_Type10                           0.128972
    Wilderness_Area3                      0.122146
    Soil_Type35                           0.114327
    Id                                    0.108363
    Horizontal_Distance_To_Roadways       0.105662
    Soil_Type24                           0.100797
    Hillshade_Noon                        0.098905
    Horizontal_Distance_To_Fire_Points    0.089389
    Slope                                 0.087722
    Soil_Type31                           0.079882
    Soil_Type33                           0.078955
    Wilderness_Area4                      0.075774
    Vertical_Distance_To_Hydrology        0.075647
    Soil_Type37                           0.071210
    Hillshade_3pm                         0.053399
    Soil_Type20                           0.053013
    Soil_Type17                           0.042453
    Soil_Type13                           0.040528
    Soil_Type19                           0.031824
    Soil_Type4                            0.027816
    Soil_Type5                            0.027692
    Name: Cover_Type, dtype: float64




```python
# Explore correlations between numerical features
train_corr = train_df[num_cols].corr()

# Plot a heat map for correlations
ax = plt.figure(figsize=(8,8))
sns.heatmap(train_corr, xticklabels=train_corr.columns.values, yticklabels=train_corr.columns.values)
plt.show()
```


![png](backups/clear-cut-solution_files/backups/clear-cut-solution_28_0.png)


From the above, "Hillshade_9am" has strong correlation with "Hillshade_3pm" and "Aspect". We may drop this feature to avoid multi-collinearity.

#### Soil Types

Now, we'll isolate and explore the distribution of soil types. 


```python
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

```


```python
# Visualize cover type VS soil type in a pivot table. 
df1 = soil_df_unpivoted.groupby(["Cover_Type","soil_type"], as_index=False).count()
df1 = df1.pivot("Cover_Type","soil_type","yes")
df1
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
      <th>soil_type</th>
      <th>Soil_Type1</th>
      <th>Soil_Type10</th>
      <th>Soil_Type11</th>
      <th>Soil_Type12</th>
      <th>Soil_Type13</th>
      <th>Soil_Type14</th>
      <th>Soil_Type16</th>
      <th>Soil_Type17</th>
      <th>Soil_Type18</th>
      <th>Soil_Type19</th>
      <th>...</th>
      <th>Soil_Type4</th>
      <th>Soil_Type40</th>
      <th>Soil_Type5</th>
      <th>Soil_Type6</th>
      <th>Soil_Type8</th>
      <th>Soil_Type9</th>
      <th>Wilderness_Area1</th>
      <th>Wilderness_Area2</th>
      <th>Wilderness_Area3</th>
      <th>Wilderness_Area4</th>
    </tr>
    <tr>
      <th>Cover_Type</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>9.0</td>
      <td>5.0</td>
      <td>24.0</td>
      <td>17.0</td>
      <td>NaN</td>
      <td>9.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>15.0</td>
      <td>...</td>
      <td>5.0</td>
      <td>49.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1062.0</td>
      <td>181.0</td>
      <td>917.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>81.0</td>
      <td>67.0</td>
      <td>203.0</td>
      <td>84.0</td>
      <td>NaN</td>
      <td>14.0</td>
      <td>7.0</td>
      <td>16.0</td>
      <td>13.0</td>
      <td>...</td>
      <td>20.0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>7.0</td>
      <td>1.0</td>
      <td>9.0</td>
      <td>1134.0</td>
      <td>66.0</td>
      <td>940.0</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>121.0</td>
      <td>717.0</td>
      <td>89.0</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>34.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>462.0</td>
      <td>NaN</td>
      <td>55.0</td>
      <td>248.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>863.0</td>
      <td>1297.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>139.0</td>
      <td>170.0</td>
      <td>24.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>128.0</td>
      <td>40.0</td>
      <td>350.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>133.0</td>
      <td>NaN</td>
      <td>39.0</td>
      <td>244.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2160.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NaN</td>
      <td>64.0</td>
      <td>154.0</td>
      <td>NaN</td>
      <td>305.0</td>
      <td>NaN</td>
      <td>9.0</td>
      <td>131.0</td>
      <td>44.0</td>
      <td>18.0</td>
      <td>...</td>
      <td>129.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>856.0</td>
      <td>NaN</td>
      <td>1304.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>95.0</td>
      <td>1101.0</td>
      <td>67.0</td>
      <td>NaN</td>
      <td>66.0</td>
      <td>37.0</td>
      <td>37.0</td>
      <td>88.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>87.0</td>
      <td>NaN</td>
      <td>71.0</td>
      <td>151.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>962.0</td>
      <td>1198.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>7.0</td>
      <td>407.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>545.0</td>
      <td>252.0</td>
      <td>1363.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>7 rows × 42 columns</p>
</div>



As we can see in the pivot table above, there are similar combinations of soil types for different "cover type". We'll combine the soil types that share same "cover types" to reduce dimensionality. Further, "cover type 1" and "cover type 2" , "cover type 3" and "cover type 6" share many overlapping features. To magnify the signal, we'll combine features as an extra feature where there is a difference between the 2 pairs of cover types.


```python
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
```


![png](backups/clear-cut-solution_files/backups/clear-cut-solution_35_0.png)


Here we can examine the relationship between soil type and cover type for each soil type. # TODO: Discuss more

#### Wilderness Types

Now, we'll isolate and explore the distribution of wilderness types. 


```python
wilderness_list =['Wilderness_Area1','Wilderness_Area2','Wilderness_Area3','Wilderness_Area4']

# Visualize the distribution of wilderness area and "cover type"
fig, axes = plt.subplots(2,2,figsize=(24,12))
for i in range(4):
    sns.violinplot(y=train_df['Cover_Type'],x=train_df[wilderness_list[i]], ax=axes[i//2,i%2])
plt.show()
```


![png](backups/clear-cut-solution_files/backups/clear-cut-solution_39_0.png)


## Feature Engineering 

#### Scale Hillshade

Now we'll normalize the "Hillsdale" variables by dividing them by 255. TODO: Can we explain why? We'll scale them all later? 


```python
fe1_cols = ['Hillshade_9am', 'Hillshade_Noon',
       'Hillshade_3pm']
train_df[fe1_cols] = train_df[fe1_cols]/255
```

#### Create New Soil Types

Now we'll create additional features to magnify the differences betweeen cover type1 and 2, and covery type3 and 6.


```python
# Create additional features to magnify the differences between cover type 1 and 2
# Combine soil type 2,18,25,3,36,6,8,28,34 and wildness area 4 as only cover type 2 appers under these features
train_df["type2stwa4"] = train_df["Soil_Type6"] + train_df["Wilderness_Area4"] +  \
train_df["Soil_Type2"]+ train_df["Soil_Type18"] +  train_df["Soil_Type25"] +  \
train_df["Soil_Type3"] + train_df["Soil_Type36"]+ \
train_df["Soil_Type8"] + train_df["Soil_Type34"]+ train_df["Soil_Type28"]

# Combine soil type 20, 23, 24, 31, 33 and 34 as only cover type 6 appears under these features but not cover type 3.
train_df["type6st"] = train_df["Soil_Type20"] + train_df["Soil_Type23"]+ \
train_df["Soil_Type24"] +  train_df["Soil_Type31"] + train_df["Soil_Type33"] +  train_df["Soil_Type34"]
```

#### Drop Non-Existant Soil Types

Now we'll drop soil types that don't exist in the training set. Then we will combine soil types 35, 38, 39 and 40 because they have a very similar distribution. 


```python
# Remove soil type 7 and 15 due to no data
train_df.drop(columns=["Soil_Type7", "Soil_Type15"], inplace=True)

# Remove soil type 19, 37, 34, 21, 27,36,28,8,25 due to no limited data - TODO: should we be dropping these? 
train_df.drop(columns=["Soil_Type19", "Soil_Type37","Soil_Type34", "Soil_Type21","Soil_Type27", "Soil_Type36","Soil_Type28","Soil_Type8", "Soil_Type25"], inplace=True)
```

#### Combine Similar Soil Types


```python
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
```

    (15120, 53)


#### Transform Aspect 

Now we'll transform the Asepct feature.
TODO: Explain more


```python
# Convert aspect into sine and cosine values 
train_df["ap_ew"] = np.sin(train_df["Aspect"]/180*np.pi)
train_df["ap_ns"] = np.cos(train_df["Aspect"]/180*np.pi)

# Drop Aspect column
train_df.drop(columns= ["Aspect"], inplace=True)
              
# Check shape is as expected
print(train_df.shape)
```

    (15120, 54)



```python
# Visualize cover type VS the cosine of Aspect degerees
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(12,4))
sns.violinplot(x=train_df['Cover_Type'],y=train_df['ap_ew'],ax=ax1)
sns.histplot(train_df['ap_ew'],ax=ax2)
plt.show()
```


![png](backups/clear-cut-solution_files/backups/clear-cut-solution_52_0.png)


After the feature transformation, we see improved distinction in median values, espeically for cover type 6, where the median is notably higher than that of other cover types  and the distribution is concentrated around the median. 

#### Log and Polynomial Transformations

Now we'll log transform the features related to the distances.


```python
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
```

    Vertical_Distance_To_Hydrology Minimum:  -146


#### Drop Id Column


```python
# TODO: Can this be removed? We should drop id in the training data we actually use. 
# Make a copy of train_df for modelling
train_df1 = train_df.copy()

# Drop Id column as it is not a meaningful feature.
train_df.drop(columns=["Id"],inplace=True)
test_df.drop(columns=["Id"],inplace=True)
```

#### Drop Hillshade_9am

Hillshade_9am has a strong correlation with Hillshade_3pm and Aspect. TODO: If we're only dropping Hillshade_9am here we can drop it directly  


```python
all_features = set(train_df.columns.to_list())

# Select features to drop. 
to_drop = set(['Hillshade_9am'])

sel_features = list(all_features - to_drop)
train_df =train_df[sel_features]
```

#### Split Data into Train/Dev/Test

Then, we split the training data into a training data set (80%) and development data set (20%). We will also have a large, separate test data set. 


```python
# Split training data (labeled) into 80% training and 20% dev) and randomly sample 
training_data = train_df.sample(frac=0.8)
dev_data_df = train_df.drop(training_data.index)

# Examine shape of both data sets
print(training_data.shape)
print(dev_data_df.shape)

# Briefly examine feature attributes for the training data 
training_data.describe()
```

    (12096, 53)
    (3024, 53)





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
      <th>Soil_Type32</th>
      <th>Cover_Type</th>
      <th>Soil_Type2</th>
      <th>Soil_Type16</th>
      <th>Soil_Type23</th>
      <th>Vertical_Distance_To_Hydrology</th>
      <th>soil_type35383940</th>
      <th>Horizontal_Distance_To_Roadways</th>
      <th>Slope</th>
      <th>Soil_Type14</th>
      <th>...</th>
      <th>Soil_Type13</th>
      <th>Hillshade_3pm</th>
      <th>st6w4</th>
      <th>Soil_Type39</th>
      <th>Soil_Type12</th>
      <th>Horizontal_Distance_To_Fire_Points</th>
      <th>Soil_Type24</th>
      <th>Wilderness_Area2</th>
      <th>Soil_Type4</th>
      <th>Wilderness_Area3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>12096.000000</td>
      <td>12096.000000</td>
      <td>12096.000000</td>
      <td>12096.000000</td>
      <td>12096.000000</td>
      <td>12096.000000</td>
      <td>12096.000000</td>
      <td>12096.000000</td>
      <td>12096.000000</td>
      <td>12096.000000</td>
      <td>...</td>
      <td>12096.000000</td>
      <td>12096.000000</td>
      <td>12096.000000</td>
      <td>12096.000000</td>
      <td>12096.000000</td>
      <td>12096.000000</td>
      <td>12096.000000</td>
      <td>12096.000000</td>
      <td>12096.00000</td>
      <td>12096.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.045635</td>
      <td>4.000000</td>
      <td>0.040509</td>
      <td>0.007523</td>
      <td>0.049686</td>
      <td>5.245071</td>
      <td>0.126075</td>
      <td>7.276575</td>
      <td>16.449074</td>
      <td>0.011574</td>
      <td>...</td>
      <td>0.032655</td>
      <td>0.531023</td>
      <td>0.352596</td>
      <td>0.041171</td>
      <td>0.014964</td>
      <td>7.212667</td>
      <td>0.016865</td>
      <td>0.031911</td>
      <td>0.05539</td>
      <td>0.418981</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.208701</td>
      <td>1.997642</td>
      <td>0.197158</td>
      <td>0.086413</td>
      <td>0.217304</td>
      <td>0.283198</td>
      <td>0.331947</td>
      <td>0.733137</td>
      <td>8.443615</td>
      <td>0.106963</td>
      <td>...</td>
      <td>0.177740</td>
      <td>0.179201</td>
      <td>0.559571</td>
      <td>0.198693</td>
      <td>0.121412</td>
      <td>0.649249</td>
      <td>0.128771</td>
      <td>0.175771</td>
      <td>0.22875</td>
      <td>0.493413</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4.990433</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4.990433</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5.023881</td>
      <td>0.000000</td>
      <td>6.806829</td>
      <td>10.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.419608</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>6.776507</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5.187386</td>
      <td>0.000000</td>
      <td>7.288244</td>
      <td>15.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.541176</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.250636</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.000000</td>
      <td>6.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5.420535</td>
      <td>0.000000</td>
      <td>7.787382</td>
      <td>22.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.654902</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.669028</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>7.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>6.552508</td>
      <td>1.000000</td>
      <td>8.847647</td>
      <td>50.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>0.972549</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>8.873468</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.00000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 53 columns</p>
</div>




```python
# Split into data and labels
train_data = training_data.drop(columns=["Cover_Type"])
train_labels = training_data["Cover_Type"]
dev_data = dev_data_df.drop(columns=["Cover_Type"])
dev_labels = dev_data_df["Cover_Type"]
test_data = test_df

# Double check the shape
print(train_data.shape)
print(dev_data.shape)
```

    (12096, 52)
    (3024, 52)


#### Scale Data
Additionally, we will scale the training data to have a mean of 0 and a variance of 1. Then we will retrieve the original training mean and variance for each feature and use that to standardize the development data.


```python
#compile a list for columns for scaling
ss_cols = ['Elevation','Slope', 'Horizontal_Distance_To_Hydrology',
       'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
       'Horizontal_Distance_To_Fire_Points','elv_pwd']
```


```python
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
```


```python
# Explore and confirm the shape of the data
print("Training data shape: {0} Training labels shape: {1}\n".format(train_data.shape, train_labels.shape))
print("Dev data shape: {0} Dev labels shape: {1}\n".format(dev_data.shape, dev_data.shape))
print("Test data shape: ", test_data.shape)
```

    Training data shape: (12096, 52) Training labels shape: (12096,)
    
    Dev data shape: (3024, 52) Dev labels shape: (3024, 52)
    
    Test data shape:  (565892, 54)


## Models
#### Random Forest


```python
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
```

    Random Forest Performance for 1 trees: 0.5651455026455027
    Mean Squared Error:  3.693452380952381
    Random Forest Performance for 3 trees: 0.7056878306878307
    Mean Squared Error:  2.5400132275132274
    Random Forest Performance for 5 trees: 0.7423941798941799
    Mean Squared Error:  2.3392857142857144
    Random Forest Performance for 10 trees: 0.7380952380952381
    Mean Squared Error:  2.363095238095238
    Random Forest Performance for 100 trees: 0.7493386243386243
    Mean Squared Error:  2.3191137566137567



![png](backups/clear-cut-solution_files/backups/clear-cut-solution_68_1.png)



![png](backups/clear-cut-solution_files/backups/clear-cut-solution_68_2.png)



![png](backups/clear-cut-solution_files/backups/clear-cut-solution_68_3.png)



![png](backups/clear-cut-solution_files/backups/clear-cut-solution_68_4.png)



![png](backups/clear-cut-solution_files/backups/clear-cut-solution_68_5.png)


#### Naive Bayes (Bernoulli)


```python
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
```

    BernoulliNB for alph = 0.01: accuracy = 0.5995370370370371
    
    
    



![png](backups/clear-cut-solution_files/backups/clear-cut-solution_70_1.png)



```python
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
```

#### K-Nearest Neighbors


```python
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
```

    KNN 1 neighbors : accuracy = 0.8382936507936508
    Mean Squared Error:  1.382936507936508
    KNN 2 neighbors : accuracy = 0.8184523809523809
    Mean Squared Error:  1.3640873015873016
    KNN 4 neighbors : accuracy = 0.8158068783068783
    Mean Squared Error:  1.5148809523809523
    KNN 7 neighbors : accuracy = 0.8062169312169312
    Mean Squared Error:  1.7599206349206349
    KNN 10 neighbors : accuracy = 0.7913359788359788
    Mean Squared Error:  1.8941798941798942



![png](backups/clear-cut-solution_files/backups/clear-cut-solution_73_1.png)



![png](backups/clear-cut-solution_files/backups/clear-cut-solution_73_2.png)



![png](backups/clear-cut-solution_files/backups/clear-cut-solution_73_3.png)



![png](backups/clear-cut-solution_files/backups/clear-cut-solution_73_4.png)



![png](backups/clear-cut-solution_files/backups/clear-cut-solution_73_5.png)


#### Multi-layer Perceptron


```python
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
```

    /opt/conda/lib/python3.7/site-packages/sklearn/neural_network/_multilayer_perceptron.py:585: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (300) reached and the optimization hasn't converged yet.
      % self.max_iter, ConvergenceWarning)


    MLP accuracy =  0.8508597883597884
    Mean Squared Error:  1.273478835978836



![png](backups/clear-cut-solution_files/backups/clear-cut-solution_75_2.png)


#### Logistic Regression


```python
# Logistic regression
def LR():
    model = LogisticRegression(random_state=0, multi_class='ovr',solver='lbfgs', max_iter = 300)
    model.fit(train_data, train_labels)
    score = model.score(dev_data,dev_labels)
    print("Logistic Regression accuracy = ",score)
LR()
```

    Logistic Regression accuracy =  0.6841931216931217


#### Neural Network with Tensorflow


```python

```


```python
tf.executing_eagerly()
```




    True




```python
train_data[:1].shape
train_data.to_numpy().shape, train_labels.to_numpy().shape
```




    ((12096, 52), (12096,))




```python
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

```

    Train on 12096 samples
    Epoch 1/5
    12096/12096 [==============================] - 3s 229us/sample - loss: 1.0713 - acc: 0.5930
    Epoch 2/5
    12096/12096 [==============================] - 2s 175us/sample - loss: 0.7468 - acc: 0.6924
    Epoch 3/5
    12096/12096 [==============================] - 2s 192us/sample - loss: 0.6880 - acc: 0.7173
    Epoch 4/5
    12096/12096 [==============================] - 2s 183us/sample - loss: 0.6539 - acc: 0.7281
    Epoch 5/5
    12096/12096 [==============================] - 2s 178us/sample - loss: 0.6256 - acc: 0.7423





    <tensorflow.python.keras.callbacks.History at 0x7f544fc89250>




```python
model.evaluate(dev_data.to_numpy(),  dev_labels.to_numpy(), verbose=2)
```

    3024/1 - 0s - loss: 0.7147 - acc: 0.7503





    [0.6021279484506638, 0.7503307]



#### Ensemble

Here we will combine the three best performing models and implement a "voting" system to try to improve accuracy. 


```python
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
```

    Models disagreed on 878/3024 dev examples.
    Mean Squared Error:  2.099537037037037
    Accuracy:  0.03009259259259259



```python
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
```




    (array([454., 268., 413., 484., 548., 383., 474.]),
     array([0.        , 0.85714286, 1.71428571, 2.57142857, 3.42857143,
            4.28571429, 5.14285714, 6.        ]),
     <BarContainer object of 7 artists>)




![png](backups/clear-cut-solution_files/backups/clear-cut-solution_86_1.png)


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
!jupyter nbconvert clear_cut_solution.ipynb --to="python" --output="backups/clear-cut-solution"
!jupyter nbconvert clear_cut_solution.ipynb --to markdown --output="backups/clear-cut-solution"

# Also archiving this bad boy
!jupyter nbconvert clear_cut_solution.ipynb --to html --output="backups/clear-cut-solution"
```

    [NbConvertApp] WARNING | Config option `kernel_spec_manager_class` not recognized by `NbConvertApp`.
    [NbConvertApp] Converting notebook clear_cut_solution.ipynb to python
    [NbConvertApp] Writing 26429 bytes to backups/clear-cut-solution.py
    [NbConvertApp] WARNING | Config option `kernel_spec_manager_class` not recognized by `NbConvertApp`.
    [NbConvertApp] Converting notebook clear_cut_solution.ipynb to markdown
    [NbConvertApp] Support files will be in backups/clear-cut-solution_files/
    [NbConvertApp] Making directory backups/clear-cut-solution_files/backups
    [NbConvertApp] Making directory backups/clear-cut-solution_files/backups
    [NbConvertApp] Making directory backups/clear-cut-solution_files/backups
    [NbConvertApp] Making directory backups/clear-cut-solution_files/backups
    [NbConvertApp] Making directory backups/clear-cut-solution_files/backups
    [NbConvertApp] Making directory backups/clear-cut-solution_files/backups
    [NbConvertApp] Making directory backups/clear-cut-solution_files/backups
    [NbConvertApp] Making directory backups/clear-cut-solution_files/backups
    [NbConvertApp] Making directory backups/clear-cut-solution_files/backups
    [NbConvertApp] Making directory backups/clear-cut-solution_files/backups
    [NbConvertApp] Making directory backups/clear-cut-solution_files/backups
    [NbConvertApp] Making directory backups/clear-cut-solution_files/backups
    [NbConvertApp] Making directory backups/clear-cut-solution_files/backups
    [NbConvertApp] Making directory backups/clear-cut-solution_files/backups
    [NbConvertApp] Making directory backups/clear-cut-solution_files/backups
    [NbConvertApp] Making directory backups/clear-cut-solution_files/backups
    [NbConvertApp] Making directory backups/clear-cut-solution_files/backups
    [NbConvertApp] Making directory backups/clear-cut-solution_files/backups
    [NbConvertApp] Making directory backups/clear-cut-solution_files/backups
    [NbConvertApp] Writing 52450 bytes to backups/clear-cut-solution.md

