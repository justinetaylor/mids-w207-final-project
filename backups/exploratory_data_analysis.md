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
import warnings
warnings.simplefilter("ignore")
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



    
![png](backups/exploratory_data_analysis_files/backups/exploratory_data_analysis_14_1.png)
    


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


    
![png](backups/exploratory_data_analysis_files/backups/exploratory_data_analysis_21_0.png)
    


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


    
![png](backups/exploratory_data_analysis_files/backups/exploratory_data_analysis_23_0.png)
    


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


    
![png](backups/exploratory_data_analysis_files/backups/exploratory_data_analysis_28_0.png)
    


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
df1 = df1.pivot("soil_type","Cover_Type","yes")
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
      <th>Cover_Type</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
    </tr>
    <tr>
      <th>soil_type</th>
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
      <th>Soil_Type1</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>121.0</td>
      <td>139.0</td>
      <td>NaN</td>
      <td>95.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Soil_Type10</th>
      <td>9.0</td>
      <td>81.0</td>
      <td>717.0</td>
      <td>170.0</td>
      <td>64.0</td>
      <td>1101.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Soil_Type11</th>
      <td>5.0</td>
      <td>67.0</td>
      <td>89.0</td>
      <td>24.0</td>
      <td>154.0</td>
      <td>67.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Soil_Type12</th>
      <td>24.0</td>
      <td>203.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Soil_Type13</th>
      <td>17.0</td>
      <td>84.0</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>305.0</td>
      <td>66.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Soil_Type14</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>128.0</td>
      <td>NaN</td>
      <td>37.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Soil_Type16</th>
      <td>9.0</td>
      <td>14.0</td>
      <td>5.0</td>
      <td>40.0</td>
      <td>9.0</td>
      <td>37.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Soil_Type17</th>
      <td>2.0</td>
      <td>7.0</td>
      <td>34.0</td>
      <td>350.0</td>
      <td>131.0</td>
      <td>88.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Soil_Type18</th>
      <td>NaN</td>
      <td>16.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Soil_Type19</th>
      <td>15.0</td>
      <td>13.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>18.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Soil_Type2</th>
      <td>NaN</td>
      <td>3.0</td>
      <td>283.0</td>
      <td>94.0</td>
      <td>61.0</td>
      <td>182.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Soil_Type20</th>
      <td>41.0</td>
      <td>55.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>37.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Soil_Type21</th>
      <td>12.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>Soil_Type22</th>
      <td>275.0</td>
      <td>54.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>16.0</td>
    </tr>
    <tr>
      <th>Soil_Type23</th>
      <td>376.0</td>
      <td>149.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>157.0</td>
      <td>3.0</td>
      <td>72.0</td>
    </tr>
    <tr>
      <th>Soil_Type24</th>
      <td>128.0</td>
      <td>72.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>11.0</td>
      <td>15.0</td>
      <td>31.0</td>
    </tr>
    <tr>
      <th>Soil_Type25</th>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Soil_Type26</th>
      <td>7.0</td>
      <td>19.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>28.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Soil_Type27</th>
      <td>7.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>Soil_Type28</th>
      <td>NaN</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Soil_Type29</th>
      <td>407.0</td>
      <td>554.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>254.0</td>
      <td>NaN</td>
      <td>76.0</td>
    </tr>
    <tr>
      <th>Soil_Type3</th>
      <td>NaN</td>
      <td>12.0</td>
      <td>133.0</td>
      <td>799.0</td>
      <td>NaN</td>
      <td>18.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Soil_Type30</th>
      <td>81.0</td>
      <td>144.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>480.0</td>
      <td>NaN</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>Soil_Type31</th>
      <td>114.0</td>
      <td>97.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>87.0</td>
      <td>7.0</td>
      <td>27.0</td>
    </tr>
    <tr>
      <th>Soil_Type32</th>
      <td>230.0</td>
      <td>255.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>100.0</td>
      <td>30.0</td>
      <td>70.0</td>
    </tr>
    <tr>
      <th>Soil_Type33</th>
      <td>184.0</td>
      <td>184.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>115.0</td>
      <td>66.0</td>
      <td>67.0</td>
    </tr>
    <tr>
      <th>Soil_Type34</th>
      <td>NaN</td>
      <td>12.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>Soil_Type35</th>
      <td>3.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>98.0</td>
    </tr>
    <tr>
      <th>Soil_Type36</th>
      <td>NaN</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>Soil_Type37</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>34.0</td>
    </tr>
    <tr>
      <th>Soil_Type38</th>
      <td>80.0</td>
      <td>7.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>641.0</td>
    </tr>
    <tr>
      <th>Soil_Type39</th>
      <td>79.0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>575.0</td>
    </tr>
    <tr>
      <th>Soil_Type4</th>
      <td>5.0</td>
      <td>20.0</td>
      <td>462.0</td>
      <td>133.0</td>
      <td>129.0</td>
      <td>87.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>Soil_Type40</th>
      <td>49.0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>407.0</td>
    </tr>
    <tr>
      <th>Soil_Type5</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>55.0</td>
      <td>39.0</td>
      <td>NaN</td>
      <td>71.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Soil_Type6</th>
      <td>NaN</td>
      <td>7.0</td>
      <td>248.0</td>
      <td>244.0</td>
      <td>NaN</td>
      <td>151.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Soil_Type8</th>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Soil_Type9</th>
      <td>1.0</td>
      <td>9.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Wilderness_Area1</th>
      <td>1062.0</td>
      <td>1134.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>856.0</td>
      <td>NaN</td>
      <td>545.0</td>
    </tr>
    <tr>
      <th>Wilderness_Area2</th>
      <td>181.0</td>
      <td>66.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>252.0</td>
    </tr>
    <tr>
      <th>Wilderness_Area3</th>
      <td>917.0</td>
      <td>940.0</td>
      <td>863.0</td>
      <td>NaN</td>
      <td>1304.0</td>
      <td>962.0</td>
      <td>1363.0</td>
    </tr>
    <tr>
      <th>Wilderness_Area4</th>
      <td>NaN</td>
      <td>20.0</td>
      <td>1297.0</td>
      <td>2160.0</td>
      <td>NaN</td>
      <td>1198.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
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


    
![png](backups/exploratory_data_analysis_files/backups/exploratory_data_analysis_35_0.png)
    


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


    
![png](backups/exploratory_data_analysis_files/backups/exploratory_data_analysis_39_0.png)
    


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
!jupyter nbconvert exploratory_data_analysis.ipynb --to="python" --output="backups/exploratory_data_analysis"
!jupyter nbconvert exploratory_data_analysis.ipynb --to markdown --output="backups/exploratory_data_analysis"

# Also archiving this bad boy
!jupyter nbconvert exploratory_data_analysis.ipynb --to html --output="backups/exploratory_data_analysis"
```

    [NbConvertApp] WARNING | Config option `kernel_spec_manager_class` not recognized by `NbConvertApp`.
    [NbConvertApp] Converting notebook exploratory_data_analysis.ipynb to python
    [NbConvertApp] Writing 9803 bytes to backups/exploratory_data_analysis.py
    [NbConvertApp] WARNING | Config option `kernel_spec_manager_class` not recognized by `NbConvertApp`.
    [NbConvertApp] Converting notebook exploratory_data_analysis.ipynb to markdown
    [NbConvertApp] Support files will be in backups/exploratory_data_analysis_files/
    [NbConvertApp] Making directory backups/exploratory_data_analysis_files/backups
    [NbConvertApp] Making directory backups/exploratory_data_analysis_files/backups
    [NbConvertApp] Making directory backups/exploratory_data_analysis_files/backups
    [NbConvertApp] Making directory backups/exploratory_data_analysis_files/backups
    [NbConvertApp] Making directory backups/exploratory_data_analysis_files/backups
    [NbConvertApp] Making directory backups/exploratory_data_analysis_files/backups
    [NbConvertApp] Writing 29724 bytes to backups/exploratory_data_analysis.md
    [NbConvertApp] WARNING | Config option `kernel_spec_manager_class` not recognized by `NbConvertApp`.
    [NbConvertApp] Converting notebook exploratory_data_analysis.ipynb to html
    [NbConvertApp] Writing 2414368 bytes to backups/exploratory_data_analysis.html

