# Forest Cover Type Prediction
#### Team: Clear-Cut Solution: Kevin Martin, Yang Jing, Justine Schabel


```python
# This tells matplotlib not to try opening a new window for each plot.
%matplotlib inline

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

```

## Data Engineering

### Load Data


```python
# Read in training data 
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")
```

### Initial Data Exploration

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
sns.distplot(train_df['Cover_Type'],rug=True)
plt.show()
```


![png](backups/clear-cut-solution_files/backups/clear-cut-solution_7_0.png)


Here we can see that the training data has a somewhat uniform distribution of covertype and this tells us that our data set is balanced. 


```python
sns.violinplot(x=train_df['Cover_Type'],y=train_df['Elevation'])
plt.show()
```


![png](backups/clear-cut-solution_files/backups/clear-cut-solution_9_0.png)


Here, we can see there is a relationship between the cover type and elevation. 


```python
# get NA values

print("There are {} NA values in the training data".format(train_df.isna().sum().sum()))
print("There are {} NA values in the test data".format(train_df.isna().sum().sum()))
    # `.isna()` returns a df with bools the first `.sum()` returns series, second is int 
print()
print("There are {} values in the training data".format(train_df.count()[0]))
print("There are {} values in the test data".format(test_df.count()[0]))
```

    There are 0 NA values in the training data
    There are 0 NA values in the test data
    
    There are 15120 values in the training data
    There are 565892 values in the test data


No null values in the dataset. Also noted the "aspect" variable has a value between 0 and 359. This is expressed in degrees, compared to "true north". Will conver this ino sine(EW) and cosine(NS) values. 

### Feature Engineering 1
Now we'll transform the "Aspect" into cosine and sine values to improve the representation of directions. 


```python
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
```

    /opt/conda/lib/python3.7/site-packages/ipykernel_launcher.py:13: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      del sys.path[0]
    /opt/conda/lib/python3.7/site-packages/ipykernel_launcher.py:14: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      
    /opt/conda/lib/python3.7/site-packages/ipykernel_launcher.py:15: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      from ipykernel import kernelapp as app





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
      <th>asp_ew</th>
      <th>asp_ns</th>
      <th>jitter</th>
      <th>asp_ew_jit</th>
      <th>asp_ns_jit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.670229</td>
      <td>0.742154</td>
      <td>1.542300</td>
      <td>1.033694</td>
      <td>1.144624</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.521551</td>
      <td>0.853220</td>
      <td>1.284087</td>
      <td>-0.669717</td>
      <td>1.095609</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.696080</td>
      <td>0.717964</td>
      <td>1.203784</td>
      <td>0.837930</td>
      <td>0.864274</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.873312</td>
      <td>-0.487161</td>
      <td>1.151990</td>
      <td>-1.006047</td>
      <td>-0.561205</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.850904</td>
      <td>0.525322</td>
      <td>1.121546</td>
      <td>0.954328</td>
      <td>0.589173</td>
    </tr>
  </tbody>
</table>
</div>




![png](backups/clear-cut-solution_files/backups/clear-cut-solution_14_2.png)



![png](backups/clear-cut-solution_files/backups/clear-cut-solution_14_3.png)



![png](backups/clear-cut-solution_files/backups/clear-cut-solution_14_4.png)



![png](backups/clear-cut-solution_files/backups/clear-cut-solution_14_5.png)



```python
#drop Aspect column
train_df.drop(columns=["Aspect"], inplace=True)
```

Now, we'll isolate and explore the distribution of soil types. 


```python
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
```


![png](backups/clear-cut-solution_files/backups/clear-cut-solution_17_0.png)


As we can see in the histogram above, there is an uneven distribution of occurances of soil types.


```python
# Explore correlations between features
train_corr=train_df.corr()
# Rank correlations with "cover type"
train_corr['Cover_Type'].abs().sort_values(ascending=False)
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
    Soil_Type9                            0.027012
    Soil_Type36                           0.025726
    Soil_Type21                           0.024410
    Soil_Type27                           0.023109
    Soil_Type2                            0.022627
    Soil_Type14                           0.022019
    Soil_Type26                           0.017184
    Soil_Type3                            0.016393
    Elevation                             0.016090
    Soil_Type1                            0.015069
    Wilderness_Area2                      0.014994
    Soil_Type28                           0.012202
    asp_ew                                0.012002
    Horizontal_Distance_To_Hydrology      0.010515
    Hillshade_9am                         0.010286
    Soil_Type11                           0.010228
    Soil_Type16                           0.008793
    Soil_Type25                           0.008133
    Soil_Type8                            0.008133
    Soil_Type6                            0.006521
    Soil_Type18                           0.006312
    Soil_Type34                           0.003470
    Soil_Type30                           0.001393
    asp_ns                                0.000829
    Soil_Type7                                 NaN
    Soil_Type15                                NaN
    Name: Cover_Type, dtype: float64




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


![png](backups/clear-cut-solution_files/backups/clear-cut-solution_20_0.png)


Here we can examine the relationship between soil type and cover type for each soil type. # TODO: Discuss more

Now, we'll isolate and explore the distribution of wilderness types. 


```python
wilderness_list =['Wilderness_Area1','Wilderness_Area2','Wilderness_Area3','Wilderness_Area4']

# Visualize the distribution of wilderness area and "cover type"
fig, axes = plt.subplots(2,2,figsize=(24,12))
for i in range(4):
    sns.violinplot(y=train_df['Cover_Type'],x=train_df[wilderness_list[i]], ax=axes[i//2,i%2])
plt.show()
```


![png](backups/clear-cut-solution_files/backups/clear-cut-solution_23_0.png)


### Feature Engineering 2

I'm going to hold off on dropping any soil types and just transform them to the new type


```python
# # Remove soil type 7 and 15 due to no data
# train_df.drop(columns=["Soil_Type7", "Soil_Type15"], inplace=True)

# # Remove soil type 19, 37, 34, 21, 27,36,9, 28,8,25 due to no limited data - TODO: should we be dropping these? 
# train_df.drop(columns=["Soil_Type19", "Soil_Type37","Soil_Type34", "Soil_Type21","Soil_Type27", "Soil_Type36","Soil_Type9", "Soil_Type28","Soil_Type8", "Soil_Type25"], inplace=True)

# # Combine soil type 35,38,39, 40
# train_df["soil_type35383940"] = train_df["Soil_Type38"] +  train_df["Soil_Type39"] + train_df["Soil_Type40"] +  train_df["Soil_Type35"]
# train_df.drop(columns=["Soil_Type35","Soil_Type38", "Soil_Type39",'Soil_Type40'], inplace=True)

# # Check shape is as expected
# print(train_df.shape)

```


```python
#drop Id column as it is not a meaningful feature.
train_df.drop(columns=["Id"],inplace=True)
test_df.drop(columns=["Id"],inplace=True)
```

## Soil Type

Right now we have 40 composite types of soil. We are going to re-categorize those into their component parts with a 1 for if a component is present. 0 if they are not present. 

#### Adjustments to the data 

Looking throught the data, we see several areas where the data can be cleaned. The USDA which is the cabinet department which oversees the forest service, publishes a guide on Soil Taxonomy. (https://www.nrcs.usda.gov/wps/portal/nrcs/detail/soils/home/?cid=nrcs142p2_053577). This guide was very helpful in interpreting several areas where the data can be trimmed down. 

* Filler words
  * "Family / "families" denotes a larger soil group such as Leighcan or Cryaquolls, but these words themselves don't actually carry any information about the group. 
  * "complex" is another word that appears in the docuement but doesn't actually convey any information itself
  * "typic" just means that a soil fits into the most general subgroup of the family listed. In the context of the soil types listed, it is useless because only a few soil types list it, and those that don't list it don't list a more specific subgroup that they are a part of. So, they would also generally be assumed to be typic (see guide to soil taxonomy linked above.)
* Cryaquolis and aquolis types don't exist
  * Looking through the guide to soil taxonomy, there is no such type as cryaquolis and aquolis. The simplest interpretation is that these are typos. Cryaquolls and aquolls do exist (see link above). I will change the words accordingly.
  
#### Possible future work

* See if the "very/extremely" qualifiers on stony should be removed. Do they mean the same thing?
* See if changing "rubbly", "bouldery", "stoney" to all one indicator would be better. Do they all mean the same thing?


```python
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

## Remove the features that have very low occurence rates

# print(trans_soil_df.sum(axis=0))
    # print occurence rates of the various features

# remove low occurence soil types
occ_lim = 1400 #0, 2,100,300,900,1400 default # remove columns that have less than occ_lim examples in the data
high_occ_ser = trans_soil_df.sum(axis=0) >= occ_lim
high_occ_names = [entry for entry in high_occ_ser.index if high_occ_ser[entry]]
trans_soil_df = trans_soil_df[high_occ_names]
display(trans_soil_df)

# combine the new soil features with the existing freatures in a single df
df_new = train_df.drop(columns=og_soil_col_names)
df_new = pd.concat([df_new, trans_soil_df],axis=1)
df_new = df_new[[col for col in df_new if col not in ["Cover_Type"]]+["Cover_Type"]] #want cover type as last column
display(df_new)
```

    -- New Feature Names --
    ['aquolls', 'borohemists', 'bross', 'bullwark', 'catamount', 'cathedral', 'como', 'cryaquepts', 'cryaquolls', 'cryoborolis', 'cryorthents', 'cryumbrepts', 'extremely_bouldery', 'extremely_stony', 'gateview', 'gothic', 'granile', 'haploborolis', 'legault', 'leighcan', 'limber', 'moran', 'pachic_argiborolis', 'ratake', 'rock_land', 'rock_outcrop', 'rogert', 'rubbly', 'stony', 'supervisor', 'till_substratum', 'troutville', 'vanet', 'very_stony', 'warm', 'wetmore'] 
    



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
      <th>aquolls</th>
      <th>borohemists</th>
      <th>bross</th>
      <th>bullwark</th>
      <th>catamount</th>
      <th>cathedral</th>
      <th>como</th>
      <th>cryaquepts</th>
      <th>cryaquolls</th>
      <th>cryoborolis</th>
      <th>...</th>
      <th>rogert</th>
      <th>rubbly</th>
      <th>stony</th>
      <th>supervisor</th>
      <th>till_substratum</th>
      <th>troutville</th>
      <th>vanet</th>
      <th>very_stony</th>
      <th>warm</th>
      <th>wetmore</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>15115</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15116</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15117</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15118</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15119</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>15120 rows × 36 columns</p>
</div>



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
      <th>bullwark</th>
      <th>catamount</th>
      <th>como</th>
      <th>cryaquolls</th>
      <th>extremely_stony</th>
      <th>legault</th>
      <th>leighcan</th>
      <th>moran</th>
      <th>ratake</th>
      <th>rock_land</th>
      <th>rock_outcrop</th>
      <th>rubbly</th>
      <th>vanet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>15115</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15116</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15117</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15118</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15119</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>15120 rows × 13 columns</p>
</div>



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
      <th>Vertical_Distance_To_Hydrology</th>
      <th>Horizontal_Distance_To_Roadways</th>
      <th>Hillshade_9am</th>
      <th>Hillshade_Noon</th>
      <th>Hillshade_3pm</th>
      <th>Horizontal_Distance_To_Fire_Points</th>
      <th>Wilderness_Area1</th>
      <th>...</th>
      <th>extremely_stony</th>
      <th>legault</th>
      <th>leighcan</th>
      <th>moran</th>
      <th>ratake</th>
      <th>rock_land</th>
      <th>rock_outcrop</th>
      <th>rubbly</th>
      <th>vanet</th>
      <th>Cover_Type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2596</td>
      <td>3</td>
      <td>258</td>
      <td>0</td>
      <td>510</td>
      <td>221</td>
      <td>232</td>
      <td>148</td>
      <td>6279</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2590</td>
      <td>2</td>
      <td>212</td>
      <td>-6</td>
      <td>390</td>
      <td>220</td>
      <td>235</td>
      <td>151</td>
      <td>6225</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2804</td>
      <td>9</td>
      <td>268</td>
      <td>65</td>
      <td>3180</td>
      <td>234</td>
      <td>238</td>
      <td>135</td>
      <td>6121</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2785</td>
      <td>18</td>
      <td>242</td>
      <td>118</td>
      <td>3090</td>
      <td>238</td>
      <td>238</td>
      <td>122</td>
      <td>6211</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2595</td>
      <td>2</td>
      <td>153</td>
      <td>-1</td>
      <td>391</td>
      <td>220</td>
      <td>234</td>
      <td>150</td>
      <td>6172</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>15115</th>
      <td>2607</td>
      <td>23</td>
      <td>258</td>
      <td>7</td>
      <td>660</td>
      <td>170</td>
      <td>251</td>
      <td>214</td>
      <td>1282</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>15116</th>
      <td>2603</td>
      <td>19</td>
      <td>633</td>
      <td>195</td>
      <td>618</td>
      <td>249</td>
      <td>221</td>
      <td>91</td>
      <td>1325</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>15117</th>
      <td>2492</td>
      <td>25</td>
      <td>365</td>
      <td>117</td>
      <td>335</td>
      <td>250</td>
      <td>220</td>
      <td>83</td>
      <td>1187</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>15118</th>
      <td>2487</td>
      <td>28</td>
      <td>218</td>
      <td>101</td>
      <td>242</td>
      <td>229</td>
      <td>237</td>
      <td>119</td>
      <td>932</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>15119</th>
      <td>2475</td>
      <td>34</td>
      <td>319</td>
      <td>78</td>
      <td>270</td>
      <td>189</td>
      <td>244</td>
      <td>164</td>
      <td>914</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>15120 rows × 29 columns</p>
</div>



```python
# We'll just reassign the train_df here to be equal to df_new
train_df = df_new
```

### Additional Data Mungling

#### Split into traingin and dev

split the training data into a training data set (80%) and development data set (20%). We will also have a large, separate test data set. 


```python
# Split training data (labeled) into 80% training and 20% dev) and randomly sample 
training_data = train_df.sample(frac=0.8)
dev_data_df = train_df.drop(training_data.index)

# Examine shape of both data sets
print(training_data.shape)
print(dev_data_df.shape)

# Briefly examine feature attributes for the training data 
display(training_data.describe())
display(dev_data_df.describe())
```

    (12096, 29)
    (3024, 29)



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
      <th>Vertical_Distance_To_Hydrology</th>
      <th>Horizontal_Distance_To_Roadways</th>
      <th>Hillshade_9am</th>
      <th>Hillshade_Noon</th>
      <th>Hillshade_3pm</th>
      <th>Horizontal_Distance_To_Fire_Points</th>
      <th>Wilderness_Area1</th>
      <th>...</th>
      <th>extremely_stony</th>
      <th>legault</th>
      <th>leighcan</th>
      <th>moran</th>
      <th>ratake</th>
      <th>rock_land</th>
      <th>rock_outcrop</th>
      <th>rubbly</th>
      <th>vanet</th>
      <th>Cover_Type</th>
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
      <td>12096.00000</td>
      <td>12096.000000</td>
      <td>12096.00000</td>
      <td>12096.000000</td>
      <td>12096.000000</td>
      <td>12096.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2751.317874</td>
      <td>16.464451</td>
      <td>225.819610</td>
      <td>50.661293</td>
      <td>1716.772652</td>
      <td>212.892692</td>
      <td>219.010582</td>
      <td>134.959491</td>
      <td>1517.698826</td>
      <td>0.238343</td>
      <td>...</td>
      <td>0.410466</td>
      <td>0.148727</td>
      <td>0.294643</td>
      <td>0.120536</td>
      <td>0.09838</td>
      <td>0.152612</td>
      <td>0.43502</td>
      <td>0.329448</td>
      <td>0.094246</td>
      <td>3.984127</td>
    </tr>
    <tr>
      <th>std</th>
      <td>416.257896</td>
      <td>8.460626</td>
      <td>206.602678</td>
      <td>60.565495</td>
      <td>1328.347663</td>
      <td>30.331696</td>
      <td>22.870946</td>
      <td>45.800508</td>
      <td>1104.388638</td>
      <td>0.426088</td>
      <td>...</td>
      <td>0.491939</td>
      <td>0.355834</td>
      <td>0.455901</td>
      <td>0.325600</td>
      <td>0.29784</td>
      <td>0.359628</td>
      <td>0.49578</td>
      <td>0.470032</td>
      <td>0.292183</td>
      <td>2.002540</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1863.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-146.000000</td>
      <td>0.000000</td>
      <td>59.000000</td>
      <td>99.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2376.750000</td>
      <td>10.000000</td>
      <td>67.000000</td>
      <td>5.000000</td>
      <td>765.000000</td>
      <td>196.000000</td>
      <td>207.000000</td>
      <td>107.000000</td>
      <td>731.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2755.000000</td>
      <td>15.000000</td>
      <td>180.000000</td>
      <td>32.000000</td>
      <td>1315.000000</td>
      <td>220.000000</td>
      <td>223.000000</td>
      <td>138.000000</td>
      <td>1265.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3105.000000</td>
      <td>22.000000</td>
      <td>324.000000</td>
      <td>79.000000</td>
      <td>2271.000000</td>
      <td>235.000000</td>
      <td>235.000000</td>
      <td>167.000000</td>
      <td>1996.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>1.00000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3849.000000</td>
      <td>50.000000</td>
      <td>1343.000000</td>
      <td>554.000000</td>
      <td>6890.000000</td>
      <td>254.000000</td>
      <td>254.000000</td>
      <td>248.000000</td>
      <td>6993.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.00000</td>
      <td>1.000000</td>
      <td>1.00000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>7.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 29 columns</p>
</div>



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
      <th>Vertical_Distance_To_Hydrology</th>
      <th>Horizontal_Distance_To_Roadways</th>
      <th>Hillshade_9am</th>
      <th>Hillshade_Noon</th>
      <th>Hillshade_3pm</th>
      <th>Horizontal_Distance_To_Fire_Points</th>
      <th>Wilderness_Area1</th>
      <th>...</th>
      <th>extremely_stony</th>
      <th>legault</th>
      <th>leighcan</th>
      <th>moran</th>
      <th>ratake</th>
      <th>rock_land</th>
      <th>rock_outcrop</th>
      <th>rubbly</th>
      <th>vanet</th>
      <th>Cover_Type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3024.000000</td>
      <td>3024.000000</td>
      <td>3024.000000</td>
      <td>3024.000000</td>
      <td>3024.000000</td>
      <td>3024.000000</td>
      <td>3024.000000</td>
      <td>3024.000000</td>
      <td>3024.000000</td>
      <td>3024.000000</td>
      <td>...</td>
      <td>3024.000000</td>
      <td>3024.000000</td>
      <td>3024.000000</td>
      <td>3024.000000</td>
      <td>3024.00000</td>
      <td>3024.000000</td>
      <td>3024.000000</td>
      <td>3024.000000</td>
      <td>3024.000000</td>
      <td>3024.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2741.341270</td>
      <td>16.650132</td>
      <td>232.700066</td>
      <td>52.737434</td>
      <td>1703.025463</td>
      <td>211.950728</td>
      <td>218.785714</td>
      <td>135.622024</td>
      <td>1484.941138</td>
      <td>0.236111</td>
      <td>...</td>
      <td>0.408730</td>
      <td>0.146825</td>
      <td>0.284061</td>
      <td>0.127646</td>
      <td>0.09127</td>
      <td>0.158399</td>
      <td>0.436839</td>
      <td>0.333664</td>
      <td>0.098545</td>
      <td>4.063492</td>
    </tr>
    <tr>
      <th>std</th>
      <td>423.287169</td>
      <td>8.426834</td>
      <td>223.377780</td>
      <td>63.847319</td>
      <td>1312.019492</td>
      <td>31.456746</td>
      <td>22.526782</td>
      <td>46.275802</td>
      <td>1081.728307</td>
      <td>0.424761</td>
      <td>...</td>
      <td>0.491681</td>
      <td>0.353990</td>
      <td>0.451041</td>
      <td>0.333750</td>
      <td>0.28804</td>
      <td>0.365176</td>
      <td>0.496077</td>
      <td>0.471599</td>
      <td>0.298099</td>
      <td>1.989204</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1874.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>-103.000000</td>
      <td>30.000000</td>
      <td>0.000000</td>
      <td>103.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2369.750000</td>
      <td>10.000000</td>
      <td>67.000000</td>
      <td>5.000000</td>
      <td>750.000000</td>
      <td>195.000000</td>
      <td>207.000000</td>
      <td>106.000000</td>
      <td>726.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2738.500000</td>
      <td>16.000000</td>
      <td>175.000000</td>
      <td>33.000000</td>
      <td>1320.000000</td>
      <td>219.000000</td>
      <td>222.000000</td>
      <td>139.000000</td>
      <td>1216.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3101.000000</td>
      <td>23.000000</td>
      <td>331.000000</td>
      <td>80.000000</td>
      <td>2255.000000</td>
      <td>235.000000</td>
      <td>235.000000</td>
      <td>168.000000</td>
      <td>1951.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3849.000000</td>
      <td>52.000000</td>
      <td>1294.000000</td>
      <td>411.000000</td>
      <td>6297.000000</td>
      <td>254.000000</td>
      <td>254.000000</td>
      <td>248.000000</td>
      <td>6723.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.00000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>7.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 29 columns</p>
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

    (12096, 28)
    (3024, 28)



```python
train_data.columns
```




    Index(['Elevation', 'Slope', 'Horizontal_Distance_To_Hydrology',
           'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
           'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
           'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area1',
           'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4', 'asp_ew',
           'asp_ns', 'bullwark', 'catamount', 'como', 'cryaquolls',
           'extremely_stony', 'legault', 'leighcan', 'moran', 'ratake',
           'rock_land', 'rock_outcrop', 'rubbly', 'vanet'],
          dtype='object')



##### Normalize the numeric data

Additionally, we will scale the training data to have a mean of 0 and a variance of 1. Then we will retrieve the original training mean and variance for each feature and use that to standardize the development data.


```python
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


```

    (12096, 28)
    (3024, 28)



```python
# Double check shape
print(train_data.shape, dev_data.shape)
```

    (12096, 28) (3024, 28)



```python
# Explore and confirm the shape of the data
print("Training data shape: {0} Training labels shape: {1}".format(train_data.shape, train_labels.shape))
print("Dev data shape: {0} Dev labels shape: {1}".format(dev_data.shape, dev_labels.shape))
print("Test data shape: ", test_data.shape)
```

    Training data shape: (12096, 28) Training labels shape: (12096,)
    Dev data shape: (3024, 28) Dev labels shape: (3024,)
    Test data shape:  (565892, 54)



```python
# Examine Training Data 
dev_data.head()
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
      <th>Vertical_Distance_To_Hydrology</th>
      <th>Horizontal_Distance_To_Roadways</th>
      <th>Hillshade_9am</th>
      <th>Hillshade_Noon</th>
      <th>Hillshade_3pm</th>
      <th>Horizontal_Distance_To_Fire_Points</th>
      <th>Wilderness_Area1</th>
      <th>...</th>
      <th>cryaquolls</th>
      <th>extremely_stony</th>
      <th>legault</th>
      <th>leighcan</th>
      <th>moran</th>
      <th>ratake</th>
      <th>rock_land</th>
      <th>rock_outcrop</th>
      <th>rubbly</th>
      <th>vanet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14</th>
      <td>-0.596573</td>
      <td>-1.473291</td>
      <td>-0.768752</td>
      <td>-0.770459</td>
      <td>-0.785047</td>
      <td>0.366210</td>
      <td>0.917771</td>
      <td>0.350240</td>
      <td>3.696588</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0.143384</td>
      <td>-1.827890</td>
      <td>-0.066893</td>
      <td>-0.341154</td>
      <td>1.470480</td>
      <td>0.234329</td>
      <td>0.830320</td>
      <td>0.415744</td>
      <td>3.735525</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>-0.029593</td>
      <td>0.890698</td>
      <td>-0.478327</td>
      <td>0.038616</td>
      <td>1.177622</td>
      <td>1.322345</td>
      <td>-0.393991</td>
      <td>-1.396538</td>
      <td>4.088676</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>35</th>
      <td>0.357202</td>
      <td>0.299700</td>
      <td>0.078320</td>
      <td>-0.506271</td>
      <td>2.621581</td>
      <td>0.267299</td>
      <td>-1.049872</td>
      <td>-0.763331</td>
      <td>2.351897</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>44</th>
      <td>-0.176143</td>
      <td>-1.355092</td>
      <td>-0.633220</td>
      <td>-0.456736</td>
      <td>-0.042741</td>
      <td>0.531061</td>
      <td>0.742869</td>
      <td>0.131893</td>
      <td>4.553206</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 28 columns</p>
</div>



## Models
#### Random Forest


```python
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
```

    Random Forest Performance for 1 trees: 0.6418650793650794
    Random Forest Performance for 3 trees: 0.7195767195767195
    Random Forest Performance for 5 trees: 0.7255291005291006
    Random Forest Performance for 10 trees: 0.7447089947089947
    Random Forest Performance for 100 trees: 0.7605820105820106



![png](backups/clear-cut-solution_files/backups/clear-cut-solution_40_1.png)



![png](backups/clear-cut-solution_files/backups/clear-cut-solution_40_2.png)



![png](backups/clear-cut-solution_files/backups/clear-cut-solution_40_3.png)



![png](backups/clear-cut-solution_files/backups/clear-cut-solution_40_4.png)



![png](backups/clear-cut-solution_files/backups/clear-cut-solution_40_5.png)


#### Naive Bayes (Bernoulli)


```python
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
```

    BernoulliNB for alph = 0.01: accuracy = 0.5773809523809523
    
    
    



![png](backups/clear-cut-solution_files/backups/clear-cut-solution_42_1.png)


#### K-Nearest Neighbors


```python
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
```

    KNN 1 neighbors : accuracy = 0.753968253968254
    KNN 2 neighbors : accuracy = 0.7136243386243386
    KNN 4 neighbors : accuracy = 0.7450396825396826
    KNN 7 neighbors : accuracy = 0.7414021164021164
    KNN 10 neighbors : accuracy = 0.7371031746031746



![png](backups/clear-cut-solution_files/backups/clear-cut-solution_44_1.png)



![png](backups/clear-cut-solution_files/backups/clear-cut-solution_44_2.png)



![png](backups/clear-cut-solution_files/backups/clear-cut-solution_44_3.png)



![png](backups/clear-cut-solution_files/backups/clear-cut-solution_44_4.png)



![png](backups/clear-cut-solution_files/backups/clear-cut-solution_44_5.png)


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
    model = MLPClassifier(alpha=1e-3, hidden_layer_sizes=(100,), random_state=0, max_iter=300) 
    model.fit(train_data, train_labels) 
    score = model.score(dev_data, dev_labels)
    print("MLP accuracy = ",score)

    
MLP()
```

### End matter

#### Acknowledgements/Sources

* That helpful stack overflow post
  * the url for it
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
!jupyter nbconvert clear-cut-solution.ipynb --to="python" --output="backups/clear-cut-solution"
!jupyter nbconvert clear-cut-solution.ipynb --to markdown --output="backups/clear-cut-solution"

# Also archiving this bad boy
!jupyter nbconvert clear-cut-solution.ipynb --to html --output="backups/clear-cut-solution"
```


```python

```


```python

```
