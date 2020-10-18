```python
# This tells matplotlib not to try opening a new window for each plot.
%matplotlib inline

# Libraries for reading, cleaning and plotting the dataa
import numpy as np 
import pandas as pd 
import csv
import matplotlib.pyplot as plt


# Libraries for models 
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
```


```python
# Read in training data 
training_data = []
with open('data/train.csv', newline='') as csvfile:
    train_data = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in train_data:
        training_data.append(row)
            
# Convert to a numpy array of type int (except for the label row)
training_data = np.array(training_data[1:]).astype(int)   

# Read in test data
testing_data = []
with open('data/test.csv', newline='') as csvfile:
    test_data = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in test_data:
        testing_data.append(row)

# The testing file is huge so only read in max_test_data
max_test_data = 30001
test_data = np.array(testing_data[1:max_test_data]).astype(int)        
```


```python
# Shuffle the input: create a random permutation of the integers between 0 and the number of data points and apply this
# permutation to X and Y.
# NOTE: Each time you run this cell, you'll re-shuffle the data, resulting in a different ordering.
shuffle = np.random.permutation(np.arange(training_data.shape[0]))
training_data = training_data[shuffle]

# Split training data (labeled) into 80% training and 20% dev) and skip over the id column (it doesn't add an information)
# Immediately cast train data as floats so we can normalize it later 
split_index = int(len(training_data) * 0.8)
train_data = training_data[:split_index, 1:-1].astype(np.float64)
train_labels = training_data[:split_index, -1]
dev_data = training_data[split_index:, 1:-1].astype(np.float64)
dev_labels = training_data[split_index:, -1]
test_data = test_data[:,1:]

# Retrieve the mean and standard deviation of each feature - axis=0 is for going along the columns, keepdims forces the dimensions to stay the same
# Only compute it for the first ten features (they're numeric - not one hot or categorical)
num_columns = 10
# To avoid dividing by 0, add 1e-10 to the standard deviation 
smoothing = 1e-10
# USe the mean and standard deviation of the training data
feature_mean = train_data[:,:num_columns].mean(axis=0, keepdims=True)
feature_std = train_data[:,:num_columns].std(axis=0, keepdims=True)
# Normalize all numeric features except wilderness type and soil type (one-hot) - first 10 columns
train_data[:,:num_columns] = train_data[:,:num_columns] - feature_mean
train_data[:,:num_columns] = np.divide(train_data[:,:num_columns], feature_std + smoothing)
# Normalize dev data as well (using training mean and standard deviation)
dev_data[:,:num_columns] = dev_data[:,:num_columns] - feature_mean
dev_data[:,:num_columns] = dev_data[:,:num_columns]/(feature_std + smoothing)
```


```python
# Explore and confirm the shape of the data
print("Training data shape: {0} Training labels shape: {1}".format(train_data.shape, train_labels.shape))
print("Dev data shape: {0} Dev labels shape: {1}".format(train_data.shape, train_labels.shape))
print("Test data shape: ", test_data.shape)
print("First training example: ", train_data[0], train_labels [0])
print("First dev example: ", dev_data[0,:], dev_labels [0])
print("First test example: ", test_data[0])
```

    Training data shape: (12096, 54) Training labels shape: (12096,)
    Dev data shape: (12096, 54) Dev labels shape: (12096,)
    Test data shape:  (30000, 54)
    First training example:  [-1.15073206  1.76309729  2.55351957 -0.79227979  0.02151651 -0.38583463
     -2.80073415 -3.09550296  0.14965231 -0.99172149  0.          0.
      0.          1.          0.          0.          0.          0.
      0.          0.          0.          0.          0.          1.
      0.          0.          0.          0.          0.          0.
      0.          0.          0.          0.          0.          0.
      0.          0.          0.          0.          0.          0.
      0.          0.          0.          0.          0.          0.
      0.          0.          0.          0.          0.          0.        ] 3
    First dev example:  [-0.74688418  1.23486517  0.41571389 -0.79227979 -0.48780573 -1.0186886
     -1.75572141  0.57058401  1.65349798 -0.54355007  0.          0.
      0.          1.          0.          0.          0.          0.
      0.          0.          0.          0.          0.          1.
      0.          0.          0.          0.          0.          0.
      0.          0.          0.          0.          0.          0.
      0.          0.          0.          0.          0.          0.
      0.          0.          0.          0.          0.          0.
      0.          0.          0.          0.          0.          0.        ] 3
    First test example:  [2680  354   14    0    0 2684  196  214  156 6645    1    0    0    0
        0    0    0    0    0    0    0    0    0    0    0    0    0    0
        0    0    0    0    0    0    0    0    0    0    0    0    0    0
        1    0    0    0    0    0    0    0    0    0    0    0]



```python
# Try a random forest - before any data cleaning 
def RandomForest(num_trees):
    model = RandomForestClassifier(num_trees,max_depth=8)
    model.fit(train_data, train_labels)
    predictions = model.predict(dev_data)
    score = model.score(dev_data, dev_labels)
    print("Random Forest Performance for {0} trees: {1}".format(num_trees,score))
    print("Random Forest Confusion Matrix:\n")
    print(confusion_matrix(predictions, dev_labels))
    
    #plot_confusion_matrix
    plot_confusion_matrix(model, dev_data, dev_labels, values_format = "d")
    plt.title("{} Tree Random Forest Confusion Matrix:".format(num_trees))
    plt.plot()
    print('\n\n')
    
num_trees_list = [1,3,5,10,100]
for num_trees in num_trees_list:
    RandomForest(num_trees)
```

    Random Forest Performance for 1 trees: 0.576058201058201
    Random Forest Confusion Matrix:
    
    [[331 154   0   0   7   0 216]
     [ 43 137   0   0  64   1   0]
     [  0   1 155  70   0  90   0]
     [  0   0  35 334   0  29   0]
     [ 29 114 156   0 360 118   0]
     [  1  17  63   8  13 214   0]
     [ 38  15   0   0   0   0 211]]
    
    
    
    Random Forest Performance for 3 trees: 0.667989417989418
    Random Forest Confusion Matrix:
    
    [[256 123   0   0  25   0  32]
     [106 180   4   0  16   1   8]
     [  0  13 201   9  10  86   0]
     [  0   0  59 394   0  43   0]
     [ 27  97  60   0 380  98   2]
     [  1  14  83   9  12 224   0]
     [ 52  11   2   0   1   0 385]]
    
    
    
    Random Forest Performance for 5 trees: 0.7116402116402116
    Random Forest Confusion Matrix:
    
    [[293 143   1   0  11   0  32]
     [ 57 163   4   0  34   2   3]
     [  0  10 270  20  17 109   0]
     [  0   0  27 388   0  29   0]
     [ 35  96  11   0 369  30   5]
     [  0  14  95   4  13 282   0]
     [ 57  12   1   0   0   0 387]]
    
    
    
    Random Forest Performance for 10 trees: 0.7341269841269841
    Random Forest Confusion Matrix:
    
    [[278 104   0   0   5   0  22]
     [ 62 204   2   0  28   3   2]
     [  0  10 284  17  13 112   0]
     [  0   0  34 383   0  22   0]
     [ 26  91  23   0 387  34   0]
     [  1  14  66  12  11 281   0]
     [ 75  15   0   0   0   0 403]]
    
    
    
    Random Forest Performance for 100 trees: 0.7493386243386243
    Random Forest Confusion Matrix:
    
    [[298 110   0   0   1   0  24]
     [ 53 208   0   0  35   0   0]
     [  0   7 296  12  18 108   0]
     [  0   0  31 394   0  21   0]
     [ 25  87  16   0 382  35   3]
     [  1  14  66   6   8 288   0]
     [ 65  12   0   0   0   0 400]]
    
    
    



![png](backups/clear-cut-solution_files/backups/clear-cut-solution_4_1.png)



![png](backups/clear-cut-solution_files/backups/clear-cut-solution_4_2.png)



![png](backups/clear-cut-solution_files/backups/clear-cut-solution_4_3.png)



![png](backups/clear-cut-solution_files/backups/clear-cut-solution_4_4.png)



![png](backups/clear-cut-solution_files/backups/clear-cut-solution_4_5.png)



```python
# Try Naive Bayes - before any data cleaning 
def NB(alf):
    model = BernoulliNB(alpha = alf)
    model.fit(train_data, train_labels)
    predictions = model.predict(dev_data)
    score = model.score(dev_data, dev_labels)
    print("BernoulliNB for alph = {0}: accuracy = {1}".format(alf,score))
    
    #plot_confusion_matrix
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

    BernoulliNB for alph = 0.01: accuracy = 0.6121031746031746
    
    
    



![png](backups/clear-cut-solution_files/backups/clear-cut-solution_5_1.png)



```python
# Try K Nearest Neighbors - before any data cleaning 
def KNN(kn):
    model = KNeighborsClassifier(n_neighbors = kn)
    model.fit(train_data, train_labels)
    predictions = model.predict(dev_data)
    score = model.score(dev_data, dev_labels)
    print("KNN {0} neighbors : accuracy = {1}".format(kn,score))
    
    #plot_confusion_matrix
    plot_confusion_matrix(model, dev_data, dev_labels, values_format = "d")
    plt.title("KNN Confusion Matrix with {} Neighbors".format(kn))
    plt.plot()
    print('\n\n')
    
# the alpha isn't actually making a difference 
neigh_list = [1,2,4, 7, 10]
# neigh_list = [5]
for neigh in neigh_list:
    KNN(neigh)
```

    KNN 1 neighbors : accuracy = 0.8118386243386243
    
    
    
    KNN 2 neighbors : accuracy = 0.7797619047619048
    
    
    
    KNN 4 neighbors : accuracy = 0.7876984126984127
    
    
    
    KNN 7 neighbors : accuracy = 0.7757936507936508
    
    
    
    KNN 10 neighbors : accuracy = 0.7622354497354498
    
    
    



![png](backups/clear-cut-solution_files/backups/clear-cut-solution_6_1.png)



![png](backups/clear-cut-solution_files/backups/clear-cut-solution_6_2.png)



![png](backups/clear-cut-solution_files/backups/clear-cut-solution_6_3.png)



![png](backups/clear-cut-solution_files/backups/clear-cut-solution_6_4.png)



![png](backups/clear-cut-solution_files/backups/clear-cut-solution_6_5.png)


### End matter

#### Acknowledgements/Sources

* That helpful stack overflow post
  * the url for it
* Relevant Documentation
  * KNeighborsClassifier
    * https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
  * Pretty Confusion Matrix
    * https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_confusion_matrix.html
* Soil information
  * https://www.uidaho.edu/cals/soil-orders/aridisols
  
#### Backup Formats

*because sometimes you just want to look at the markdown real quick*


```python
#Create a backup of the jupyter notebook in a format for where changes are easier to see.
!jupyter nbconvert clear-cut-solution.ipynb --to="python" --output="backups/clear-cut-solution"
!jupyter nbconvert clear-cut-solution.ipynb --to markdown --output="backups/clear-cut-solution"

# Also archiving this bad boy
!jupyter nbconvert clear-cut-solution.ipynb --to html --output="backups/clear-cut-solution"
```
