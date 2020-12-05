import seaborn as sns
import numpy as np
import pandas as pd 

# Libraries for models
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import tensorflow as tf

# Libraries for plotting
import matplotlib.pyplot as plt
plt.style.use('seaborn')

def ensemble(forest_predictions_all_data, 
             knn_predictions_all_data, 
             mlp_predictions_all_data,
             xgb_predictions_all_data,
             ada_predictions_all_data):
    """ 
    This function retrieves the results from five models and chooses new labels based on the frequency a class was chosen 
    
    Parameters
    ----------
    mpl_resuls: np array: 
        Predicted labels (1-7) (number of dev_labels ,1)
    knn_results: array: np array: 
        Predicted labels (1-7) (number of dev_labels ,1)
    random_forest_results: array: np array: 
        Predicted labels (1-7) (number of dev_labels ,1)

    Returns
    -------
    new_predictions: list 
        New predictions based on a combination of all three models
    """
    # Determine final predictions
    new_predictions = []
    # Keep track of instances in which the models disagree for insight
    count = 0
    for i in range(len(forest_predictions_all_data)):
        labels = [
            forest_predictions_all_data[i],
            knn_predictions_all_data[i],
            mlp_predictions_all_data[i],
            xgb_predictions_all_data[i],
            ada_predictions_all_data[i]
        ]
        unique, counts = np.unique(labels, return_counts=True)
        zipped = dict(zip(unique, counts))
        # Initialize Classification
        classification = 0
        # If there's only 1 unique value, all models agreed
        if len(unique) == 1:
            classification = unique[0]
       # All five models disagree. Choose the label from xgboost
        elif len(unique) == 5:
            count += 1
            classification = xgb_predictions_all_data[i]
         # Two or more out of five models agreed
        else:
            count += 1
            classification = unique[np.argmax(counts)]
            
        # Assign the new prediction
        new_predictions.append(classification) 
        
    print(
        "Models disagreed on {0}/{1} examples.".format(count, forest_predictions_all_data.shape[0]))
    return np.array(new_predictions).astype(int)


"""
We did not include these functions in the final notebook, but they were vital in our exploration and research for this project 
"""

def pca(train_data,train_labels, dev_data, dev_labels):
    """ 
    This function completes a principle component analysis on the training data in order to apply dimensionality reduction
    Parameters
    ----------
    train_data: np array: 
        Training data
    train_labels: array: np array: 
        Known training labels (1-7) 
    dev_data: array: np array: 
        Development data
    dev_labels: array: np array: 
        Known development labels (1-7) 
    Returns
    -------
    new_predictions: list 
        new_data:
            Transformed training data (based on pca)
        new_dev_data:
            Transformed development data (based on pca)
        pca:
            Principle Component Analysis object
    """
    num_cols = ['Elevation', 'Slope', 'Horizontal_Distance_To_Hydrology',
       'Horizontal_Distance_To_Roadways', 'Hillshade_Noon', 'Hillshade_3pm',
       'Horizontal_Distance_To_Fire_Points','ap_ew', 'ap_ns', 'Elevation_squared']
    pca = PCA(n_components=4) 
    new_x = pca.fit_transform(train_data[num_cols])
    new_dev = pca.transform(dev_data[num_cols])
    pca_df = pd.DataFrame(data=new_x, index=train_data.index,columns=["pca_fe1","pca_fe2","pca_fe3","pca_fe4"])
    pca_dev_df = pd.DataFrame(data=new_dev, index=dev_labels.index,columns=["pca_fe1","pca_fe2","pca_fe3","pca_fe4"])
    new_data = pd.concat((train_data.drop(columns=num_cols),pca_df),axis=1)
    new_dev_data = pd.concat((dev_data.drop(columns=num_cols),pca_dev_df),axis=1)
    y =  train_labels
    zip(new_x,y)
    
    colors = ["red","green","blue","yellow","orange","black","purple"]
    target_names = ['type' +str(i) for i in range(1,8)] # labels
      
    fig, axes = plt.subplots(1,2, figsize=(12,6))
    lw = 2
    for i,color, target_name in zip([1,2,3,4,5,6,7], colors,target_names):
        axes[0].scatter(new_x[y==i,0],new_x[y==i,1], color=color, alpha=.8, lw=lw,label=target_name)
        axes[1].scatter(new_x[y==i,2],new_x[y==i,3], color=color, alpha=.8, lw=lw,label=target_name)
    axes[0].legend(loc='best', shadow=False, scatterpoints=1)
    axes[0].set_title('First 2 dims of 4-Dimension PCA of Forest Cover Type')
    axes[1].set_title('Last 2 dims of 4-Dimension PCA of Forest Cover Type')     
    plt.show()
    
    return new_data, new_dev_data, pca
    
    
def neural_network_tf(train_data, train_labels, num_features, num_output, num_epochs):
    """ 
    Source: https://www.tensorflow.org/tutorials/quickstart/beginner
    This function trains a neural network using tensorflow 
    Parameters
    ----------
    train_data: np array: 
        Training data
    train_labels: array: np array: 
        Known training labels (1-7) 
    num_features:
        Number of features in the training set - required to set the input later
    num_output:
        Number of output labels - required to set the output layer
    Returns
    -------
    model:
        Fitted neural network that can be used for evaluation and testing 
        
    Example:
    neuralnet_tf_original = models.neural_network_tf(train_data_original, 
                                                 train_labels_original-1, 
                                                 num_features=48, 
                                                 num_output=7,
                                                 num_epochs=100) 
    """
    # Construct a model with 4 layers, relu activation, drop-out and batch normalization 
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(num_features,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(num_output),
    ])

    # Each example has a prediction in the form of logits 
    predictions = model(train_data).numpy()

    # Establish a loss function with the logits 
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Subtract one from the labels because the output layer is [0-6] instead of [1-7] like the labels 
    loss_fn(train_labels, predictions).numpy()

    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])
    
    # Adjust model parameters to minimize loss
    # Set epochs to 20 and re-run this line multiple times while experimenting (to get a closer look at whats happening)
    model.fit(train_data, train_labels, epochs=100)

    return model

    
def logistic_regression(train_data, train_labels, dev_data, dev_labels):
    """
    This function creates and fits a Logistic Regression model and evaluates the model on the dev set. 

    Parameters
    ----------
    train_data: pd.DataFrame
        The training data of shape (Training Examples, Features)
    train_labels: pd.DataFrame
            The training labels of shape (Training Examples, 1)
    dev_data: pd.DataFrame
                The dev labels of shape (Dev Examples, Features)
    dev_labels: pd.DataFrame
                The dev labels of shape (Dev Examples, 1)
    """
    model = LogisticRegression(
            random_state=0,
            multi_class='ovr',
            solver='lbfgs',
            max_iter=300)
    model.fit(train_data, train_labels)
    score = model.score(dev_data, dev_labels)
    print("Logistic Regression accuracy = ", score)
