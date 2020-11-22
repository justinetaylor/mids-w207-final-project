import seaborn as sns
import numpy as np

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

# Libraries for plotting
import matplotlib.pyplot as plt
plt.style.use('seaborn')

def ensemble(forest_predictions_all_data, knn_predictions_all_data, mlp_predictions_all_data):
    """ 
    This function retrieves the results from three models and chooses new labels based on the frequency a class was chosen 
    
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
        labels = [forest_predictions_all_data[i],knn_predictions_all_data[i],mlp_predictions_all_data[i]]
        unique, counts = np.unique(labels, return_counts=True)
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
        # All three models disagree. Choose the label from MLP
        else:
            count += 1
            classification = mlp_predictions_all_data[i]
        # Assign the new prediction
        new_predictions.append(classification) 
        
    print(
        "Models disagreed on {0}/{1} examples.".format(count, forest_predictions_all_data.shape[0]))
    return np.array(new_predictions).astype(int)




"""
TODO: Remove all functions below if we don't end up using 
"""

def neural_network(train_data, train_labels, dev_data, dev_labels):
    """ 
    This function creates and fits a Neural Network and evaluates the model on the dev set. 
    
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
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(54,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])

    # Retrieve predictions
    predictions = model(train_data[:-1].to_numpy())
    # Convert logits to probabilities
    tf.nn.softmax(predictions).numpy()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])
    model.fit(train_data.to_numpy(), train_labels.to_numpy(), epochs=5)
    model.evaluate(dev_data.to_numpy(), dev_labels.to_numpy(), verbose=2)

    
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

def multi_layer_perceptron(train_data, train_labels, dev_data, dev_labels):
    """ 
    This function creates and fits a Multi-Layer Perceptron Classifier and evaluates the model on the dev set. 
    
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
            
    Returns
    -------
    score: float
        Mean Accuracy of the model
    probailities: list of type float
        Probabilities assigned to each class 
    model: Multi-Layer Perceptron Classifier
        Fitted KNeighbors Classifier that can be used for testing 
    """
    # Default activation is 'relu', random state lets us get the same result every time (so we can tune other parameters)
    # max_iter is 200 by default, but more helps. alpha is the regularization
    # parameter. solver is 'adam' by default
    model = MLPClassifier(
        alpha=1e-3,
        hidden_layer_sizes=(
            200,
        ),
        random_state=0,
        max_iter=200)
    model.fit(train_data, train_labels)
    predictions = model.predict(dev_data)
    score = model.score(dev_data, dev_labels)
    probabilities = model.predict_proba(dev_data)
    plot_confusion_matrix(model, dev_data, dev_labels, values_format="d")
    plt.title("MLP Confusion Matrix")
    plt.plot()
    print("MLP accuracy = ", score)
    mse_nn = mean_squared_error(dev_labels, predictions)
    print("Mean Squared Error: ", mse_nn)

    return score, probabilities, model


def random_forest(num_trees, train_data, train_labels, dev_data, dev_labels):
    """
    This function creates and fits a Random Forest model and evaluates the model on the dev set.

    Parameters
    ----------
    num_trees: int
        Number of trees to create in the forest
    train_data: pd.DataFrame
        The training data of shape (Training Examples, Features)
    train_labels: pd.DataFrame
        The training labels of shape (Training Examples, 1)
    dev_data: pd.DataFrame
            The dev labels of shape (Dev Examples, Features)
    dev_labels: pd.DataFrame
            The dev labels of shape (Dev Examples, 1)

    Returns
    -------
    score: float
        Mean Accuracy of the model
    probailities: list of type float
        Probabilities assigned to each class
    model: RandomForestClassifier
        Fitted Random Forest Classifier that can be used for testing
    """
    model = RandomForestClassifier(num_trees, max_depth=8)
    model.fit(train_data, train_labels)
    predictions = model.predict(dev_data)
    score = model.score(dev_data, dev_labels)
    probabilities = model.predict_proba(dev_data)
    print(
        "Random Forest Performance for {0} trees: {1}".format(
            num_trees,
            score))
    # Plot_confusion_matrix
    plot_confusion_matrix(model, dev_data, dev_labels, values_format="d")
    plt.title("{} Tree Random Forest Confusion Matrix:".format(num_trees))
    plt.plot()
    mse_forest = mean_squared_error(dev_labels, predictions)
    print("Mean Squared Error: ", mse_forest)
    return score, probabilities, model


def k_nearest_neighbors(kn, train_data, train_labels, dev_data, dev_labels):
    """
    This function creates and fits a K-Nearest Neighbors model and evaluates the model on the dev set.

    Parameters
    ----------
    kn: int
        Number of neighbors
    train_data: pd.DataFrame
        The training data of shape (Training Examples, Features)
    train_labels: pd.DataFrame
        The training labels of shape (Training Examples, 1)
    dev_data: pd.DataFrame
            The dev labels of shape (Dev Examples, Features)
    dev_labels: pd.DataFrame
            The dev labels of shape (Dev Examples, 1)

    Returns
    -------
    score: float
        Mean Accuracy of the model
    probailities: list of type float
        Probabilities assigned to each class
    model: KNeighbors Classifier
        Fitted KNeighbors Classifier that can be used for testing
    """
    model = KNeighborsClassifier(n_neighbors=kn)
    model.fit(train_data, train_labels)
    predictions = model.predict(dev_data)
    score = model.score(dev_data, dev_labels)
    print("KNN {0} neighbors : accuracy = {1}".format(kn, score))
    probabilities = model.predict_proba(dev_data)
    # Plot Confusion Matrix
    plot_confusion_matrix(model, dev_data, dev_labels, values_format="d")
    plt.title("KNN Confusion Matrix with {} Neighbors".format(kn))
    plt.plot()
    mse_knn = mean_squared_error(dev_labels, predictions)
    print("Mean Squared Error: ", mse_knn)
    return score, probabilities, model