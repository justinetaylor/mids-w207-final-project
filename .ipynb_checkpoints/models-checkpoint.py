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
import seaborn as sns


# Try a random forest - before any data cleaning 
def random_forest(num_trees,train_data, train_labels, dev_data, dev_labels):
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
    return score, probabilities, model
        
    
# Try K Nearest Neighbors - before any data cleaning 
def k_nearest_neighbors(kn,train_data, train_labels, dev_data, dev_labels):
    model = KNeighborsClassifier(n_neighbors=kn)
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
    return score, probabilities, model
    
    
    
# Try Multi-Layer Perceptron - before any data cleaning 
def multi_layer_perceptron(train_data, train_labels, dev_data, dev_labels):
    #    model = MLPClassifier(solver='adam', alpha=1e-6, hidden_layer_sizes=(100, ), random_state=0) .8257
    #    model = MLPClassifier(solver='adam', alpha=1e-3, hidden_layer_sizes=(100, ), random_state=0)  .82969
    #    model = MLPClassifier(solver='adam', alpha=1e-3, hidden_layer_sizes=(200, ), random_state=0) .837
    #    model = MLPClassifier(solver='adam', alpha=1e-3, hidden_layer_sizes=(100, ), random_state=0, activation='tanh') .83068

    # Default activation is 'relu', random state lets us get the same result every time (so we can tune other parameters)
    # max_iter is 200 by default, but more helps. alpha is the regularization parameter. solver is 'adam' by default
    model = MLPClassifier(alpha=1e-3, hidden_layer_sizes=(200,), random_state=0, max_iter=200) 
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
        
    return score, probabilities, model



def logistic_regression(train_data, train_labels, dev_data, dev_labels):
    model = LogisticRegression(random_state=0, multi_class='ovr',solver='lbfgs', max_iter = 300)
    model.fit(train_data, train_labels)
    score = model.score(dev_data,dev_labels)
    print("Logistic Regression accuracy = ",score)


def neural_network(train_data, train_labels, dev_data, dev_labels):
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
    model.evaluate(dev_data.to_numpy(),  dev_labels.to_numpy(), verbose=2)

    
def ensemble(mlp_results,knn_results,random_forest_results, dev_labels):
    # Find max score from each model. best_scores shape: (3,)
    best_scores = [max(mlp_results.keys()), max(knn_results.keys()),max(random_forest_results.keys())]
    # Find maximum probability for each example for each model. prediction_probabilities shape: (3024, 3)
    prediction_probabilities = [np.max(mlp_results[best_scores[0]],axis=1),np.max(knn_results[best_scores[1]],axis=1),np.max(random_forest_results[best_scores[2]],axis=1)]
    prediction_probabilities = np.transpose(np.array(prediction_probabilities))
    # Find highest predicted label. predicted_classes shape: (3024, 3)
    predicted_classes = [np.argmax(mlp_results[best_scores[0]],axis=1)+1,np.argmax(knn_results[best_scores[1]],axis=1)+1,np.argmax(random_forest_results[best_scores[2]]+1,axis=1)]
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
