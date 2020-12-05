# Libraries for reading, cleaning and plotting the data.
import numpy as np
import pandas as pd
import re

# Count the occurences of words.
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


def scale_hillside(data):
    """
    This function scales the hillside features between 0 and 1.

    Parameters
    ----------
    data: pd.DataFrame
        The training data of shape (Training Examples, Features)

    Returns
    -------
    data: pd.DataFrame
        The training data of shape (Training Examples, Features)
    """
    fe1_cols = ['Hillshade_9am', 'Hillshade_Noon',
                'Hillshade_3pm']
    data[fe1_cols] = data[fe1_cols] / 255

    return data



def transform_aspect(data):
    """
    This function transforms the aspect feature (radians) to degrees and transforms it into a unit vector.

    Parameters
    ----------
    data: pd.DataFrame
        The training data of shape (Training Examples, Features)

    Returns
    -------
    data: pd.DataFrame
        The training data of shape (Training Examples, Features)
    """
    # Convert aspect into sine and cosine values
    data["ap_ew"] = np.sin(data["Aspect"] / 180 * np.pi)
    data["ap_ns"] = np.cos(data["Aspect"] / 180 * np.pi)

    # Drop Aspect column
    data.drop(columns=["Aspect"], inplace=True)

    return data


def log_features(data, features):
    """
    This function performs log changes.

    Parameters
    ----------
    data: pd.DataFrame
        The training data of shape (Training Examples, Features)
    features: list of type string
        Names of features

    Returns
    -------
    data: pd.DataFrame
        The training data of shape (Training Examples, Features)
    """
    # Add minimum plus 1 to ensure no negative or 0 in the values
    data[features] = data[features] + 1

    # Log transform
    data[features] = np.log(data[features])

    return data


def add_polynomial_features(data, features):
    """
    This function performs polynomial changes.

    Parameters
    ----------
    data: pd.DataFrame
        The training data of shape (Training Examples, Features)
    features: list of type string
        Names of features

    Returns
    -------
    data: pd.DataFrame
        The training data of shape (Training Examples, Features)
    """
    # Add a polynominal feature
    for feature in features:
        new_name = feature + "_squared"
        data[new_name] = data[feature]**2

    return data


def drop_features(data, features):
    """
    This function drops features.

    Parameters
    ----------
    data: pd.DataFrame
        The training data of shape (Training Examples, Features)
    features: list of type string
        Names of features

    Returns
    -------
    data: pd.DataFrame
        The training data of shape (Training Examples, Features)
    """
    data.drop(columns=features, inplace=True)

    return data


def split_data(data, train_indicies):
    """
    This function splits data into train and dev sets.

    Parameters
    ----------
    data: pd.DataFrame
        The training data of shape (Training Examples, Features)

    Returns
    -------
    data: pd.DataFrame
        The training data of shape (Training Examples, Features)
    """
    # Split training data (labeled) into 80% training and 20% dev) and
    # randomly sample
    train_data_df = data.loc[train_indicies]
    dev_data_df = data.drop(train_indicies)
    
    # Split into data and labels
    train_data = train_data_df.drop(columns=["Cover_Type"])
    train_labels = train_data_df["Cover_Type"]
    dev_data = dev_data_df.drop(columns=["Cover_Type"])
    dev_labels = dev_data_df["Cover_Type"]

    return train_data, train_labels, dev_data, dev_labels


def scale_training_data(cols, train_data, scaler_type="standard"):
    """
    This function standardizes features by removing the mean and scaling to unit variance.

    Parameters
    ----------
    cols: list
        Number of trees to create in the forest
    data: pd.DataFrame
        The training data of shape (Training Examples, Features)
    scaler: sklearn Scalar
        Scalar object fitted to the training data

    Returns
    -------
    data: pd.DataFrame
        The training data of shape (Training Examples, Features)
    data_scaler: sklearn Scaler
        Scalar object fitted to the training data
    """
    if scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    data_scaler = scaler.fit(train_data[cols])
    train_data[cols] = data_scaler.transform(train_data[cols])

    return train_data, data_scaler


def scale_non_training_data(cols, data, scaler):
    """
    This function standardizes features using a scaler fitted by the training data.

    Parameters
    ----------
    cols: list
        Number of trees to create in the forest
    data: pd.DataFrame
        The training data of shape (Training Examples, Features)
    scaler: sklearn Scalar
        Scalar object fitted to the training data

    Returns
    -------
    data: pd.DataFrame
        The training data of shape (Training Examples, Features)
    """
    # Normalize features using the standard scaler [dev data]
    data[cols] = scaler.transform(data[cols])

    return data


def subset_data(data):
    """
    This function groups cover type 1 and 2 data, cover type 3 and 6 data 
    into 2 spearate datasets and change the label to "12" and "36" in the main data set.

    Parameters
    ----------
    data: pd.DataFrame
        The training data of shape (Training Examples, Features)

    Returns
    -------
    data: pd.DataFrame
        The main dataset, and 2 separate subsets
    """
    train_data_12_36_5_7 = data.copy()
    train_data_12_36_5_7["New_Cover"] = np.where(train_data_12_36_5_7["Cover_Type"].apply(lambda x: x in [1,2]),
                                                 12,
                                                 np.where(train_data_12_36_5_7["Cover_Type"].apply(lambda x: x in [3,6]),
                                                          36,
                                                          train_data_12_36_5_7["Cover_Type"]))
    train_data_12 = train_data_12_36_5_7[train_data_12_36_5_7["New_Cover"] == 12]
    train_data_36 = train_data_12_36_5_7[train_data_12_36_5_7["New_Cover"] == 36]
    train_data_12_36_5_7.drop(columns=["Cover_Type"],inplace=True)
    train_data_12_36_5_7.rename(columns={"New_Cover": "Cover_Type"}, inplace=True)
    train_data_12.drop(columns=["New_Cover"],inplace=True)
    train_data_36.drop(columns=["New_Cover"],inplace=True)
    
    return data , train_data_12_36_5_7, train_data_12, train_data_36


def manipulate_data(data):
    """ 
    This function applys transformations on the input data set
    
    Parameters: 
        data (dataframe): n_examples x m_features (int64) dataframe 
    """
    
    data = scale_hillside(data)
    
    data = drop_unseen_soil_types(data)  
    data = combine_soil_types(data)
    
    data = transform_aspect(data)
    
    features_to_log = ['Horizontal_Distance_To_Hydrology',
           'Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points']
    data = log_features(data, features_to_log)
    
    features_to_square = ["Elevation"]
    data = add_polynomial_features(data, features_to_square)

    # These are already being dropped by now? 
    features_to_drop = ["Id","Hillshade_9am","Vertical_Distance_To_Hydrology"]
    drop_features(data, features_to_drop)
    
    return data


def drop_unseen_soil_types(data):
    """
    This function drops soils that are either not seen in the dataset at all or are very rare.

    Parameters
    ----------
    data: pd.DataFrame
        The training data of shape (Training Examples, Features)

    Returns
    -------
    data: pd.DataFrame
        The training data of shape (Training Examples, Features)
    """
    # Remove soil type 7 and 15 due to no data
    data.drop(columns=["Soil_Type7", "Soil_Type15"], inplace=True)
    # Remove soil type 19, 37, 21, 27,36 due to low frequency in training data -
    # TODO: should we be dropping these?
    data.drop(
        columns=[
            "Soil_Type19",
            "Soil_Type37",
            "Soil_Type21",
            "Soil_Type27",
            ],
        inplace=True)
    return data


def combine_environment_features_ct12(data):
    """
    This function creates new features by combining features that are common to specific environments (and canopy types).

    Parameters
    ----------
    data: pd.DataFrame
        The training data of shape (Training Examples, Features)

    Returns
    -------
    data: pd.DataFrame
        The training data of shape (Training Examples, Features)
    """
    # Create additional features to magnify the differences between cover type 1 and 2
    # Combine soil type 2,18,25,3,36,6,8,28,34 and wildness area 4 as only
    # cover type 2 appers under these features
    data["type2stwa4"] = data["Soil_Type6"] + data["Wilderness_Area4"] +  \
        data["Soil_Type2"] + data["Soil_Type18"] + data["Soil_Type25"] +  \
        data["Soil_Type3"] + data["Soil_Type36"] + \
        data["Soil_Type8"] + data["Soil_Type34"] + data["Soil_Type28"]

    # Combine soil type 12 and 9 as they appear a lot more under cover type 2
    # but fewer under cover type 1, none under cover type 5
    data["st_912"] = data["Soil_Type12"] + data["Soil_Type9"]

    return data

def combine_environment_features_ct36(data):
    """
    This function creates new features by combining features that are common to specific environments (and canopy types).

    Parameters
    ----------
    data: pd.DataFrame
        The training data of shape (Training Examples, Features)

    Returns
    -------
    data: pd.DataFrame
        The training data of shape (Training Examples, Features)
    """
    # Combine soil type 20, 23, 24, 31, 33 and 34 as only cover type 6 appears
    # under these features but not cover type 3.
    data["type6st"] = data["Soil_Type20"] + data["Soil_Type23"] + data["Soil_Type24"] + \
        data["Soil_Type31"] + data["Soil_Type33"] + data["Soil_Type34"]

    return data


def combine_soil_types(data):
    """
    This function combines soil types with the same distribution.

    Parameters
    ----------
    data: pd.DataFrame
        The training data of shape (Training Examples, Features)

    Returns
    -------
    data: pd.DataFrame
        The training data of shape (Training Examples, Features)
    """
    # Combine soil type 35,38,39, 40
    data["soil_type35383940"] = data["Soil_Type38"] + \
        data["Soil_Type39"] + data["Soil_Type40"] + data["Soil_Type35"]
    # Combine soil type 10,11, 16, 17
    data["st10111617"] = data["Soil_Type10"] + data["Soil_Type11"] + \
        data["Soil_Type16"] + data["Soil_Type17"]
    # Combine soil type 9, 12
    data["st912"] = data["Soil_Type9"] + data["Soil_Type12"]
    # Combine soil type 31,33
    data["st3133"] = data["Soil_Type31"] + data["Soil_Type33"]
    # Combine soil type 23, 24
    data["st2324"] = data["Soil_Type23"] + data["Soil_Type24"]
    # Combine soil type 6 and wilderness area 4
    data["st6w4"] = data["Soil_Type6"] + data["Wilderness_Area4"]

#     data.drop(columns=["Soil_Type35","Soil_Type38", "Soil_Type39",'Soil_Type40','Soil_Type10','Soil_Type11','Soil_Type16','Soil_Type17','Soil_Type9','Soil_Type12','Soil_Type31','Soil_Type33','Soil_Type23','Soil_Type24','Soil_Type6','Wilderness_Area4'], inplace=True)

    return data





"""
The functions below are no longer used, but contributed to the research and exploration in the project
"""


def preprocess_soil_type(s):
    """
    This function drops the Soil Type features that are not present in the training dataset.

    Parameters
    ----------
        s: string
            Full soil names and descriptions
    Returns
    -------
        s: string
            Regex manipulated string
    """
    # Lowercase everything to make it easier to work with.
    s = s.lower()

    # Take out punctuation and numbers.
    pattern = re.compile(r"[\d\.,-]")
    s = pattern.sub(" ", s)

    # Take out filler words "family","families","complex".
    pattern = re.compile(r"(family|families|complex|typic)")
    s = pattern.sub(" ", s)

    # Replace cryaquolis/aquolis (doesn't exist) with cryaquolls/aquolls.
    pattern = re.compile(r"aquolis")
    s = pattern.sub("aquolls", s)

    # The "unspecified" row doesn't contain any data.
    pattern = re.compile(r"unspecified in the usfs soil and elu survey")
    s = pattern.sub(" ", s)

    # Replace the space in words separated by a single space with an
    # underscore.
    pattern = re.compile(r"(\w+) (\w+)")
    s = pattern.sub(r"\1_\2", s)

    return s


def set_soil_type_by_attributes(data):
    """
    This function parses the soil type descriptions to create new features.
    This will account for the overlap in soil types.
    Parameters
    ----------
    data: pd.DataFrame
        The training data of shape (Training Examples, Features)
    Returns
    -------
    data: pd.DataFrame
        The training data of shape (Training Examples, Features)
    """
    # Pull in the text of the soil types
    with open("km_EDA/soil_raw.txt") as f:
        s_raw = f.read()

    s = preprocess_soil_type(s_raw)
    
    ### COUNT AND TRANSFORM THE DATA ###
    cv = CountVectorizer()
    # Create the counts matrix based on word occurences in our processed soil
    # types
    counts = cv.fit_transform(s.split("\n"))
    # We can use the counts as a transformation matrix to convert to our
    # refined categories
    xform = counts.toarray()

    # Explanation of xform
    # It turns out that multiplying our original soil matrix by xform using matrix mutiplication
    # will just give us a matrix that has been converted to the new feature
    # space.

    # Grab out the new features (that are replacing s_01 thru s_40)
    new_cats = cv.get_feature_names()

    # Get original soil names
    og_soil_col_names = [("Soil_Type{:d}".format(ii + 1)) for ii in range(40)]

    # Get columns containing soil information from our dataframe.
    soil_cols = np.array(data[og_soil_col_names])

    # Transform the soil features. Put them into a dataframe.
    trans_soil = np.matmul(soil_cols, xform)
    trans_soil_df = pd.DataFrame(data=trans_soil, columns=new_cats)

    # Remove the features that have very low occurence rates

    # Remove low occurence soil types
    occ_lim = 1400  # 0, 2,100,300,900,1400 default # remove columns that have less than occ_lim examples in the data
    high_occ_ser = trans_soil_df.sum(axis=0) >= occ_lim
    high_occ_names = [
        entry for entry in high_occ_ser.index if high_occ_ser[entry]]
    trans_soil_df = trans_soil_df[high_occ_names]

    # Combine the new soil features with the existing freatures in a single df
    data = data.drop(columns=og_soil_col_names)
    data = pd.concat([data, trans_soil_df], axis=1)
    
    return data
