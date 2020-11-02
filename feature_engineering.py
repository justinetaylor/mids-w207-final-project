# Libraries for reading, cleaning and plotting the data.
import numpy as np 
import pandas as pd 
import re

# Count the occurences of words.
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


# Todo: I don't think we need to do this because we scale eveything (feature wise) later. 
def scale_hillside(data):
    """ 
    Scales the hillside features between 0 and 1.

    Parameters: 
        data (dataframe): n_examples x m_features (int64) dataframe 
    """
    fe1_cols = ['Hillshade_9am', 'Hillshade_Noon',
           'Hillshade_3pm']
    data[fe1_cols] = data[fe1_cols]/255
    
    return data


def combine_environment_features(data):
    """ 
    This function creates new features by combining features that are common to specific environments (and canopy types).

    Parameters: 
        data (dataframe): n_examples x m_features (int64) dataframe 
    """
    # Create additional features to magnify the differences between cover type 1 and 2
    # Combine soil type 2,18,25,3,36,6,8,28,34 and wildness area 4 as only cover type 2 appers under these features
    data["type2stwa4"] = data["Soil_Type6"] + data["Wilderness_Area4"] +  \
        data["Soil_Type2"]+ data["Soil_Type18"] +  data["Soil_Type25"] +  \
        data["Soil_Type3"] + data["Soil_Type36"]+ \
        data["Soil_Type8"] + data["Soil_Type34"]+ data["Soil_Type28"]

    # Combine soil type 12 and 9 as they appear a lot more under cover type 2 but fewer under cover type 1, none under cover type 5 
    data["st_912"] = data["Soil_Type12"] + data["Soil_Type9"] 

    # Combine soil type 20, 23, 24, 31, 33 and 34 as only cover type 6 appears under these features but not cover type 3.
    data["type6st"] = data["Soil_Type20"] + data["Soil_Type23"]+ \
    data["Soil_Type24"] +  data["Soil_Type31"] + data["Soil_Type33"] +  data["Soil_Type34"]
    
    return data


def drop_unseen_soil_types(data):
    # Remove soil type 7 and 15 due to no data
    data.drop(columns=["Soil_Type7", "Soil_Type15"], inplace=True)
    # Remove soil type 19, 37, 34, 21, 27,36,28,8,25 due to no limited data - TODO: should we be dropping these? 
    data.drop(columns=["Soil_Type19", "Soil_Type37","Soil_Type34", "Soil_Type21","Soil_Type27", "Soil_Type36","Soil_Type28","Soil_Type8", "Soil_Type25"], inplace=True)
    return data


def combine_soil_types(data):
    """ 
    This function combines soil types with the same distribution.

    Parameters: 
        data (dataframe): n_examples x m_features (int64) dataframe 
    """
    # Combine soil type 35,38,39, 40
    data["soil_type35383940"] = data["Soil_Type38"] +  data["Soil_Type39"] + data["Soil_Type40"] +  data["Soil_Type35"]
    # Combine soil type 10,11, 16, 17
    data["st10111617"] = data["Soil_Type10"] + data["Soil_Type11"] + data["Soil_Type16"] + data["Soil_Type17"]
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


def preprocess_soil_type(s):
    """ 
    This function drops the Soil Type features that are not present in the training dataset.

    Parameters: 
        s (string): full soil name and description 
    """
    # Lowercase everything to make it easier to work with.
    s = s.lower()

    # Take out punctuation and numbers.
    pattern = re.compile(r"[\d\.,-]")
    s = pattern.sub(" ",s)

    # Take out filler words "family","families","complex".
    pattern = re.compile(r"(family|families|complex|typic)")
    s = pattern.sub(" ",s)

    # Replace cryaquolis/aquolis (doesn't exist) with cryaquolls/aquolls.
    pattern = re.compile(r"aquolis")
    s = pattern.sub("aquolls",s)

    # The "unspecified" row doesn't contain any data.
    pattern = re.compile(r"unspecified in the usfs soil and elu survey")
    s = pattern.sub(" ",s)

    # Replace the space in words separated by a single space with an underscore.
    pattern = re.compile(r"(\w+) (\w+)")
    s = pattern.sub(r"\1_\2",s)
    
    return s
    
    
def set_soil_type_by_attributes(data): 
    """ 
    This function parses the soil type descriptions to create new features.
    This will account for the overlap in soil types.

    Parameters: 
        data (dataframe): n_examples x m_features (int64) dataframe 
    """
     # pull in the text of the soil types
    with open("km_EDA/soil_raw.txt") as f:
        s_raw = f.read()
    
    s = preprocess_soil_type(s_raw)

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
    soil_cols = np.array(df[og_soil_col_names])

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
    data = data.drop(columns=og_soil_col_names)
    data = pd.concat([data, trans_soil_df],axis=1)
    data = data[[col for col in data if col not in ["Cover_Type"]]+["Cover_Type"]] #want cover type as last column
    
    return data


def transform_aspect(data):
    """ 
    This function transforms the aspect feature (radians) to degrees and transforms it into a unit vector.
    
    Parameters: 
        data (dataframe): n_examples x m_features (int64) dataframe 
    """
    # Convert aspect into sine and cosine values 
    data["ap_ew"] = np.sin(data["Aspect"]/180*np.pi)
    data["ap_ns"] = np.cos(data["Aspect"]/180*np.pi)

    # Drop Aspect column
    data.drop(columns= ["Aspect"], inplace=True)
    
    return data



def log_features(data, features):
    """ 
    This function performs log changes. 
    
    Parameters: 
        data (dataframe): n_examples x m_features (int64) dataframe 
        features (list): names of features to apply the transformation to
    """
    # Add minimum plus 1 to ensure no negative or 0 in the values 
    data[features] = data[features] + 1

    # Log transform
    data[features] = np.log(data[features])
    
    return data


def add_polynomial_features(data, features):  
    """ 
    This function performs polynomial changes. 
    
    Parameters: 
        data (dataframe): n_examples x m_features (int64) dataframe 
        features (list): names of features to apply the transformation to
    """
    # Add a polynominal feature
    for feature in features:
        new_name = feature + "_squared"
        data[new_name] = data[feature]**2
        
    return data


def drop_features(data, features):
    """ 
    This function drops features. 
    
    Parameters: 
        data (dataframe): n_examples x m_features (int64) dataframe 
        features (list): names of features to drop
    """
    data.drop(columns=features,inplace=True)
    
    return data


def split_data(data):
    """ 
    This function splits data into train and dev sets
    
    Parameters: 
        data (dataframe): n_examples x m_features (int64) dataframe 
    """
    # Split training data (labeled) into 80% training and 20% dev) and randomly sample 
    train_data_df = data.sample(frac=0.8)
    dev_data_df = data.drop(train_data_df.index)

    # Split into data and labels
    train_data = train_data_df.drop(columns=["Cover_Type"])
    train_labels = train_data_df["Cover_Type"]
    dev_data = dev_data_df.drop(columns=["Cover_Type"])
    dev_labels = dev_data_df["Cover_Type"]

    return train_data, train_labels, dev_data, dev_labels


def scale_training_data(cols, train_data,scaler_type="standard"):
    """ 
    This function standardizes features by removing the mean and scaling to unit variance
    
    Parameters: 
        scaler (string): name of scaler to use (MinMax or Standard)
        cols (list): names of features to scale
        data (dataframe): n_examples x m_features (int64) dataframe 
    """
    if scaler_type == "minmax":
        scaler = MinMaxScaler()
    else :
        scaler = StandardScaler()
    data_scaler = scaler.fit(train_data[cols])
    train_data[cols] = data_scaler.transform(train_data[cols])
    
    return train_data, data_scaler


def scale_non_training_data(cols, data, scaler):
    """ 
    This function standardizes features using a scaler fitted by the training data
    
    Parameters: 
        cols (list): names of features to scale
        data (dataframe): n_examples x m_features (int64) dataframe 
        scaler (Scaler Object): Fitted to training data
    """
    # Normalize features using the standard scaler [dev data]
    data[cols] = scaler.transform(data[cols])
    
    return data



