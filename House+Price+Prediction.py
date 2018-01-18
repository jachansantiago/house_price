
# coding: utf-8

# In[29]:

from sklearn import linear_model
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


# ## LOAD DATA:

# In[30]:

# Load Data
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# In[35]:

# Store Ids of test data for later in the csv creation to kaggle submition
test_ids = test_data["Id"]


# In[36]:

# Load Targets
y = train_data["SalePrice"]
# Convert to numpy array
y = np.array(y)


# ## Prepare Data for Machine Learning:
#
# **Don't look much the next piece of code** basicly do:
#     - Numerical Data:
#         Change NaN for the median of the column
#     - Categorical Data:
#         Change to Int
#         Hot One encode Categorical Data

# In[33]:


imputer = Imputer(strategy="median")
scaler = StandardScaler()
hot_enc = OneHotEncoder()


def prepare_data(dataframe, feature_list=list(), training=True):

    if not feature_list:
        feature_list = list(dataframe)

    dataframe = dataframe[feature_list]

    # Split dataframe in numerical and categorial data
    num_data = dataframe.select_dtypes(include=[np.number])
    cat_data = dataframe.select_dtypes(include=[object])

    if not num_data.empty:

        # Replace all NaN with the median in numerial data
        if training:
            imputer.fit(num_data)
        X_num = imputer.transform(num_data)

        # Scale between -1, 1
        if training:
            scaler.fit(X_num)

        X_num = scaler.transform(X_num)

        # Check if have categorical data
        if cat_data.empty:
            return X_num

    if not cat_data.empty:

        # Replace all NaN with "None" as other category in categorical data
        cat_data.fillna('None', inplace=True)

        facto_cat_data = pd.DataFrame()

        # Factorize each categorical column (string -> int)
        for feature in list(cat_data):
            facto_cat_data[feature], _ = pd.factorize(cat_data[feature])

        # Hot encode
        if training:
            hot_enc.fit(facto_cat_data.values)
        X_cat_1hot = hot_enc.transform(facto_cat_data.values).todense()

        # Check if have numerical data
        if num_data.empty:
            return X_cat_1hot

    # Merge Numerical Data with One Hot encoded categorical data
    X = np.append(X_num, X_cat_1hot, axis=1)

    return X


# ## Extract our Training Features
#
#  ***TODO: Select Better Features***

# In[40]:

# Load Selected features
# X = np.array([  # TODO: Select better features
#                 train_data["YearBuilt"],
#                 train_data["YrSold"],
#                 train_data["LotArea"]
#             ])

# T = np.array([  # TODO: Select better features
#                 test_data["YearBuilt"],
#                 test_data["YrSold"],
#                 test_data["LotArea"]
#             ])

training_features = ["YearBuilt", "YrSold", "LotArea", "Street"]  # TODO

# Training Data need to be True
X = prepare_data(train_data, training_features, True)

# Test Data need to be False
T = prepare_data(test_data, training_features, False)


# In[41]:

# Training Features
#             #TamaNo    #Cuartos
# X = np.array([[1000,   2],
#               [2000,   2],
#               [3000,   3],
#               [10000,  5]])
# # Training Tagets
# y = np.array([2500, 5800, 7800, 18000])
# Test Data
# T = [[4000, 3], [5000, 5], [6000, 2], [7000, 8]]

# Initialize Regression Object
reg = linear_model.LinearRegression()

# Training
reg.fit(X, y)


# In[ ]:

# Predictions

# Predict
pred = reg.predict(T)


# In[52]:

# For this problem Kaggle do not accept negatives values
# NOTE: We know negatives value are wrong

# TODO Select one

# Option 1: Saturate
if False:
    pred[pred < 0] = 0

# Option 2: Absolute Value
if True:
    pred = np.abs(pred)


# In[53]:

# Create a "table" each index name is column name
# Kaggle Format
df_dict = {"SalePrice": pred,
           "Id": test_ids}

# Convert to Pandas DataFrame
df = pd.DataFrame(df_dict)


# In[54]:

# Save in to csv file
pred_filename = "predictions-with-" + "+".join(training_features) + ".csv"
df.to_csv(pred_filename, index=False)
print("Output file: " + pred_filename)
