
# coding: utf-8

# In[ ]:

from sklearn import linear_model
import numpy as np
import pandas as pd



# In[78]:

# Load Data
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

#train_data.head()
#test_data.head()

# Store Ids of test data for later in the csv creation to kaggle submition
test_ids = test_data["Id"]


# In[84]:

# Load Targets 
y = train_data["SalePrice"]
# Convert to numpy array
y = np.array(y)

print(y.shape)


# #### remember X is our training features and T our test features

# In[68]:

# Load Selected features
X = np.array([train_data["YearBuilt"], train_data["YrSold"], train_data["LotArea"], train_data["YearBuhfdt"]])
T = np.array([test_data["YearBuilt"], test_data["YrSold"], test_data["LotArea"], test])

# transpose for get:
# Columns as features
X = X.T
T = T.T

print(X.shape)
print(T.shape)


# In[64]:

#Training Features
#             #TamaNo    #Cuartos
# X = np.array([[1000,   2],
#               [2000,   2], 
#               [3000,   3], 
#               [10000,  5]])
# # Training Tagets
# y = np.array([2500, 5800, 7800, 18000])
# Test Data
#T = [[4000, 3], [5000, 5], [6000, 2], [7000, 8]]


# Initialize Regression Object
reg = linear_model.LinearRegression()

# Training
reg.fit(X, y)




# In[83]:

# Predictions

# Predict
pred = reg.predict(T)

# Create a "table" each index name is column name
# Kaggle Format
df_dict = {"SalePrice" : pred,
           "Id" : test_ids }

# Convert to Pandas DataFrame
df = pd.DataFrame(df_dict)

# Show Some data
df.head()

# Save in to csv file
df.to_csv("pred.csv", index=False)

