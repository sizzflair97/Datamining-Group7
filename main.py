import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

plt.style.use('fivethirtyeight')

import warnings
warnings.filterwarnings('ignore')

diamond_df = pd.read_csv('diamonds.csv')
diamond_df.head()

diamond_df.shape