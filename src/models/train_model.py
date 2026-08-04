# --- Original notebook code cell 19 ---
# prompt: K-Fold
from imblearn.over_sampling import RandomOverSampler
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Define the number of folds
n_splits = 10  # You can change this value

# Initialize KFold
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# --- Original notebook code cell 37 ---
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

# Define the number of folds
n_splits = 10  # You can change this value

# Initialize KFold
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# --- Original notebook code cell 46 ---
# prompt: K-Fold
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

# Define the number of folds
n_splits = 10  # You can change this value

# Initialize KFold
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
