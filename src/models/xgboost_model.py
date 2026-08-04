# --- Original notebook code cell 27 ---
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

X = df[['Area_accident_occured','Day_of_week','Lanes_or_Medians','Road_surface_conditions','Age_band_of_driver','Light_conditions','Type_of_vehicle','Number_of_casualties','Cause_of_accident','Number_of_vehicles_involved','Age_band_of_casualty','Driving_experience','Type_of_collision']]
y = df['Accident_severity']


# Loop through each fold
for fold, (train_index, test_index) in enumerate(kf.split(X)):
    print(f"Fold {fold + 1}")

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Apply RandomOverSampler to the training data of the current fold
    ros = RandomOverSampler(random_state=42)
    X_train_fold, y_train_fold = ros.fit_resample(X_train, y_train)


    # Initialize and train your model (e.g., RandomForestClassifier)
    model = xgb.XGBClassifier() # Or any other classifier
    model.fit(X_train_fold, y_train_fold)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Store metrics
    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)

    print('classification report:', classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

# Calculate and print the average metrics across all folds
print("\nAverage Metrics across all folds:")
print(f"Average Accuracy: {np.mean(accuracy_scores):.4f}")
print(f"Average Precision: {np.mean(precision_scores):.4f}")
print(f"Average Recall: {np.mean(recall_scores):.4f}")
print(f"Average F1-score: {np.mean(f1_scores):.4f}")

# --- Original notebook code cell 35 ---
import xgboost as xgb # Import the xgboost library

# XGBoost

# Create an XGBoost classifier
xgb_model = xgb.XGBClassifier(objective="multi:softprob") # Assuming 3 classes for Accident_severity

# Fit the model to the training data
xgb_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate the model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_test, y_pred_xgb)
precision = precision_score(y_test, y_pred_xgb, average='weighted')
recall = recall_score(y_test, y_pred_xgb, average='weighted')
f1 = f1_score(y_test, y_pred_xgb, average='weighted')

print(f"XGBoost Accuracy: {accuracy:.4f}")
print(f"XGBoost Precision: {precision:.4f}")
print(f"XGBoost Recall: {recall:.4f}")
print(f"XGBoost F1-score: {f1:.4f}")

print('classification report:',classification_report(y_test,y_pred_xgb))
print(f"XGBoost Accuracy: {accuracy:.4f}")
print(f"XGBoost Precision: {precision:.4f}")
print(f"XGBoost Recall: {recall:.4f}")
print(f"XGBoost F1-score: {f1:.4f}")


# --- Original notebook code cell 43 ---
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

X = df[['Area_accident_occured','Day_of_week','Lanes_or_Medians','Road_surface_conditions','Age_band_of_driver','Light_conditions','Type_of_vehicle','Number_of_casualties','Cause_of_accident','Number_of_vehicles_involved','Age_band_of_casualty','Driving_experience','Type_of_collision']]
y = df['Accident_severity']


# Loop through each fold
for fold, (train_index, test_index) in enumerate(kf.split(X)):
    print(f"Fold {fold + 1}")

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Apply RandomUnderSampler to the training data of the current fold
    rus = RandomUnderSampler(random_state=42)
    X_train_fold, y_train_fold = rus.fit_resample(X_train, y_train)

    # Initialize and train your model (e.g., RandomForestClassifier)
    model = xgb.XGBClassifier() # Or any other classifier
    model.fit(X_train_fold, y_train_fold)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Store metrics
    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)

    print('classification report:', classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

# Calculate and print the average metrics across all folds
print("\nAverage Metrics across all folds:")
print(f"Average Accuracy: {np.mean(accuracy_scores):.4f}")
print(f"Average Precision: {np.mean(precision_scores):.4f}")
print(f"Average Recall: {np.mean(recall_scores):.4f}")
print(f"Average F1-score: {np.mean(f1_scores):.4f}")

# --- Original notebook code cell 52 ---
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

X = df[['Area_accident_occured','Day_of_week','Lanes_or_Medians','Road_surface_conditions','Age_band_of_driver','Light_conditions','Type_of_vehicle','Number_of_casualties','Cause_of_accident','Number_of_vehicles_involved','Age_band_of_casualty','Driving_experience','Type_of_collision']]
y = df['Accident_severity']


# Loop through each fold
for fold, (train_index, test_index) in enumerate(kf.split(X)):
    print(f"Fold {fold + 1}")

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Apply SMOTE to the training data of the current fold
    smote = SMOTE(random_state=42)
    X_train_fold, y_train_fold = smote.fit_resample(X_train, y_train)

    # Initialize and train your model (e.g., RandomForestClassifier)
    model = xgb.XGBClassifier() # Or any other classifier
    model.fit(X_train_fold, y_train_fold)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Store metrics
    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)

    print('classification report:', classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

# Calculate and print the average metrics across all folds
print("\nAverage Metrics across all folds:")
print(f"Average Accuracy: {np.mean(accuracy_scores):.4f}")
print(f"Average Precision: {np.mean(precision_scores):.4f}")
print(f"Average Recall: {np.mean(recall_scores):.4f}")
print(f"Average F1-score: {np.mean(f1_scores):.4f}")
