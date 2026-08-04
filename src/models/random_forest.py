# --- Original notebook code cell 23 ---
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
    model = RandomForestClassifier() # Or any other classifier
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

# --- Original notebook code cell 41 ---
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

X = df[['Area_accident_occured','Day_of_week','Lanes_or_Medians','Road_surface_conditions','Age_band_of_driver','Light_conditions','Type_of_vehicle','Number_of_casualties','Cause_of_accident','Number_of_vehicles_involved','Age_band_of_casualty','Driving_experience','Type_of_collision']]
y = df['Accident_severity']

for fold, (train_index, test_index) in enumerate(kf.split(X, y)):
    X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
    y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

    # Initialize Model
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train_fold, y_train_fold)

    # Predict
    y_pred = rf_model.predict(X_test_fold)

    # Collect Metrics
    accuracy_scores.append(accuracy_score(y_test_fold, y_pred))
    precision_scores.append(precision_score(y_test_fold, y_pred, average='weighted', zero_division=0))
    recall_scores.append(recall_score(y_test_fold, y_pred, average='weighted', zero_division=0))
    f1_scores.append(f1_score(y_test_fold, y_pred, average='weighted', zero_division=0))

    print(f"Fold {fold + 1} completed")

print("\n--- Random Forest Average Results ---")
print(f"Avg Accuracy: {np.mean(accuracy_scores):.4f}")
print(f"Avg F1-score: {np.mean(f1_scores):.4f}")

# --- Original notebook code cell 50 ---
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
    model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,              # ลดความลึกเพื่อไม่ให้มันจำกลุ่ม Slight มากเกินไป
    class_weight='balanced_subsample', # บังคับให้โมเดลให้น้ำหนักเคสหนักสูงๆ
    random_state=42)
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

# --- Original notebook code cell: Random Forest feature-result experiment ---
# prompt: ใช้rf รัน df
from sklearn.ensemble import RandomForestClassifier  # Import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, accuracy_score, precision_score, recall_score, f1_score # Import necessary metrics
from sklearn.metrics import classification_report

# Random Forest
randomforest = RandomForestClassifier()

# fit the model with data
randomforest.fit(X_train, y_train)

y_pred_rf = randomforest.predict(X_test)  # ผลลัพธ์เป็น int อัตโนมัติ

# Calculate and print the evaluation metrics
accuracy = accuracy_score(y_test, y_pred_rf)
precision = precision_score(y_test, y_pred_rf, average='weighted')  # Use 'weighted' for multi-class
recall = recall_score(y_test, y_pred_rf, average='weighted')  # Use 'weighted' for multi-class
f1 = f1_score(y_test, y_pred_rf, average='weighted')  # Use 'weighted' for multi-class

print('classification report:',classification_report(y_test,y_pred_rf))
print(f"randomforest Accuracy: {accuracy:.4f}")
print(f"randomforest Precision: {precision:.4f}")
print(f"randomforest Recall: {recall:.4f}")
print(f"randomforest F1-score: {f1:.4f}")
