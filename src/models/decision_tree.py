# --- Original notebook code cell 21 ---
# Lists to store evaluation metrics for each fold
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
    model = DecisionTreeClassifier() # Or any other classifier
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

# --- Original notebook code cell 31 ---
from sklearn.tree import DecisionTreeClassifier # Import DecisionTreeClassifier

# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
y_pred_dt = decision_tree.predict(X_test)

# Calculate and print the evaluation metrics
accuracy = accuracy_score(y_test, y_pred_dt)
precision = precision_score(y_test, y_pred_dt, average='weighted')  # Use 'weighted' for multi-class
recall = recall_score(y_test, y_pred_dt, average='weighted')  # Use 'weighted' for multi-class
f1 = f1_score(y_test, y_pred_dt, average='weighted')  # Use 'weighted' for multi-class

print('classification report:',classification_report(y_test,y_pred_dt))
print(f"Decision Tree Accuracy: {accuracy:.4f}")
print(f"Decision Tree Precision: {precision:.4f}")
print(f"Decision Tree Recall: {recall:.4f}")
print(f"Decision Tree F1-score: {f1:.4f}")

# --- Original notebook code cell 39 ---
# Lists to store evaluation metrics for each fold
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

# ใช้ kf.split(X, y) เพื่อให้แบ่งตามสัดส่วนคลาสที่ถูกต้อง
for fold, (train_index, test_index) in enumerate(kf.split(X, y)):
    print(f"Fold {fold + 1}")

    # ดึงข้อมูล Train และ Test ของ Fold นั้นๆ ออกมา
    X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
    y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

    # Initialize และ Train โมเดล (แบบ Plain ไม่ใช้เทคนิค Sampling)
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train_fold, y_train_fold)

    # ทำนายผลบน X_test_fold ของ Fold นั้นๆ (ต้องมีจำนวนแถวเท่ากับ y_test_fold)
    y_pred = model.predict(X_test_fold)

    # ตรวจสอบเบื้องต้น (ถ้ายัง Error อีก ให้เช็ค print สองบรรทัดนี้)
    # print(f"Test labels shape: {y_test_fold.shape}")
    # print(f"Predicted labels shape: {y_pred.shape}")

    # คำนวณ Metrics
    accuracy = accuracy_score(y_test_fold, y_pred)
    precision = precision_score(y_test_fold, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test_fold, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test_fold, y_pred, average='weighted', zero_division=0)

    # เก็บผล
    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)

    print(f"Accuracy: {accuracy:.4f} | F1: {f1:.4f}")
    # ถ้าอยากดู Report ราย Fold ให้ปลดคอมเมนต์ข้างล่าง
    # print(classification_report(y_test_fold, y_pred))

# 4. สรุปผลเฉลี่ย
print("\n" + "="*30)
print("AVERAGE METRICS (BASELINE 10-FOLD)")
print("="*30)
print(f"Average Accuracy: {np.mean(accuracy_scores):.4f}")
print(f"Average Precision: {np.mean(precision_scores):.4f}")
print(f"Average Recall: {np.mean(recall_scores):.4f}")
print(f"Average F1-score: {np.mean(f1_scores):.4f}")

# --- Original notebook code cell 48 ---
# Lists to store evaluation metrics for each fold
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
    model = DecisionTreeClassifier() # Or any other classifier
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
