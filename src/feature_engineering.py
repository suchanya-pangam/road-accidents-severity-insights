# --- Original notebook code cell 6 ---
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns

print("Numerical variables:")
print(numerical_cols)

print("\nCategorical variables:")
print(categorical_cols)

# --- Original notebook code cell 15 ---
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for column in df.columns:
    if df[column].dtype == 'object' and column != 'Accident_severity':
        df[column] = le.fit_transform(df[column])

df['Accident_severity'] = le.fit_transform(df['Accident_severity'])
# encoding จากข้อความเป็นตัวเลข

# --- Original notebook code cell 17 ---
df['Accident_severity'].value_counts()

# --- Original notebook code cell 24 ---
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# สมมติว่า X = features, y = target
# และ train model เรียบร้อยแล้ว

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# ดึงค่า importance
importances = model.feature_importances_

# สร้าง DataFrame
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
})

# เรียงจากมากไปน้อย
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# แสดง 5 อันดับแรก
top5 = feature_importance_df.head(5)

print(top5)
