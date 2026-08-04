# --- Original notebook code cell 25 ---
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. สร้าง Confusion Matrix จากค่าจริงและค่าที่ทำนาย
cm = confusion_matrix(y_test_fold, y_pred)

# 2. กำหนด Labels ให้ตรงกับ Index ของข้อมูล (0=ตาย, 1=สาหัส, 2=เล็กน้อย)
# สำคัญ: ลำดับใน list นี้ต้องตรงกับเลข Class ในตัวแปร y
class_names = ['Fatal (0)', 'Severe Injury (1)', 'Minor Injury (2)']

print("Confusion Matrix (Row=Predicted, Column=Actual):")
print(cm)

# 3. พล็อตกราฟให้ดูง่ายและถูกต้อง
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=class_names,
            yticklabels=class_names)

plt.xlabel('Actual (ค่าจริง)')
plt.ylabel('Predicted (โมเดลทาย)')
plt.title('Confusion Matrix: Accident Severity')
plt.show()
