# --- Original notebook code cell 1 ---
from google.colab import files
uploaded = files.upload()

# --- Original notebook code cell 2 ---
import pandas as pd
import numpy as np

df = pd.read_csv('Dataset.csv')
df


# --- Original notebook code cell 3 ---
df.info()

# --- Original notebook code cell 4 ---
df.isnull().sum() # เช็ค null

# --- Original notebook code cell 5 ---
#ลบคอลัมล์'Defect_of_vehicle', 'Service_year_of_vehicle', 'Fitness_of_casuality', 'Work_of_casuality' เพราะ
# มีnullมากซึ่งถ้าทำการลบmissingโดยไม่ลบออก จะทำให้ข้อมูลหายไปเกือบหมดเหลือประมาณ 2800โดยประมาณ
df.drop(df[['Time', 'Defect_of_vehicle', 'Service_year_of_vehicle', 'Fitness_of_casuality', 'Work_of_casuality']],axis=1,inplace=True)

# --- Original notebook code cell 7 ---
df.isnull().sum()

# --- Original notebook code cell 8 ---
df.dropna(inplace=True)

# --- Original notebook code cell 9 ---
df.info()

# --- Original notebook code cell 10 ---
df.isnull().sum()

# --- Original notebook code cell 11 ---
df

# --- Original notebook code cell 12 ---
from google.colab import drive
drive.mount('/content/drive')

# --- Original notebook code cell 13 ---
file_path = '/content/drive/MyDrive/my_data.xlsx'

# --- Original notebook code cell 14 ---
# save
df.to_excel(file_path, index=False)

print("บันทึกเรียบร้อย!")
