import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Set random seed for reproducibility
np.random.seed(42)

# Function to generate random data for each column
def generate_data(n):
    respiration_per_minute = np.random.randint(10, 35, size=n)
    age = np.random.randint(18, 80, size=n)
    gender = np.random.choice(['Male', 'Female'], size=n)
    weight = np.random.randint(40, 150, size=n)
    body_mass_index = np.random.randint(15, 40, size=n)
    cough = np.random.choice([0, 1], size=n)
    dyspnea = np.random.choice([0, 1], size=n)
    fever = np.random.choice([0, 1], size=n)
    weight_loss = np.random.choice([0, 1], size=n)
    chest_pain = np.random.choice([0, 1], size=n)
    hemoptysis = np.random.choice([0, 1], size=n)
    exposure_to_tb = np.random.choice([0, 1], size=n)
    diabetes = np.random.choice([0, 1], size=n)
    hiv_status = np.random.choice(['Positive', 'Negative', 'Unknown'], size=n)
    smoking_history = np.random.choice(['Never', 'Former', 'Current'], size=n)
    alcohol_use = np.random.choice([0, 1], size=n)
    tb_diagnosis = np.random.choice(['Risk of Extrapulmonary TB',
    'Risk of Pulmonary TB',
    'High Risk of Chronic/Drug Resistant Tuberculosis',
    'Pulmonary Fungucitis',
    'No Lung Problems'], size=n)
    
    return {
        'Respiration_per_minute': respiration_per_minute,
        'Age': age,
        'Gender': gender,
        'Weight': weight,
        'Body_Mass_Index': body_mass_index,
        'Cough': cough,
        'Dyspnea': dyspnea,
        'Fever': fever,
        'Weight_Loss': weight_loss,
        'Chest_Pain': chest_pain,
        'Hemoptysis': hemoptysis,
        'Exposure_to_TB': exposure_to_tb,
        'Diabetes': diabetes,
        'HIV_Status': hiv_status,
        'Smoking_History': smoking_history,
        'Alcohol_Use': alcohol_use,
        'TB_Diagnosis': tb_diagnosis
    }

# Generate data for approximately 2573 rows
data = generate_data(2573)

# Create DataFrame
df = pd.DataFrame(data)

# Save DataFrame to CSV
df.to_csv('tuberculosis_dataset.csv', index=False)



df = pd.read_csv("tuberculosis_dataset.csv")

df['Gender'] = LabelEncoder().fit_transform(df['Gender'])
print(df["HIV_Status"].unique())
df['HIV_Status'] = LabelEncoder().fit_transform(df['HIV_Status'])
print(df["Smoking_History"].unique())
df['Smoking_History'] = LabelEncoder().fit_transform(df['Smoking_History'])
df['TB_Diagnosis'] = LabelEncoder().fit_transform(df['TB_Diagnosis'])

'''mapping={'Risk of Extrapulmonary TB':0,
    'Risk of Pulmonary TB':1,
    'High Risk of Chronic/Drug Resistant Tuberculosis':2,
    'Pulmonary Fungucitis':3,
    'No Lung Problems':4}

df["TB_Diagnosis"] = df['TB_Diagnosis'].map(mapping)'''



print(df.head(8))