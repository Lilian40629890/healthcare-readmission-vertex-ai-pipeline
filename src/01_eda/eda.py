"""Exploratory data analysis of the healthcare readmissions training set.

Transcribed from the project's final presentation (EDA section). The original
analysis ran in Google Colab with the CSV mounted from Google Drive; here the
file path is passed as a command-line argument instead.

Usage:
    python eda.py path/to/healthcare_readmissions_dataset_train.csv
"""

import sys

import matplotlib.pyplot as plt
import pandas as pd

DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else "healthcare_readmissions_dataset_train.csv"
TARGET = "Readmission within 30 Days"

df = pd.read_csv(DATA_PATH)

# --- Data quality ---------------------------------------------------------
print(df.shape)
print(df.isnull().sum())

# --- Target variable distribution -----------------------------------------
print(df[TARGET].value_counts())

df[TARGET].value_counts().plot(kind="bar")

plt.title("Distribution of Readmission within 30 Days")
plt.xlabel("Readmission within 30 Days")
plt.ylabel("Count")
plt.show()

# --- Statistical summary --------------------------------------------------
print(df.describe())

# --- Univariate analysis --------------------------------------------------
df["Age"].hist()

plt.title("Distribution of Age")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

df["BMI"].hist()

plt.title("Distribution of BMI")
plt.xlabel("BMI")
plt.ylabel("Frequency")
plt.show()

df["Length of Stay"].hist()

plt.title("Distribution of Length of Stay")
plt.xlabel("Length of Stay")
plt.ylabel("Frequency")
plt.show()

df["Gender"].value_counts().plot(kind="bar")

plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()

df["Ethnicity"].value_counts().plot(kind="bar")

plt.title("Ethnicity Distribution")
plt.xlabel("Ethnicity")
plt.ylabel("Count")
plt.show()

df["Type of Treatment"].value_counts(dropna=False).plot(kind="bar")

plt.title("Type of Treatment Distribution")
plt.xlabel("Type of Treatment")
plt.ylabel("Count")
plt.show()

# --- Bivariate analysis ---------------------------------------------------
df.boxplot(column="Age", by=TARGET)

plt.title("Age by Readmission Status")
plt.suptitle("")
plt.xlabel("Readmission within 30 Days")
plt.ylabel("Age")
plt.show()

df.boxplot(column="BMI", by=TARGET)

plt.title("BMI by Readmission Status")
plt.suptitle("")
plt.xlabel("Readmission within 30 Days")
plt.ylabel("BMI")
plt.show()

df.boxplot(column="Length of Stay", by=TARGET)

plt.title("Length of Stay by Readmission Status")
plt.suptitle("")
plt.xlabel("Readmission within 30 Days")
plt.ylabel("Length of Stay")
plt.show()

pd.crosstab(df["Has Diabetes"], df[TARGET]).plot(kind="bar")

plt.title("Diabetes Status vs Readmission")
plt.xlabel("Has Diabetes")
plt.ylabel("Count")
plt.show()

pd.crosstab(df["Has Hypertension"], df[TARGET]).plot(kind="bar")

plt.title("Hypertension Status vs Readmission")
plt.xlabel("Has Hypertension")
plt.ylabel("Count")
plt.show()

pd.crosstab(df["Type of Treatment"], df[TARGET]).plot(kind="bar")

plt.title("Type of Treatment vs Readmission")
plt.xlabel("Type of Treatment")
plt.ylabel("Count")
plt.show()
