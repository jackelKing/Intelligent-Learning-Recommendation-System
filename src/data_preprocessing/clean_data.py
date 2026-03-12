import pandas as pd
from .load_data import load_datasets

def clean_data():

    data = load_datasets()

    student_info = data["student_info"]
    student_assessment = data["student_assessment"]
    student_vle = data["student_vle"]

    # Merge student info with assessment
    merged = pd.merge(student_info, student_assessment, on="id_student", how="left")

    # Remove missing values
    merged = merged.dropna()

    # Remove duplicates
    merged = merged.drop_duplicates()

    # Save cleaned dataset
    merged.to_csv("data/processed/cleaned_data.csv", index=False)

    print("Cleaned data saved!")

if __name__ == "__main__":
    clean_data()