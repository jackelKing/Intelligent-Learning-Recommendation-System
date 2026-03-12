import pandas as pd
from src.recommender.recommendation_engine import generate_recommendations


def main():

    student_id = 11391

    # Load interaction data
    student_vle = pd.read_csv("data/raw/studentVle.csv")

    # Use all interactions for demo recommendation
    student_data = student_vle[
        student_vle["id_student"] == student_id
    ]

    if len(student_data) == 0:
        print("Student not found.")
        return

    recs = generate_recommendations(student_id, student_data)

    print("Recommended learning resources:")
    print(recs)


if __name__ == "__main__":
    main()