import pandas as pd
import numpy as np
from src.recommender.recommendation_engine import generate_recommendations

import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# Metrics
# -----------------------------

def precision_at_k(recommended, relevant, k=5):

    recommended_k = recommended[:k]

    hits = len(set(recommended_k) & set(relevant))

    return hits / k


def recall_at_k(recommended, relevant, k=5):

    recommended_k = recommended[:k]

    hits = len(set(recommended_k) & set(relevant))

    if len(relevant) == 0:
        return 0

    return hits / len(relevant)


def f1_score(p, r):

    if p + r == 0:
        return 0

    return 2 * (p * r) / (p + r)


# -----------------------------
# Evaluate single student
# -----------------------------

def evaluate_student(student_id, student_vle, k=5):

    student_data = student_vle[
        student_vle["id_student"] == student_id
    ]

    # Skip students with too few interactions
    if len(student_data) < 10:
        return None

    # Split interactions
    train = student_data.sample(frac=0.8, random_state=42)
    test = student_data.drop(train.index)

    # Relevant resources (ground truth)
    relevant = test["id_site"].unique()

    # Generate recommendations using TRAIN data
    recs = generate_recommendations(student_id, train)

    if recs is None:
        return None

    recommended = list(recs)

    p = precision_at_k(recommended, relevant, k)
    r = recall_at_k(recommended, relevant, k)
    f = f1_score(p, r)

    print("Recommended:", recommended)
    print("Relevant:", relevant[:10])
    print("Intersection:", set(recommended) & set(relevant))
    print("-----------------------------------")
    
    return p, r, f


# -----------------------------
# Evaluate multiple students
# -----------------------------

def evaluate_system(num_students=100, k=5):

    print("Loading interaction data...")

    student_vle = pd.read_csv("data/raw/studentVle.csv")

    students = student_vle["id_student"].unique()

    # Random sample of students
    students_sample = np.random.choice(students, num_students, replace=False)

    results = []

    for student in students_sample:

        metrics = evaluate_student(student, student_vle, k)
        print("Evaluating student:", student)

        if metrics is None:
            continue

        p, r, f = metrics

        results.append({
            "student_id": student,
            "precision": p,
            "recall": r,
            "f1_score": f
        })

    results_df = pd.DataFrame(results)

    # -----------------------------
    # Save detailed results
    # -----------------------------

    results_df.to_csv(
        "outputs/evaluation/student_metrics.csv",
        index=False
    )

    # -----------------------------
    # Average metrics
    # -----------------------------

    avg_precision = results_df["precision"].mean()
    avg_recall = results_df["recall"].mean()
    avg_f1 = results_df["f1_score"].mean()

    summary = pd.DataFrame({
        "Metric": ["Precision@5", "Recall@5", "F1 Score"],
        "Value": [avg_precision, avg_recall, avg_f1]
    })

    summary.to_csv(
        "outputs/evaluation/summary_metrics.csv",
        index=False
    )

    print("\nEvaluation completed!")
    print("\nAverage Metrics:")

    print("Precision@5:", avg_precision)
    print("Recall@5:", avg_recall)
    print("F1 Score:", avg_f1)


# -----------------------------
# Run evaluation
# -----------------------------

if __name__ == "__main__":

    evaluate_system(num_students=100, k=5)