import pandas as pd

def create_interaction_matrix():

    student_vle = pd.read_csv("data/raw/studentVle.csv")

    # Keep important columns
    interaction = student_vle[["id_student", "id_site", "sum_click"]]

    # Create interaction matrix
    matrix = interaction.pivot_table(
        index="id_student",
        columns="id_site",
        values="sum_click",
        fill_value=0
    )

    matrix.to_csv("data/processed/interaction_matrix.csv")

    print("Interaction matrix created!")

if __name__ == "__main__":
    create_interaction_matrix()