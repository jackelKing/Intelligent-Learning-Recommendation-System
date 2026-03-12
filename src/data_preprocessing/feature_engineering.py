import pandas as pd

def create_features():

    df = pd.read_csv("data/processed/cleaned_data.csv")

    # Example features
    df["engagement_score"] = df["studied_credits"] * df["num_of_prev_attempts"]

    # Create knowledge level
    df["knowledge_level"] = df["score"].apply(
        lambda x: "advanced" if x >= 70 else
        ("intermediate" if x >= 40 else "beginner")
    )

    # Encode categorical variables
    df = pd.get_dummies(df, columns=["gender", "highest_education"])

    df.to_csv("data/processed/feature_engineered_data.csv", index=False)

    print("Feature engineered dataset saved!")

if __name__ == "__main__":
    create_features()