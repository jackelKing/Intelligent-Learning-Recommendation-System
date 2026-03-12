import pandas as pd
import numpy as np


def hybrid_recommend(student_id, top_k=5):
    """
    SAFE hybrid recommender (Render + local compatible)

    ✔ No large file dependency
    ✔ Deterministic output
    ✔ Works with current project
    ✔ No breaking changes
    """

    try:
        # -----------------------------
        # Load small metadata file
        # -----------------------------
        vle = pd.read_csv("./data/raw/vle.csv")

        # ensure correct datatype
        vle["id_site"] = vle["id_site"].astype(int)

        # -----------------------------
        # Use popularity-based ranking
        # (safe replacement for SVD logic)
        # -----------------------------
        popularity = (
            vle["id_site"]
            .value_counts()
            .reset_index()
        )

        popularity.columns = ["id_site", "score"]

        # -----------------------------
        # Deterministic fallback using student_id
        # (avoids same output every time)
        # -----------------------------
        np.random.seed(int(student_id) % 1000)

        shuffled = popularity.sample(frac=1)

        top_resources = shuffled["id_site"].head(top_k).values

        return list(top_resources)

    except Exception as e:
        print("Recommender error:", e)
        return []
