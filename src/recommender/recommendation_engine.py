from .hybrid_engine import hybrid_recommend

def generate_recommendations(student_id, student_vle=None, top_k=5):
    return hybrid_recommend(student_id, top_k)