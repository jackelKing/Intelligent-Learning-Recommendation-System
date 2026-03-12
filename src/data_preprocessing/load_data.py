import pandas as pd

def load_datasets(data_path="data/raw/"):
    
    student_info = pd.read_csv(data_path + "studentInfo.csv")
    student_assessment = pd.read_csv(data_path + "studentAssessment.csv")
    student_vle = pd.read_csv(data_path + "studentVle.csv")
    vle = pd.read_csv(data_path + "vle.csv")
    courses = pd.read_csv(data_path + "courses.csv")
    student_registration = pd.read_csv(data_path + "studentRegistration.csv")

    return {
        "student_info": student_info,
        "student_assessment": student_assessment,
        "student_vle": student_vle,
        "vle": vle,
        "courses": courses,
        "student_registration": student_registration
    }