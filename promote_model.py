import os
import shutil
from datetime import datetime


CANDIDATE_MODEL = "models/candidate/churn_model.pkl"
PRODUCTION_MODEL = "models/production/churn_model.pkl"
ARCHIVE_DIR = "models/archive"


def promote_model():
    if not os.path.exists(CANDIDATE_MODEL):
        raise FileNotFoundError("Candidate model not found")

    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(PRODUCTION_MODEL), exist_ok=True)

    if os.path.exists(PRODUCTION_MODEL):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archived_model = f"{ARCHIVE_DIR}/churn_model_{timestamp}.pkl"
        shutil.copy2(PRODUCTION_MODEL, archived_model)
        print(f"Previous production model archived: {archived_model}")

    shutil.copy2(CANDIDATE_MODEL, PRODUCTION_MODEL)
    print("Candidate model promoted to production")


if __name__ == "__main__":
    promote_model()