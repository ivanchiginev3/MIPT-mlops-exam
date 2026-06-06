import json
import os
import shutil
from datetime import datetime


CANDIDATE_MODEL = "models/candidate/churn_model.pkl"
CANDIDATE_METRICS = "models/candidate/metrics.json"

PRODUCTION_MODEL = "models/production/churn_model.pkl"
PRODUCTION_METRICS = "models/production/metrics.json"

ARCHIVE_DIR = "models/archive"


MIN_ROC_AUC = 0.85
MIN_PRECISION = 0.60
MIN_RECALL = 0.60
MIN_F1 = 0.60


def load_metrics(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metrics file not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def candidate_meets_slo(metrics: dict) -> bool:
    checks = {
        "roc_auc": metrics["roc_auc"] >= MIN_ROC_AUC,
        "precision": metrics["precision"] >= MIN_PRECISION,
        "recall": metrics["recall"] >= MIN_RECALL,
        "f1_score": metrics["f1_score"] >= MIN_F1,
    }

    print("Candidate SLO check:")
    for metric, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"{metric}: {metrics[metric]:.4f} -> {status}")

    return all(checks.values())


def candidate_is_better(candidate: dict, production: dict) -> bool:
    print("\nCandidate vs Production comparison:")
    print(f"candidate model: {candidate.get('model_type')}")
    print(f"production model: {production.get('model_type')}")

    print(f"candidate ROC-AUC: {candidate['roc_auc']:.4f}")
    print(f"production ROC-AUC: {production['roc_auc']:.4f}")

    print(f"candidate F1-score: {candidate['f1_score']:.4f}")
    print(f"production F1-score: {production['f1_score']:.4f}")

    # Основная метрика выбора — F1-score,
    # потому что она балансирует Precision и Recall.
    if candidate["f1_score"] > production["f1_score"]:
        print("Candidate is better by F1-score")
        return True

    if (
        candidate["f1_score"] == production["f1_score"]
        and candidate["roc_auc"] > production["roc_auc"]
    ):
        print("Candidate has equal F1-score but better ROC-AUC")
        return True

    print("Candidate is not better than current production model")
    return False


def archive_production_model() -> None:
    if not os.path.exists(PRODUCTION_MODEL):
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = os.path.join(ARCHIVE_DIR, timestamp)

    os.makedirs(archive_path, exist_ok=True)

    shutil.copy2(PRODUCTION_MODEL, os.path.join(archive_path, "churn_model.pkl"))

    if os.path.exists(PRODUCTION_METRICS):
        shutil.copy2(PRODUCTION_METRICS, os.path.join(archive_path, "metrics.json"))

    print(f"\nPrevious production model archived: {archive_path}")


def promote_model() -> None:
    if not os.path.exists(CANDIDATE_MODEL):
        raise FileNotFoundError("Candidate model not found")

    if not os.path.exists(CANDIDATE_METRICS):
        raise FileNotFoundError("Candidate metrics not found")

    candidate_metrics = load_metrics(CANDIDATE_METRICS)

    if not candidate_meets_slo(candidate_metrics):
        print("\nCandidate model rejected: SLO requirements are not satisfied")
        return

    has_production_model = os.path.exists(PRODUCTION_MODEL)
    has_production_metrics = os.path.exists(PRODUCTION_METRICS)

    if has_production_model and has_production_metrics:
        production_metrics = load_metrics(PRODUCTION_METRICS)

        if not candidate_is_better(candidate_metrics, production_metrics):
            print("\nCandidate model rejected: current production model is better or equal")
            return
    else:
        print("\nProduction model or production metrics not found.")
        print("Candidate will be promoted as the first production model.")

    os.makedirs("models/production", exist_ok=True)
    os.makedirs(ARCHIVE_DIR, exist_ok=True)

    archive_production_model()

    shutil.copy2(CANDIDATE_MODEL, PRODUCTION_MODEL)
    shutil.copy2(CANDIDATE_METRICS, PRODUCTION_METRICS)

    print("\nCandidate model promoted to production")
    print(f"Production model updated: {PRODUCTION_MODEL}")
    print(f"Production metrics updated: {PRODUCTION_METRICS}")


if __name__ == "__main__":
    promote_model()