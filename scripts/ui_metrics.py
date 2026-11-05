"""Presentation-only helpers. Never modify the underlying model decisions."""


def display_class_score(result: dict) -> tuple[str, float | None]:
    """Return the score for the predicted class, not always the abnormal class.

    A calibrated SVM probability may not agree with the original SVM hard
    decision boundary. This function intentionally does *not* relabel it.
    """
    class_id = int(result["class_id"])
    if class_id not in (0, 1):
        raise ValueError("Unknown binary class")
    label = "비정상 패턴 추정 점수" if class_id == 1 else "정상 패턴 추정 점수"
    p_abnormal = result.get("probability_abnormal")
    if p_abnormal is None:
        return label, None
    p_abnormal = float(p_abnormal)
    if not 0.0 <= p_abnormal <= 1.0:
        raise ValueError("Probability out of range")
    return label, p_abnormal if class_id == 1 else 1.0 - p_abnormal


def model_predictions_disagree(predictions: dict) -> bool:
    classes = {int(item["class_id"]) for item in predictions.values()}
    return len(classes) > 1
