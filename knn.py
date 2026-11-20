"""
k-Nearest Neighbors (k-NN) Algorithm Implementation
SE313.R11 - Intelligent Algorithms
Supports:
- Distance metrics: Euclidean, Manhattan, Chebyshev, Cosine
- Min-Max Normalization
- Tasks: Classification (Majority Vote, 1/d Weighting, 1-NN Tie-break)
         Regression (Mean, 1/d Inverse Distance Weighting)
- Model Evaluation: Leave-One-Out Cross-Validation (LOOCV)
"""

import math
from typing import List, Dict, Any, Tuple, Optional, Union


# ============================================================================
# Distance Metrics
# ============================================================================

def euclidean_distance(a: List[float], b: List[float]) -> float:
    """Euclidean distance (L2 norm)."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def manhattan_distance(a: List[float], b: List[float]) -> float:
    """Manhattan distance (L1 norm / City block)."""
    return sum(abs(x - y) for x, y in zip(a, b))


def chebyshev_distance(a: List[float], b: List[float]) -> float:
    """Chebyshev distance (L-infinity norm / Chessboard)."""
    return max(abs(x - y) for x, y in zip(a, b)) if a else 0.0


def cosine_distance(a: List[float], b: List[float]) -> float:
    """Cosine distance (1 - cosine similarity)."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a > 0 and norm_b > 0:
        sim = dot / (norm_a * norm_b)
        return max(0.0, 1.0 - sim)
    return 1.0


DIST_METRICS = {
    "euclid": euclidean_distance,
    "manhattan": manhattan_distance,
    "chebyshev": chebyshev_distance,
    "cosine": cosine_distance,
}


# ============================================================================
# Min-Max Normalization
# ============================================================================

def get_bounds(data: List[Dict[str, Any]], dims: int = 2) -> Tuple[List[float], List[float]]:
    """Compute Min-Max bounding box for normalization."""
    if not data:
        return [0.0] * dims, [1.0] * dims

    mins = [float("inf")] * dims
    maxs = [float("-inf")] * dims

    for p in data:
        coords = [p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0)][:dims]
        for i in range(dims):
            if coords[i] < mins[i]:
                mins[i] = coords[i]
            if coords[i] > maxs[i]:
                maxs[i] = coords[i]

    for i in range(dims):
        if mins[i] == maxs[i]:
            mins[i] -= 1.0
            maxs[i] += 1.0

    return mins, maxs


def normalize_coords(coords: List[float], mins: List[float], maxs: List[float]) -> List[float]:
    """Scale coordinates into [0, 1] range."""
    res = []
    for val, mi, ma in zip(coords, mins, maxs):
        span = ma - mi if ma != mi else 1.0
        res.append((val - mi) / span)
    return res


# ============================================================================
# k-NN Core Model
# ============================================================================

class KNN:
    """
    k-Nearest Neighbors classifier and regressor.
    """

    def __init__(
        self,
        k: int = 5,
        metric: str = "euclid",
        norm: bool = False,
        weighting: bool = False,
        task: str = "classification",
    ):
        self.k = k
        self.metric_name = metric
        self.metric_fn = DIST_METRICS.get(metric, euclidean_distance)
        self.norm = norm
        self.weighting = weighting
        self.task = task
        self.data: List[Dict[str, Any]] = []
        self.classes: List[int] = []
        self.mins: List[float] = []
        self.maxs: List[float] = []

    def fit(self, data: List[Dict[str, Any]], classes: Optional[List[int]] = None) -> "KNN":
        """Store training data and calculate normalization bounds."""
        self.data = list(data)
        if classes is not None:
            self.classes = list(classes)
        else:
            self.classes = sorted(list({p.get("c", 0) for p in data}))

        if self.norm:
            self.mins, self.maxs = get_bounds(self.data, dims=2)
        return self

    def predict(self, point_coords: List[float], k: Optional[int] = None) -> Dict[str, Any]:
        """
        Predict class label or continuous target value for a query point.
        """
        if not self.data:
            return {
                "label": -1,
                "y_hat": 0.0,
                "nearest": [],
                "votes": {},
                "weights": {},
                "r": 0.0,
                "is_regression": (self.task == "regression"),
            }

        k_val = k if k is not None else self.k
        query_pt = point_coords[:2]

        if self.norm:
            query_pt = normalize_coords(query_pt, self.mins, self.maxs)

        # 1. Compute distances to all points
        distances = []
        for idx, p in enumerate(self.data):
            raw = [p.get("x", 0.0), p.get("y", 0.0)]
            p_coords = normalize_coords(raw, self.mins, self.maxs) if self.norm else raw
            d = self.metric_fn(query_pt, p_coords)
            distances.append({"index": idx, "point": p, "dist": d})

        # 2. Stable sort by distance with epsilon tie-break (eliminates float noise)
        distances.sort(key=lambda item: (round(item["dist"], 6), item["index"]))

        # 3. Extract k nearest neighbors
        k_actual = min(k_val, len(distances))
        nearest = distances[:k_actual]
        radius = nearest[-1]["dist"] if nearest else 0.0

        # --- Case A: Regression ---
        if self.task == "regression":
            sum_y = 0.0
            sum_w = 0.0
            for item in nearest:
                val = item["point"].get("val", item["point"].get("y", 0.0))
                w = (1.0 / (item["dist"] + 1e-5)) if self.weighting else 1.0
                sum_y += val * w
                sum_w += w

            y_hat = (sum_y / sum_w) if sum_w > 0 else 0.0
            return {
                "label": -1,
                "y_hat": round(y_hat, 2),
                "nearest": nearest,
                "radius": radius,
                "is_regression": True,
            }

        # --- Case B: Classification ---
        votes = {cid: 0 for cid in self.classes}
        weights = {cid: 0.0 for cid in self.classes}

        for item in nearest:
            cid = item["point"].get("c", 0)
            if cid in votes:
                votes[cid] += 1
            w = (1.0 / (item["dist"] + 1e-5)) if self.weighting else 1.0
            if cid in weights:
                weights[cid] += w

        best_class = -1
        max_score = -1.0
        tied_classes = []

        for cid in self.classes:
            score = weights[cid] if self.weighting else float(votes[cid])
            if score > max_score:
                max_score = score
                best_class = cid
                tied_classes = [cid]
            elif score == max_score and score > 0:
                tied_classes.append(cid)

        # Tie-breaking: 1-NN nearest neighbor fallback
        if len(tied_classes) > 1:
            for item in nearest:
                cid = item["point"].get("c", 0)
                if cid in tied_classes:
                    best_class = cid
                    break

        return {
            "label": best_class,
            "votes": votes,
            "weights": weights,
            "nearest": nearest,
            "radius": radius,
            "is_regression": False,
        }


# ============================================================================
# Leave-One-Out Cross-Validation (LOOCV)
# ============================================================================

def evaluate_loocv(
    data: List[Dict[str, Any]],
    k: int = 5,
    metric: str = "euclid",
    norm: bool = False,
    weighting: bool = False,
    task: str = "classification",
) -> Dict[str, Any]:
    """
    Evaluate k-NN performance using Leave-One-Out Cross-Validation.
    Returns accuracy (%) for classification, or MSE / RMSE for regression.
    """
    n = len(data)
    if n <= 1:
        return {"accuracy": 0.0, "mse": 0.0, "rmse": 0.0, "total": n, "correct": 0}

    classes = sorted(list({p.get("c", 0) for p in data})) if task == "classification" else []
    correct = 0
    total_error_sq = 0.0

    for i in range(n):
        test_pt = data[i]
        train_data = data[:i] + data[i + 1:]

        model = KNN(k=k, metric=metric, norm=norm, weighting=weighting, task=task)
        model.fit(train_data, classes=classes)
        pred = model.predict([test_pt.get("x", 0.0), test_pt.get("y", 0.0)])

        if task == "regression":
            actual_y = test_pt.get("val", test_pt.get("y", 0.0))
            diff = pred["y_hat"] - actual_y
            total_error_sq += diff * diff
        else:
            if pred["label"] == test_pt.get("c", 0):
                correct += 1

    if task == "regression":
        mse = total_error_sq / n
        return {
            "mse": round(mse, 4),
            "rmse": round(math.sqrt(mse), 4),
            "total": n,
        }

    accuracy = (correct / n) * 100.0
    return {
        "accuracy": round(accuracy, 2),
        "correct": correct,
        "total": n,
    }


def format_casio(val: float, max_dec: int = 10) -> str:
    """Format floating point numbers like Casio calculator (up to max_dec decimal places, stripping trailing zeros)."""
    if val is None:
        return "—"
    s = f"{float(val):.{max_dec}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return "0" if s == "-0" or not s else s


# ============================================================================
# Example / Unit Test Run with "Đậu hay Rớt?" Dataset (Nhóm 5 SE313.R11)
# ============================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  k-NN Algorithm in Python · Đậu hay Rớt? (Nhóm 5 - SE313.R11)")
    print("=" * 65)

    # 8 students from presentation document:
    # x: giờ tự học/tuần, y: số buổi vắng, c: 1 (Đậu) / 0 (Rớt), val: Điểm cuối kỳ
    students_data = [
        {"id": 0, "label": "A", "x": 8.0, "y": 1.0, "c": 1, "val": 8.0},
        {"id": 1, "label": "B", "x": 6.0, "y": 2.0, "c": 1, "val": 6.5},
        {"id": 2, "label": "C", "x": 9.0, "y": 3.0, "c": 1, "val": 8.5},
        {"id": 3, "label": "D", "x": 7.0, "y": 1.0, "c": 1, "val": 7.0},
        {"id": 4, "label": "E", "x": 9.0, "y": 2.0, "c": 1, "val": 9.0},
        {"id": 5, "label": "F", "x": 2.0, "y": 3.0, "c": 0, "val": 3.5},
        {"id": 6, "label": "G", "x": 1.0, "y": 5.0, "c": 0, "val": 2.0},
        {"id": 7, "label": "H", "x": 5.0, "y": 4.0, "c": 0, "val": 4.5},
    ]

    class_names = {0: "Rớt", 1: "Đậu"}

    # 1. Classification for Z = (5, 3)
    query_z = [5.0, 3.0]
    print(f"\n[1. Phân lớp cho Điểm truy vấn Z = (5, 3)]")

    for k_val in [1, 3, 5]:
        model = KNN(k=k_val, metric="euclid", task="classification")
        model.fit(students_data, classes=[0, 1])
        res = model.predict(query_z)
        nb_str = ", ".join([f"{n['point']['label']} (d={format_casio(n['dist'])})" for n in res["nearest"]])
        print(f" • k = {k_val}: Dự đoán = {class_names[res['label']]} | Phiếu = {res['votes']} | Láng giềng: [{nb_str}]")

    # Manhattan comparison for Z = (5, 3) at k = 3
    model_man = KNN(k=3, metric="manhattan", task="classification")
    model_man.fit(students_data, classes=[0, 1])
    res_man = model_man.predict(query_z)
    nb_man = ", ".join([f"{n['point']['label']} (d={format_casio(n['dist'])})" for n in res_man["nearest"]])
    print(f" • Manhattan (k = 3): Dự đoán = {class_names[res_man['label']]} | Láng giềng: [{nb_man}] (Đổi hàm đổi kết quả)")

    # Cosine comparison for Z = (5, 3) at k = 5
    model_cos = KNN(k=5, metric="cosine", task="classification")
    model_cos.fit(students_data, classes=[0, 1])
    res_cos = model_cos.predict(query_z)
    nb_cos = ", ".join([f"{n['point']['label']} (d={format_casio(n['dist'])})" for n in res_cos["nearest"]])
    print(f" • Cosine (k = 5): Dự đoán = {class_names[res_cos['label']]} | Láng giềng: [{nb_cos}]")

    # 2. Regression for Z = (5, 3) (Điểm cuối kỳ)
    print(f"\n[2. Hồi quy dự đoán Điểm cuối kỳ cho Z = (5, 3)]")
    for k_val in [1, 3, 5, 8]:
        reg_model = KNN(k=k_val, metric="euclid", task="regression")
        reg_model.fit(students_data)
        res_reg = reg_model.predict(query_z)
        vals_str = ", ".join([f"{n['point']['label']}:{n['point']['val']}" for n in res_reg["nearest"]])
        print(f" • k = {k_val}: ŷ = {res_reg['y_hat']} điểm | Láng giềng: [{vals_str}]")

    # 3. Model Evaluation LOOCV
    print(f"\n[3. Đánh giá mô hình Kiểm thử chéo Leave-One-Out (LOOCV)]")
    cv_cls = evaluate_loocv(students_data, k=3, task="classification")
    print(f" • Phân lớp k = 3 LOOCV: Accuracy = {cv_cls['accuracy']}% ({cv_cls['correct']}/{cv_cls['total']} đúng)")
    cv_reg = evaluate_loocv(students_data, k=3, task="regression")
    print(f" • Hồi quy k = 3 LOOCV: RMSE = {cv_reg['rmse']}, MSE = {cv_reg['mse']}")
    print("=" * 65)
