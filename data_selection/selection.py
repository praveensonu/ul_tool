from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from scipy.optimize import nnls
from sklearn.cluster import KMeans


SelectionMethod = Literal["raslik", "grace"]
EPSILON = 1e-12


def validate_selection_parameters(
    *,
    method: SelectionMethod,
    dataset_size: int,
    forget_size: int,
    retain_size: int,
    top_n: int | None = None,
    num_clusters: int | None = None,
) -> None:
    if forget_size < 1 or retain_size < 1:
        raise ValueError("forget_size and retain_size must both be at least 1.")

    if method == "raslik":
        if forget_size + retain_size > dataset_size:
            raise ValueError(
                "RASLIK requires forget_size + retain_size to be no greater "
                "than the full dataset size."
            )
        return

    if method != "grace":
        raise ValueError(f"Unsupported selection method: {method}")
    if top_n is None or num_clusters is None:
        raise ValueError("GRACE requires top_n and num_clusters.")
    if top_n < forget_size:
        raise ValueError("GRACE top_n must be at least forget_size.")
    if top_n + retain_size > dataset_size:
        raise ValueError(
            "GRACE requires top_n + retain_size to be no greater than "
            "the full dataset size."
        )
    remaining_size = dataset_size - top_n
    if num_clusters > retain_size:
        raise ValueError("GRACE num_clusters cannot exceed retain_size.")
    if num_clusters > remaining_size:
        raise ValueError(
            "GRACE num_clusters cannot exceed the pool remaining after top_n."
        )


def _load_gradient_matrix(directory: Path) -> tuple[list[str], np.ndarray]:
    files = sorted(directory.glob("*.pt"), key=lambda path: path.stem)
    if not files:
        raise ValueError(f"No gradient .pt files found in {directory}.")

    ids: list[str] = []
    vectors: list[np.ndarray] = []
    expected_size: int | None = None
    for path in files:
        value = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Gradient file does not contain a tensor: {path}")
        vector = value.detach().float().reshape(-1).numpy()
        if expected_size is None:
            expected_size = vector.size
        elif vector.size != expected_size:
            raise ValueError(
                f"Gradient dimension mismatch in {path}: "
                f"expected {expected_size}, found {vector.size}."
            )
        if not np.isfinite(vector).all():
            raise ValueError(f"Gradient file contains NaN or infinity: {path}")
        ids.append(path.stem)
        vectors.append(vector)

    return ids, np.stack(vectors).astype(np.float32, copy=False)


def _mean_poison_gradient(poison_matrix: np.ndarray) -> np.ndarray:
    return poison_matrix.mean(axis=0, dtype=np.float32)


def _rank_by_average_inner_product(
    training_ids: list[str],
    training_matrix: np.ndarray,
    average_poison: np.ndarray,
) -> tuple[list[int], np.ndarray]:
    scores = training_matrix @ average_poison
    descending = sorted(
        range(len(training_ids)),
        key=lambda index: (-float(scores[index]), training_ids[index]),
    )
    return descending, scores


def _nnomp(
    target: np.ndarray,
    dictionary: np.ndarray,
    count: int,
    tolerance: float = 1e-4,
) -> list[int]:
    target = np.asarray(target, dtype=np.float32).reshape(-1)
    dictionary = np.asarray(dictionary, dtype=np.float32)
    target_norm = float(np.linalg.norm(target))
    if target_norm < EPSILON or count < 1:
        return []

    residual = target.copy()
    active: list[int] = []
    for _ in range(min(count, len(dictionary))):
        if float(np.linalg.norm(residual)) / (target_norm + EPSILON) < tolerance:
            break
        correlations = dictionary @ residual
        correlations = np.maximum(correlations, 0.0)
        if active:
            correlations[active] = -np.inf
        best = int(np.argmax(correlations))
        if not np.isfinite(correlations[best]) or correlations[best] < 1e-10:
            break
        active.append(best)
        coefficients, _ = nnls(dictionary[active].T, target)
        residual = target - dictionary[active].T @ coefficients

    return active


def _omp(
    target: np.ndarray,
    dictionary: np.ndarray,
    count: int,
    tolerance: float = 1e-4,
) -> list[int]:
    target = np.asarray(target, dtype=np.float32).reshape(-1)
    dictionary = np.asarray(dictionary, dtype=np.float32)
    target_norm = float(np.linalg.norm(target))
    if target_norm < EPSILON or count < 1:
        return []

    residual = target.copy()
    active: list[int] = []
    for _ in range(min(count, len(dictionary))):
        if float(np.linalg.norm(residual)) / (target_norm + EPSILON) < tolerance:
            break
        correlations = np.abs(dictionary @ residual)
        if active:
            correlations[active] = -np.inf
        best = int(np.argmax(correlations))
        if not np.isfinite(correlations[best]) or correlations[best] < 1e-10:
            break
        active.append(best)
        coefficients, *_ = np.linalg.lstsq(
            dictionary[active].T,
            target,
            rcond=None,
        )
        residual = target - dictionary[active].T @ coefficients

    return active


def _balanced_cluster_quotas(cluster_sizes: list[int], total: int) -> list[int]:
    quotas = [0] * len(cluster_sizes)
    assigned = 0
    while assigned < total:
        progressed = False
        for cluster_id, cluster_size in enumerate(cluster_sizes):
            if assigned >= total:
                break
            if quotas[cluster_id] < cluster_size:
                quotas[cluster_id] += 1
                assigned += 1
                progressed = True
        if not progressed:
            break
    return quotas


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, EPSILON)


def _fill_selection(
    selected: list[int],
    desired_count: int,
    candidate_order: list[int],
) -> list[int]:
    selected_set = set(selected)
    for index in candidate_order:
        if len(selected) >= desired_count:
            break
        if index not in selected_set:
            selected.append(index)
            selected_set.add(index)
    return selected


def select_raslik_ids(
    training_ids: list[str],
    training_matrix: np.ndarray,
    poison_matrix: np.ndarray,
    forget_size: int,
    retain_size: int,
) -> tuple[list[str], list[str], dict]:
    validate_selection_parameters(
        method="raslik",
        dataset_size=len(training_ids),
        forget_size=forget_size,
        retain_size=retain_size,
    )

    average_poison = _mean_poison_gradient(poison_matrix)
    descending, scores = _rank_by_average_inner_product(
        training_ids,
        training_matrix,
        average_poison,
    )
    forget_indices = descending[:forget_size]
    forget_index_set = set(forget_indices)
    ascending = sorted(
        range(len(training_ids)),
        key=lambda index: (float(scores[index]), training_ids[index]),
    )
    retain_indices = [
        index for index in ascending if index not in forget_index_set
    ][:retain_size]

    return (
        [training_ids[index] for index in forget_indices],
        [training_ids[index] for index in retain_indices],
        {
            "average_inner_product_scores": {
                training_ids[index]: float(scores[index])
                for index in range(len(training_ids))
            }
        },
    )


def select_grace_ids(
    training_ids: list[str],
    training_matrix: np.ndarray,
    poison_matrix: np.ndarray,
    forget_size: int,
    retain_size: int,
    top_n: int,
    num_clusters: int,
) -> tuple[list[str], list[str], dict]:
    dataset_size = len(training_ids)
    validate_selection_parameters(
        method="grace",
        dataset_size=dataset_size,
        forget_size=forget_size,
        retain_size=retain_size,
        top_n=top_n,
        num_clusters=num_clusters,
    )

    average_poison = _mean_poison_gradient(poison_matrix)
    descending, scores = _rank_by_average_inner_product(
        training_ids,
        training_matrix,
        average_poison,
    )
    candidate_indices = descending[:top_n]
    candidate_matrix = training_matrix[candidate_indices]

    selected_local = _nnomp(
        target=average_poison,
        dictionary=candidate_matrix,
        count=forget_size,
    )
    selected_local = _fill_selection(
        selected_local,
        desired_count=forget_size,
        candidate_order=list(range(top_n)),
    )
    forget_indices = [candidate_indices[index] for index in selected_local]

    remaining_indices = descending[top_n:]
    remaining_matrix = training_matrix[remaining_indices]
    average_norm_squared = float(average_poison @ average_poison)
    if average_norm_squared > EPSILON:
        coefficients = (
            (remaining_matrix @ average_poison) / average_norm_squared
        )[:, None]
        projected = remaining_matrix - coefficients * average_poison[None, :]
    else:
        projected = remaining_matrix.copy()

    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto")
    cluster_labels = kmeans.fit_predict(projected)
    centroids = kmeans.cluster_centers_.astype(np.float32, copy=False)
    cluster_members = [
        np.flatnonzero(cluster_labels == cluster_id)
        for cluster_id in range(num_clusters)
    ]
    quotas = _balanced_cluster_quotas(
        [len(members) for members in cluster_members],
        retain_size,
    )

    normalized_projected = _normalize_rows(projected)
    retain_local_indices: list[int] = []
    cluster_counts: dict[str, int] = {}
    for cluster_id, (members, quota) in enumerate(zip(cluster_members, quotas)):
        if quota == 0:
            cluster_counts[str(cluster_id)] = 0
            continue
        cluster_dictionary = normalized_projected[members]
        selected = _omp(
            target=centroids[cluster_id],
            dictionary=cluster_dictionary,
            count=quota,
        )
        representative_order = np.argsort(
            np.linalg.norm(projected[members] - centroids[cluster_id], axis=1),
            kind="stable",
        ).tolist()
        selected = _fill_selection(selected, quota, representative_order)
        chosen = [int(members[index]) for index in selected]
        retain_local_indices.extend(chosen)
        cluster_counts[str(cluster_id)] = len(chosen)

    if len(retain_local_indices) != retain_size:
        raise RuntimeError(
            f"GRACE selected {len(retain_local_indices)} retain samples; "
            f"expected {retain_size}."
        )

    retain_indices = [remaining_indices[index] for index in retain_local_indices]
    return (
        [training_ids[index] for index in forget_indices],
        [training_ids[index] for index in retain_indices],
        {
            "top_n_ids": [training_ids[index] for index in candidate_indices],
            "cluster_selection_counts": cluster_counts,
            "average_inner_product_scores": {
                training_ids[index]: float(scores[index])
                for index in range(len(training_ids))
            },
        },
    )


def _select_rows_by_id(dataframe: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
    indexed = dataframe.copy()
    indexed["id"] = indexed["id"].astype(str)
    indexed = indexed.set_index("id", drop=False)
    missing = [sample_id for sample_id in ids if sample_id not in indexed.index]
    if missing:
        raise ValueError(
            f"Selected gradient ids are missing from the full dataset: {missing[:5]}"
        )
    return indexed.loc[ids].reset_index(drop=True)


def select_and_export_datasets(
    *,
    method: SelectionMethod,
    training_grads_path: Path,
    poison_grads_path: Path,
    training_data_path: Path,
    output_dir: Path,
    forget_size: int,
    retain_size: int,
    top_n: int | None = None,
    num_clusters: int | None = None,
) -> dict:
    training_ids, training_matrix = _load_gradient_matrix(training_grads_path)
    _, poison_matrix = _load_gradient_matrix(poison_grads_path)
    if training_matrix.shape[1] != poison_matrix.shape[1]:
        raise ValueError(
            "Training and poison gradient dimensions do not match: "
            f"{training_matrix.shape[1]} != {poison_matrix.shape[1]}."
        )

    full_dataframe = pd.read_json(training_data_path, lines=True)
    data_ids = full_dataframe["id"].astype(str).tolist()
    if set(training_ids) != set(data_ids):
        missing_gradients = sorted(set(data_ids) - set(training_ids))
        unknown_gradients = sorted(set(training_ids) - set(data_ids))
        raise ValueError(
            "Training gradient ids do not match the normalized full dataset "
            f"(missing={missing_gradients[:5]}, unknown={unknown_gradients[:5]})."
        )

    validate_selection_parameters(
        method=method,
        dataset_size=len(training_ids),
        forget_size=forget_size,
        retain_size=retain_size,
        top_n=top_n,
        num_clusters=num_clusters,
    )

    if method == "raslik":
        forget_ids, retain_ids, metadata = select_raslik_ids(
            training_ids,
            training_matrix,
            poison_matrix,
            forget_size,
            retain_size,
        )
    elif method == "grace":
        assert top_n is not None and num_clusters is not None
        forget_ids, retain_ids, metadata = select_grace_ids(
            training_ids,
            training_matrix,
            poison_matrix,
            forget_size,
            retain_size,
            top_n,
            num_clusters,
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    forget_dataframe = _select_rows_by_id(full_dataframe, forget_ids)
    retain_dataframe = _select_rows_by_id(full_dataframe, retain_ids)
    forget_path = output_dir / "forget.parquet"
    retain_path = output_dir / "retain.parquet"
    forget_dataframe.to_parquet(forget_path, index=False)
    retain_dataframe.to_parquet(retain_path, index=False)

    with (output_dir / "selection.json").open("w", encoding="utf-8") as output:
        json.dump(
            {
                "method": method,
                "forget_ids": forget_ids,
                "retain_ids": retain_ids,
                **metadata,
            },
            output,
            indent=2,
        )
        output.write("\n")
    torch.save(
        torch.from_numpy(_mean_poison_gradient(poison_matrix)),
        output_dir / "average_poison_gradient.pt",
    )

    return {
        "method": method,
        "forget_path": forget_path,
        "retain_path": retain_path,
        "forget_dataframe": forget_dataframe,
        "retain_dataframe": retain_dataframe,
        "metadata_path": output_dir / "selection.json",
    }
