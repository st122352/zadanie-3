"""Small, reproducible PCA / factor / clustering study for SML2010.

Uses NumPy plus the Python standard library. Run from this folder:
    python analysis.py
"""
from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
SEED = 20261117
FA_LIMIT = 2000
CLUSTER_LIMIT = 1200

CLIMATE = [
    "Temperature_Comedor_Sensor",
    "Temperature_Habitacion_Sensor",
    "Weather_Temperature",
    "Temperature_Exterior_Sensor",
    "Humedad_Comedor_Sensor",
    "Humedad_Habitacion_Sensor",
    "Humedad_Exterior_Sensor",
]
SOLAR = [
    "Meteo_Exterior_Sol_Oest",
    "Meteo_Exterior_Sol_Est",
    "Meteo_Exterior_Sol_Sud",
    "Meteo_Exterior_Piranometro",
]
LIGHT = ["Lighting_Comedor_Sensor", "Lighting_Habitacion_Sensor"]
CLIMATE_SOLAR = CLIMATE + SOLAR
EXPANDED = CLIMATE_SOLAR + LIGHT

LABELS = {
    "Temperature_Comedor_Sensor": "Dining-room temperature (°C)",
    "Temperature_Habitacion_Sensor": "Room temperature (°C)",
    "Weather_Temperature": "Forecast outdoor temperature (°C)",
    "Temperature_Exterior_Sensor": "Outdoor temperature (°C)",
    "Humedad_Comedor_Sensor": "Dining-room humidity (%)",
    "Humedad_Habitacion_Sensor": "Room humidity (%)",
    "Humedad_Exterior_Sensor": "Outdoor humidity (%)",
    "Meteo_Exterior_Sol_Oest": "West facade sunlight (lux)",
    "Meteo_Exterior_Sol_Est": "East facade sunlight (lux)",
    "Meteo_Exterior_Sol_Sud": "South facade sunlight (lux)",
    "Meteo_Exterior_Piranometro": "Solar irradiance (W/m²)",
    "Lighting_Comedor_Sensor": "Dining-room illuminance (lux)",
    "Lighting_Habitacion_Sensor": "Room illuminance (lux)",
}
COLORS = ["#2878b5", "#de7d22", "#26936c", "#b94d58", "#8c63ad"]


def load_files() -> tuple[list[str], np.ndarray, list[datetime], dict[str, tuple[int, int]]]:
    files = [DATA_DIR / "NEW-DATA-1.T15.txt", DATA_DIR / "NEW-DATA-2.T15.txt"]
    all_names: list[str] | None = None
    rows: list[list[float]] = []
    stamps: list[datetime] = []
    boundaries: dict[str, tuple[int, int]] = {}
    offset = 0
    for path in files:
        lines = path.read_text(encoding="utf-8").splitlines()
        header = next((line for line in lines if line.startswith("#")), None)
        if header is None:
            raise ValueError(f"Missing column header in {path.name}")
        names = [name for _, name in re.findall(r"(\d+):([A-Za-z0-9_]+)", header)]
        if all_names is None:
            all_names = names
        elif names != all_names:
            raise ValueError("The two source files have different schemas")
        start = len(rows)
        for line_no, line in enumerate(lines, 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split()
            if len(fields) != len(names):
                raise ValueError(f"{path.name}:{line_no}: expected {len(names)} columns, got {len(fields)}")
            stamp = datetime.strptime(fields[0] + " " + fields[1], "%d/%m/%Y %H:%M")
            try:
                values = [float(v) for v in fields[2:]]
            except ValueError as exc:
                raise ValueError(f"{path.name}:{line_no}: non-numeric measurement") from exc
            if not np.isfinite(values).all():
                raise ValueError(f"{path.name}:{line_no}: missing or non-finite measurement; no future-value imputation is used")
            stamps.append(stamp)
            rows.append(values)
        boundaries[path.name] = (start, len(rows))
        offset += len(rows) - start
    assert all_names is not None
    return all_names, np.asarray(rows, dtype=float), stamps, boundaries


def select_matrix(names: list[str], values: np.ndarray, columns: list[str]) -> np.ndarray:
    indexes = [names.index(col) - 2 for col in columns]
    return values[:, indexes]


def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0, ddof=0)
    scale[scale < 1e-12] = 1.0
    return (x - mean) / scale, mean, scale


def pca_fit(x: np.ndarray) -> dict:
    z, mean, scale = standardize_fit(x)
    _, singular, vt = np.linalg.svd(z, full_matrices=False)
    eigen = singular**2 / (len(z) - 1)
    explained = eigen / eigen.sum()
    loadings = vt.T * np.sqrt(eigen)[None, :]
    scores = z @ vt.T
    # Fix the arbitrary sign so plots and reruns are easy to compare.
    for j in range(loadings.shape[1]):
        i = int(np.argmax(np.abs(loadings[:, j])))
        if loadings[i, j] < 0:
            loadings[:, j] *= -1
            scores[:, j] *= -1
            vt[j, :] *= -1
    return {"mean": mean, "scale": scale, "vectors": vt.T, "loadings": loadings,
            "scores": scores, "eigenvalues": eigen, "explained": explained}


def reconstruct(x: np.ndarray, model: dict, k: int) -> np.ndarray:
    z = (x - model["mean"]) / model["scale"]
    v = model["vectors"][:, :k]
    return (z @ v) @ v.T


def varimax(loadings: np.ndarray, gamma: float = 1.0, q: int = 100, tol: float = 1e-7) -> tuple[np.ndarray, np.ndarray]:
    p, k = loadings.shape
    rotation = np.eye(k)
    old = 0.0
    for _ in range(q):
        rotated = loadings @ rotation
        u, singular, vh = np.linalg.svd(
            loadings.T @ (rotated**3 - (gamma / p) * rotated @ np.diag(np.sum(rotated**2, axis=0))),
            full_matrices=False,
        )
        rotation = u @ vh
        value = float(singular.sum())
        if old and value - old < tol:
            break
        old = value
    return loadings @ rotation, rotation


def principal_axis_fa(x: np.ndarray, n_factors: int) -> dict:
    z, mean, scale = standardize_fit(x)
    corr = np.corrcoef(z, rowvar=False)
    inv = np.linalg.pinv(corr)
    communalities = np.clip(1 - 1 / np.diag(inv), 0.05, 0.99)
    raw = None
    for _ in range(500):
        reduced = corr.copy()
        np.fill_diagonal(reduced, communalities)
        eigenvalues, eigenvectors = np.linalg.eigh(reduced)
        order = np.argsort(eigenvalues)[::-1][:n_factors]
        raw = eigenvectors[:, order] * np.sqrt(np.maximum(eigenvalues[order], 1e-8))[None, :]
        updated = np.clip(np.sum(raw**2, axis=1), 0.01, 0.995)
        if np.max(np.abs(updated - communalities)) < 1e-7:
            communalities = updated
            break
        communalities = updated
    assert raw is not None
    # Avoid Heywood cases (communalities above 1) in a small principal-axis solution.
    raw_communalities = np.sum(raw**2, axis=1)
    for i, value in enumerate(raw_communalities):
        if value > 0.995:
            raw[i] *= np.sqrt(0.995 / value)
    communalities = np.sum(raw**2, axis=1)
    rotated, rotation = varimax(raw)
    return {"mean": mean, "scale": scale, "loadings": rotated, "unrotated": raw,
            "communalities": communalities, "rotation": rotation, "correlation": corr}


def parallel_analysis(x: np.ndarray, simulations: int = 100) -> dict:
    z, _, _ = standardize_fit(x)
    observed = np.linalg.eigvalsh(np.corrcoef(z, rowvar=False))[::-1]
    rng = np.random.default_rng(SEED + 11)
    random_eigen = []
    for _ in range(simulations):
        noise = rng.standard_normal(z.shape)
        random_eigen.append(np.linalg.eigvalsh(np.corrcoef(noise, rowvar=False))[::-1])
    cutoff = np.quantile(np.asarray(random_eigen), 0.95, axis=0)
    factors = int(np.sum(observed > cutoff))
    factors = max(1, min(factors, min(5, x.shape[1])))
    return {"observed": observed, "random_95": cutoff, "n_factors": factors}


def kmeans(x: np.ndarray, k: int, seed: int, n_init: int = 8, max_iter: int = 100) -> tuple[np.ndarray, np.ndarray, float]:
    best = None
    rng = np.random.default_rng(seed)
    for _ in range(n_init):
        centers = x[rng.choice(len(x), k, replace=False)].copy()
        labels = np.zeros(len(x), dtype=int)
        for _ in range(max_iter):
            distances = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            new_labels = np.argmin(distances, axis=1)
            new_centers = np.vstack([
                x[new_labels == j].mean(axis=0) if np.any(new_labels == j) else x[rng.integers(len(x))]
                for j in range(k)
            ])
            if np.array_equal(new_labels, labels) and np.allclose(new_centers, centers):
                centers = new_centers
                labels = new_labels
                break
            labels, centers = new_labels, new_centers
        inertia = float(np.sum((x - centers[labels]) ** 2))
        if best is None or inertia < best[2]:
            best = (labels.copy(), centers.copy(), inertia)
    assert best is not None
    return best


def silhouette_score(x: np.ndarray, labels: np.ndarray) -> float:
    unique = np.unique(labels)
    if len(unique) < 2:
        return float("nan")
    distances = np.sqrt(np.maximum(np.sum(x*x, axis=1)[:, None] + np.sum(x*x, axis=1)[None, :] - 2*x@x.T, 0))
    vals = []
    for i in range(len(x)):
        same = labels == labels[i]
        same[i] = False
        a = float(distances[i, same].mean()) if np.any(same) else 0.0
        b = min(float(distances[i, labels == other].mean()) for other in unique if other != labels[i])
        vals.append((b - a) / max(a, b, 1e-15))
    return float(np.mean(vals))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def svg_heatmap(names: list[str], loadings: np.ndarray, explained: np.ndarray, n_components: int = 5) -> str:
    k = min(n_components, loadings.shape[1])
    left, top, cellw, cellh = 260, 82, 82, 31
    w, h = left + cellw * k + 45, top + cellh * len(names) + 60
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
           '<rect width="100%" height="100%" fill="white"/>',
           f'<text x="{w/2}" y="27" text-anchor="middle" font-family="Arial" font-size="18">Expanded-set PCA: variable loadings</text>']
    for j in range(k):
        x = left + j * cellw + cellw / 2
        out.append(f'<text x="{x}" y="58" text-anchor="middle" font-family="Arial" font-size="12">PC{j+1} ({explained[j]*100:.1f}%)</text>')
    for i, name in enumerate(names):
        y = top + i * cellh
        out.append(f'<text x="{left-10}" y="{y+21}" text-anchor="end" font-family="Arial" font-size="11">{LABELS[name]}</text>')
        for j in range(k):
            val = float(loadings[i, j])
            intensity = min(abs(val), 1.0)
            if val >= 0:
                color = f'rgb({int(255-100*intensity)},{int(255-35*intensity)},{int(255-35*intensity)})'
            else:
                color = f'rgb({int(255-35*intensity)},{int(255-80*intensity)},{int(255-100*intensity)})'
            x = left + j * cellw
            out += [f'<rect x="{x}" y="{y}" width="{cellw}" height="{cellh}" fill="{color}" stroke="white"/>',
                    f'<text x="{x+cellw/2}" y="{y+20}" text-anchor="middle" font-family="Arial" font-size="11">{val:+.2f}</text>']
    out.append('</svg>')
    return "\n".join(out)


def svg_temporal(times: list[datetime], scores: np.ndarray, solar_z: np.ndarray, start: datetime, end: datetime) -> str:
    indices = [i for i, t in enumerate(times) if start <= t <= end]
    if len(indices) == 0:
        raise ValueError("No observations in the selected temporal figure window")
    # A few days are still visually clear when drawn at every 15-minute measurement.
    if len(indices) > 500:
        indices = indices[::int(np.ceil(len(indices) / 500))]
    sub_times = [times[i] for i in indices]
    x0, x1 = sub_times[0].timestamp(), sub_times[-1].timestamp()
    series = [scores[indices, 0], scores[indices, 1], solar_z[indices]]
    lo = min(float(v.min()) for v in series)
    hi = max(float(v.max()) for v in series)
    pad = 0.08 * max(hi - lo, 1.0)
    lo, hi = lo - pad, hi + pad
    left, top, pw, ph = 76, 52, 830, 350
    sx = lambda t: left + (t.timestamp() - x0) / max(x1 - x0, 1) * pw
    sy = lambda v: top + ph - (float(v) - lo) / (hi - lo) * ph
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="960" height="480" viewBox="0 0 960 480">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<text x="480" y="26" text-anchor="middle" font-family="Arial" font-size="18">Later period: component scores and solar irradiance</text>']
    for j in range(5):
        val = lo + j * (hi-lo) / 4
        y = sy(val)
        out += [f'<line x1="{left}" y1="{y:.1f}" x2="{left+pw}" y2="{y:.1f}" stroke="#e5e7eb"/>',
                f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="10">{val:.1f}</text>']
    for d in sorted(set(t.date() for t in sub_times)):
        t = datetime.combine(d, datetime.min.time())
        if sub_times[0] <= t <= sub_times[-1]:
            x = sx(t)
            out += [f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+ph}" stroke="#f0f0f0"/>',
                    f'<text x="{x:.1f}" y="{top+ph+18}" text-anchor="middle" font-family="Arial" font-size="10">{d.strftime("%d %b")}</text>']
    for j, vals in enumerate(series):
        points = " ".join(f"{sx(t):.1f},{sy(v):.1f}" for t, v in zip(sub_times, vals))
        color = ["#2878b5", "#de7d22", "#26936c"][j]
        out.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.5" opacity=".9"/>')
    out += [f'<line x1="{left}" y1="{top+ph}" x2="{left+pw}" y2="{top+ph}" stroke="#333"/>',
            f'<text x="200" y="454" font-family="Arial" font-size="12" fill="#2878b5">PC1 (training PCA)</text>',
            f'<text x="390" y="454" font-family="Arial" font-size="12" fill="#de7d22">PC2 (training PCA)</text>',
            f'<text x="590" y="454" font-family="Arial" font-size="12" fill="#26936c">Solar irradiance (training z-score)</text>',
            '<text x="18" y="230" transform="rotate(-90 18 230)" text-anchor="middle" font-family="Arial" font-size="12">Standardized score</text>',
            '</svg>']
    return "\n".join(out)


def svg_clusters(scores: np.ndarray, labels: np.ndarray) -> str:
    w, h, left, top, pw, ph = 720, 470, 70, 45, 600, 330
    x, y = scores[:, 0], scores[:, 1]
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    sx = lambda a: left + (float(a)-xmin)/max(xmax-xmin, 1e-12)*pw
    sy = lambda a: top + ph - (float(a)-ymin)/max(ymax-ymin, 1e-12)*ph
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<text x="360" y="25" text-anchor="middle" font-family="Arial" font-size="17">K-means clusters in expanded PCA scores</text>']
    for i in range(len(x)):
        color = COLORS[int(labels[i]) % len(COLORS)]
        out.append(f'<circle cx="{sx(x[i]):.1f}" cy="{sy(y[i]):.1f}" r="3" fill="{color}" fill-opacity=".62"/>')
    out += [f'<line x1="{left}" y1="{top+ph}" x2="{left+pw}" y2="{top+ph}" stroke="#333"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+ph}" stroke="#333"/>',
            f'<text x="{left+pw/2}" y="{top+ph+36}" text-anchor="middle" font-family="Arial" font-size="12">PC1 score</text>',
            f'<text x="18" y="{top+ph/2}" transform="rotate(-90 18 {top+ph/2})" text-anchor="middle" font-family="Arial" font-size="12">PC2 score</text>']
    for lab in sorted(np.unique(labels)):
        out += [f'<circle cx="{left+40+lab*95}" cy="430" r="5" fill="{COLORS[int(lab)%len(COLORS)]}"/>',
                f'<text x="{left+50+lab*95}" y="434" font-family="Arial" font-size="11">Cluster {lab+1}</text>']
    out.append('</svg>')
    return "\n".join(out)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    names, values, times, bounds = load_files()
    name_index = {name: i for i, name in enumerate(names)}
    if any(name not in name_index for name in CLIMATE + SOLAR + LIGHT):
        raise ValueError("A required climate, solar, or lighting variable is absent")
    x_climate = select_matrix(names, values, CLIMATE)
    x_climate_solar = select_matrix(names, values, CLIMATE_SOLAR)
    x_expanded = select_matrix(names, values, EXPANDED)

    models = {
        "climate": (CLIMATE, pca_fit(x_climate)),
        "climate_plus_solar": (CLIMATE_SOLAR, pca_fit(x_climate_solar)),
        "expanded_with_room_light": (EXPANDED, pca_fit(x_expanded)),
    }
    pca_rows = []
    for model_name, (columns, model) in models.items():
        for j, ratio in enumerate(model["explained"]):
            pca_rows.append({
                "model": model_name, "component": f"PC{j+1}",
                "explained_variance_ratio": float(ratio),
                "cumulative_explained_variance": float(model["explained"][:j+1].sum()),
            })
    write_csv(RESULTS / "pca_explained_variance.csv", pca_rows)
    loading_rows = []
    for model_name, (columns, model) in models.items():
        for i, feature in enumerate(columns):
            row = {"model": model_name, "feature": feature, "description": LABELS[feature]}
            row.update({f"PC{j+1}": float(v) for j, v in enumerate(model["loadings"][i])})
            loading_rows.append(row)
    write_csv(RESULTS / "pca_loadings.csv", loading_rows)

    # Time-ordered holdout: file 1 is the earlier contiguous run; file 2 is later.
    early_start, early_end = bounds["NEW-DATA-1.T15.txt"]
    late_start, late_end = bounds["NEW-DATA-2.T15.txt"]
    split_rows = []
    temporal_models = {}
    for model_name, columns in (("climate", CLIMATE), ("climate_plus_solar", CLIMATE_SOLAR), ("expanded_with_room_light", EXPANDED)):
        matrix = select_matrix(names, values, columns)
        train = matrix[early_start:early_end]
        test = matrix[late_start:late_end]
        fitted = pca_fit(train)
        temporal_models[model_name] = (columns, fitted)
        for k in range(1, min(6, len(columns)+1)):
            train_recon = reconstruct(train, fitted, k)
            test_recon = reconstruct(test, fitted, k)
            split_rows.append({
                "model": model_name, "components": k,
                "train_standardized_rmse": float(np.sqrt(np.mean(((train-fitted["mean"])/fitted["scale"]-train_recon)**2))),
                "later_standardized_rmse": float(np.sqrt(np.mean(((test-fitted["mean"])/fitted["scale"]-test_recon)**2))),
                "train_rows": len(train), "later_rows": len(test),
            })
    write_csv(RESULTS / "temporal_reconstruction.csv", split_rows)

    # FA is deliberately capped at 2,000 early observations.
    fa_columns = EXPANDED
    fa_x = select_matrix(names, values, fa_columns)[early_start:early_start + min(FA_LIMIT, early_end-early_start)]
    parallel = parallel_analysis(fa_x)
    fa = principal_axis_fa(fa_x, parallel["n_factors"])
    fa_rows = []
    for i, feature in enumerate(fa_columns):
        fa_rows.append({
            "feature": feature, "description": LABELS[feature],
            **{f"Factor{j+1}": float(v) for j, v in enumerate(fa["loadings"][i])},
            "communality": float(fa["communalities"][i]),
        })
    write_csv(RESULTS / "factor_loadings.csv", fa_rows)
    write_csv(RESULTS / "parallel_analysis.csv", [
        {"component": j+1, "observed_eigenvalue": float(parallel["observed"][j]),
         "random_95th_percentile": float(parallel["random_95"][j]),
         "retained": j < parallel["n_factors"]}
        for j in range(len(parallel["observed"]))
    ])

    # K-means uses at most 1,200 deterministic rows across the full timeline.
    rng = np.random.default_rng(SEED)
    cluster_all, cluster_mean, cluster_scale = standardize_fit(x_expanded)
    sample_idx = np.sort(rng.choice(len(cluster_all), min(CLUSTER_LIMIT, len(cluster_all)), replace=False))
    cluster_sample = cluster_all[sample_idx]
    cluster_pca = pca_fit(x_expanded)
    sample_scores = cluster_pca["scores"][sample_idx, :2]
    cluster_candidates = []
    cluster_fits = {}
    for k in range(2, 7):
        labels, centers, inertia = kmeans(cluster_sample, k, SEED+k)
        sil = silhouette_score(cluster_sample, labels)
        cluster_candidates.append({"k": k, "silhouette": sil, "inertia": inertia})
        cluster_fits[k] = (labels, centers, inertia, sil)
    best_k = max(cluster_candidates, key=lambda row: row["silhouette"])["k"]
    cluster_labels, cluster_centers, cluster_inertia, cluster_silhouette = cluster_fits[best_k]
    write_csv(RESULTS / "cluster_selection.csv", cluster_candidates)
    cluster_rows = []
    for row, source_idx in enumerate(sample_idx):
        cluster_rows.append({"source_row_zero_based": int(source_idx), "timestamp_utc": times[source_idx].isoformat(),
                             "cluster": int(cluster_labels[row]+1), "PC1": float(sample_scores[row,0]),
                             "PC2": float(sample_scores[row,1])})
    write_csv(RESULTS / "cluster_assignments.csv", cluster_rows)
    solar_idx = [EXPANDED.index(n) for n in SOLAR]
    light_idx = [EXPANDED.index(n) for n in LIGHT]
    temp_idx = [EXPANDED.index(n) for n in CLIMATE[:4]]
    humidity_idx = [EXPANDED.index(n) for n in CLIMATE[4:]]
    cluster_summary = []
    for cluster in sorted(np.unique(cluster_labels)):
        mask = cluster_labels == cluster
        source = x_expanded[sample_idx[mask]]
        cluster_summary.append({
            "cluster": int(cluster+1), "rows": int(mask.sum()),
            "mean_solar_features_z": float(cluster_all[sample_idx[mask]][:,solar_idx].mean()),
            "mean_room_light_z": float(cluster_all[sample_idx[mask]][:,light_idx].mean()),
            "mean_temperature_z": float(cluster_all[sample_idx[mask]][:,temp_idx].mean()),
            "mean_humidity_z": float(cluster_all[sample_idx[mask]][:,humidity_idx].mean()),
        })
    write_csv(RESULTS / "cluster_profiles.csv", cluster_summary)

    # Solar-vs-microclimate summary: absolute average loading by physical group.
    expanded = models["expanded_with_room_light"][1]
    group_indices = {
        "indoor temperature and humidity": [EXPANDED.index(c) for c in CLIMATE if "Comedor" in c or "Habitacion" in c],
        "outdoor / forecast climate": [EXPANDED.index(c) for c in ["Weather_Temperature", "Temperature_Exterior_Sensor", "Humedad_Exterior_Sensor"]],
        "solar irradiance and facade light": [EXPANDED.index(c) for c in SOLAR],
        "indoor illuminance": [EXPANDED.index(c) for c in LIGHT],
    }
    component_groups = []
    for pc in range(len(EXPANDED)):
        row = {"component": f"PC{pc+1}", "explained_variance_ratio": float(expanded["explained"][pc])}
        for group, idx in group_indices.items():
            row[group] = float(np.mean(np.abs(expanded["loadings"][idx, pc])))
        component_groups.append(row)
    write_csv(RESULTS / "component_physical_profiles.csv", component_groups)

    later_times = times[late_start:late_end]
    expanded_train_pca = temporal_models["expanded_with_room_light"][1]
    later_matrix = select_matrix(names, values, EXPANDED)[late_start:late_end]
    later_z = (later_matrix - expanded_train_pca["mean"]) / expanded_train_pca["scale"]
    later_scores = later_z @ expanded_train_pca["vectors"]
    solar_raw = select_matrix(names, values, ["Meteo_Exterior_Piranometro"])[late_start:late_end, 0]
    train_solar = select_matrix(names, values, ["Meteo_Exterior_Piranometro"])[early_start:early_end, 0]
    solar_z = (solar_raw - train_solar.mean()) / max(train_solar.std(), 1e-12)
    # Choose four days near the middle of the later sequence for a readable view.
    fig_start = later_times[min(96, len(later_times)-1)]
    fig_end = fig_start.replace(day=min(fig_start.day+3, 28), hour=23, minute=45)
    if fig_end > later_times[-1]:
        fig_end = later_times[-1]
    (FIGURES / "temporal_dynamics.svg").write_text(svg_temporal(later_times, later_scores, solar_z, fig_start, fig_end), encoding="utf-8")
    (FIGURES / "loadings_heatmap.svg").write_text(
        svg_heatmap(EXPANDED, expanded["loadings"], expanded["explained"], n_components=5), encoding="utf-8")
    (FIGURES / "cluster_scores.svg").write_text(svg_clusters(sample_scores, cluster_labels), encoding="utf-8")

    time_rows = []
    for file_name, (start, stop) in bounds.items():
        time_rows.append({"file": file_name, "rows": stop-start, "start_utc": times[start].isoformat(), "end_utc": times[stop-1].isoformat()})
    metadata = {
        "seed": SEED, "total_rows": int(len(values)), "source_columns": int(len(names)),
        "selected_columns": {"climate": CLIMATE, "climate_plus_solar": CLIMATE_SOLAR, "expanded_with_room_light": EXPANDED},
        "continuous_selected_missing_values": 0, "missing_policy": "drop rows with missing selected values; do not impute",
        "full_data_pca_rows": int(len(values)), "factor_analysis_rows": int(len(fa_x)),
        "clustering_rows": int(len(cluster_sample)), "fa_n_factors_parallel_analysis": parallel["n_factors"],
        "cluster_selected_k": int(best_k), "cluster_silhouette": float(cluster_silhouette),
        "time_segments": time_rows, "temporal_gap_hours": (times[late_start]-times[early_end-1]).total_seconds()/3600,
        "pca_full_explained": {key: model["explained"].tolist() for key, (_,model) in models.items()},
        "temporal_reconstruction": split_rows,
        "component_physical_profiles": component_groups,
        "cluster_profiles": cluster_summary,
    }
    (RESULTS / "summary.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(f"Rows: {len(values)}; PCA rows: {len(values)}; FA rows: {len(fa_x)}; clustering rows: {len(cluster_sample)}")
    print(f"Time: {times[0]} to {times[-1]}; later holdout starts {times[late_start]} (gap {metadata['temporal_gap_hours']:.1f} h)")
    print(f"Parallel analysis retains {parallel['n_factors']} FA factors; k-means selects k={best_k}, silhouette={cluster_silhouette:.3f}")
    print("Full-data PCA explained variance (first PCs):")
    for key, (_, model) in models.items():
        n80 = int(np.searchsorted(np.cumsum(model["explained"]), 0.80) + 1)
        print(f"  {key}: PC1={model['explained'][0]:.3f}, PC1-PC2={model['explained'][:2].sum():.3f}, PCs for 80%={n80}")
    print("Temporal standardized reconstruction RMSE:")
    for row in split_rows:
        if row["components"] in (2, 3, 4):
            print(f"  {row['model']} k={row['components']}: early={row['train_standardized_rmse']:.3f}, later={row['later_standardized_rmse']:.3f}")


if __name__ == "__main__":
    main()
