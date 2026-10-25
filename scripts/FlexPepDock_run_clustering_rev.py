#!/usr/bin/env python3
"""FlexPepDock 결과에서 MD 후보 medoid를 선택합니다.

실행: 분석할 PDB 파일들이 모인 폴더에서
      conda activate cabs_cluster
      python FlexPepDock_run_clustering.py
필요 package: numpy, MDAnalysis, scikit-learn, scikit-learn-extra.

[주요 변경 사항]
TSV 파일 매칭에 의한 점수 연결 오류를 원천적으로 차단하기 위해,
Rosetta가 생성한 PDB 파일 하단에 기록된 score 데이터를 직접 파싱합니다.
따라서 별도의 TSV 파일이 필요하지 않으며, PDB 이름과 무관하게 정확한 점수를 추출합니다.

Clustering: k=10, random initialization 100회, alternate update.
수렴하고 k개 cluster가 모두 비어 있지 않은 결과 중 최소 inertia를 선택합니다.

출력: clustering_results_kmedoids_score.csv
      representatives_kmedoids/cluster_01_medoid.pdb ...
"""

import csv
import glob
import os
import shutil
import warnings

import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import align
from sklearn.exceptions import ConvergenceWarning
from sklearn_extra.cluster import KMedoids

# 0. 사용자 설정
INPUT_DIR = "."
align_sel = "segid A and protein and name CA and not name H*"
pep_rmsd_sel = "segid P and protein and backbone and not name H*"
k = 10
N_RESTARTS = 100
FIRST_SEED = 0
MAX_ITER = 300
OUT_CSV = "clustering_results_kmedoids_score.csv"
REP_SUBDIR = "representatives_kmedoids"

def parse_score_from_pdb(pdb_path):
    """PDB 파일 하단에 기록된 Rosetta score에서 reweighted_sc 값을 직접 추출합니다."""
    with open(pdb_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("reweighted_sc "):
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        return float(parts[1])
                    except ValueError:
                        break
    raise ValueError(
        f"{os.path.basename(pdb_path)}: PDB 내부에서 reweighted_sc 값을 찾을 수 없습니다. "
        "Rosetta 출력이 맞는지 확인하세요."
    )

def atom_signature(atoms, filename):
    """Atom 수 및 residue/atom 대응 순서를 확인합니다."""
    keys = [(a.segid, a.resid, a.icode, a.name) for a in atoms]
    if not keys or len(set(keys)) != len(keys):
        raise ValueError(f"{filename}: 빈 selection 또는 중복 atom입니다. alternate location을 확인하세요.")
    return [(a.segid, a.resid, a.icode, a.resname, a.name) for a in atoms]

def fit_restarts(dist_matrix):
    """N_RESTARTS회 중 유효한 최소 inertia의 fit을 선택합니다."""
    best_fit = None
    for seed in range(FIRST_SEED, FIRST_SEED + N_RESTARTS):
        candidate = KMedoids(
            n_clusters=k, metric="precomputed", method="alternate",
            init="random", random_state=seed, max_iter=MAX_ITER,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            candidate.fit(dist_matrix)
            
        for warning in caught:
            if (not issubclass(warning.category, ConvergenceWarning)
                    and "Cluster " not in str(warning.message)):
                warnings.warn(str(warning.message), warning.category)
                
        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
            continue
        if (len(np.unique(candidate.labels_)) != k
                or len(np.unique(candidate.medoid_indices_)) != k
                or not np.isfinite(candidate.inertia_)):
            continue
        if not np.array_equal(candidate.labels_[candidate.medoid_indices_], np.arange(k)):
            continue
            
        if best_fit is None or candidate.inertia_ < best_fit.inertia_:
            best_fit = candidate
            
    if best_fit is None:
        raise ValueError(
            "수렴한 비어 있지 않은 k개 cluster를 얻지 못했습니다. "
            "중복 구조, model 다양성 및 MAX_ITER를 확인하세요."
        )
    return best_fit

def main():
    input_dir = os.path.abspath(INPUT_DIR)
    pdb_files = sorted(glob.glob(os.path.join(input_dir, "*.pdb")))
    if not pdb_files:
        raise ValueError(f"'{input_dir}'에 PDB 파일이 없습니다.")
        
    for name, value in (("k", k), ("N_RESTARTS", N_RESTARTS), ("MAX_ITER", MAX_ITER)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name}은 양의 정수여야 합니다.")
    if not 2 <= k <= len(pdb_files):
        raise ValueError(f"2 <= k <= PDB 개수가 필요합니다: k={k}, PDB={len(pdb_files)}")

    # 1. PDB 파일 내부에서 직접 reweighted_sc 추출
    pdb_scores = {}
    print("[INFO] Parsing reweighted_sc directly from PDB files...")
    for p in pdb_files:
        name = os.path.basename(p)
        pdb_scores[name] = parse_score_from_pdb(p)
    print(f"[INFO] Successfully loaded scores for all {len(pdb_files)} models.")

    # 2. Atom 대응 확인 및 receptor 기준 공통 reference alignment
    universes = [mda.Universe(p) for p in pdb_files]
    signatures = {}
    for filename, u in zip(pdb_files, universes):
        if len(u.trajectory) != 1 or not np.isfinite(u.atoms.positions).all():
            raise ValueError(f"{filename}: 유효한 좌표의 단일 model PDB가 필요합니다.")
        for selection in (align_sel, pep_rmsd_sel):
            signature = atom_signature(u.select_atoms(selection), filename)
            if selection not in signatures:
                signatures[selection] = signature
            elif signature != signatures[selection]:
                raise ValueError(f"{filename}: atom identity/order가 reference와 다릅니다: {selection}")
        rec_xyz = u.select_atoms(align_sel).positions.astype(float)
        if len(rec_xyz) < 3 or np.linalg.matrix_rank(rec_xyz - rec_xyz.mean(axis=0)) < 2:
            raise ValueError(f"{filename}: alignment에는 일직선에 있지 않은 receptor Cα 3개 이상이 필요합니다.")
            
    ref = universes[0]
    print(f"[INFO] Aligning to {os.path.basename(pdb_files[0])}...")
    for u in universes[1:]:
        align.alignto(u, ref, select=align_sel, weights=None, strict=True)

    # 3. 모든 pair의 peptide backbone RMSD 계산
    print("[INFO] Calculating peptide backbone RMSD matrix...")
    pep_coords = np.array([u.select_atoms(pep_rmsd_sel).positions.copy() for u in universes], dtype=float)
    n_models, n_pep, _ = pep_coords.shape
    dist_matrix = np.zeros((n_models, n_models), dtype=float)
    for i in range(n_models):
        for j in range(i + 1, n_models):
            diff = pep_coords[i] - pep_coords[j]
            d = np.sqrt((diff**2).sum() / n_pep)
            dist_matrix[i, j] = dist_matrix[j, i] = d
            
    if not np.isfinite(dist_matrix).all():
        raise ValueError("RMSD matrix에 비정상 값이 있습니다. alignment와 좌표를 확인하세요.")

    # 4. K-medoids Clustering 수행
    print(f"[INFO] Running {N_RESTARTS} random initializations with k={k}...")
    fit = fit_restarts(dist_matrix)
    clusters = []
    for cid in range(k):
        members = np.flatnonzero(fit.labels_ == cid)
        medoid = int(fit.medoid_indices_[cid])
        distances = dist_matrix[members, medoid]
        clusters.append(dict(cid=cid, members=members, medoid=medoid,
                             size=len(members), mean=float(distances.mean()), max=float(distances.max())))
    clusters.sort(key=lambda row: (-row["size"], row["cid"]))

    # 5. Medoid PDB 복사 및 CSV 저장
    rep_dir = os.path.join(input_dir, REP_SUBDIR)
    os.makedirs(rep_dir, exist_ok=True)
    csv_path = os.path.join(input_dir, OUT_CSV)
    
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "cluster_rank_by_size", "cluster_id", "cluster_size", "pdb_file", "is_medoid",
            "reweighted_sc", "rmsd_to_medoid_A",
            "cluster_mean_rmsd_to_medoid_A", "cluster_max_rmsd_to_medoid_A",
        ])
        for rank, row in enumerate(clusters, start=1):
            medoid = row["medoid"]
            shutil.copy2(pdb_files[medoid], os.path.join(rep_dir, f"cluster_{rank:02d}_medoid.pdb"))
            for idx in row["members"]:
                name = os.path.basename(pdb_files[idx])
                writer.writerow([
                    rank, row["cid"], row["size"], name, int(idx == medoid),
                    f"{pdb_scores[name]:.3f}", f"{dist_matrix[idx, medoid]:.6f}",
                    f"{row['mean']:.6f}", f"{row['max']:.6f}",
                ])
            name = os.path.basename(pdb_files[medoid])
            print(f"Cluster {rank:02d} | Size: {row['size']:3d} | Medoid: {name} "
                  f"| reweighted_sc: {pdb_scores[name]:.3f}")
                  
    print(f"[COMPLETE] {csv_path}")
    print(f"[COMPLETE] {rep_dir} ({k} medoid PDBs)")

if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as exc:
        raise SystemExit(f"[ERROR] {exc}") from None