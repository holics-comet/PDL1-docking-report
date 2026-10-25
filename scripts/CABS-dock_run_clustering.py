"""CABS-dock 결과에서 FlexPepDock에 사용할 representative model을 찾는 script.

실행 방법
---------
분석할 complex PDB만 있는 directory에서: python run_clustering.py
각 PDB에는 receptor와 peptide를 포함한 model이 하나씩 있어야 합니다.
현재 directory의 모든 *.pdb를 읽습니다. 기본 k=10에서는 최소 11개가 필요합니다.
Dependencies: MDAnalysis, numpy, scikit-learn, scikit-learn-extra.
실행 검증 환경: Python 3.11, MDAnalysis 2.10.0, numpy 1.26.4,
               scikit-learn 1.5.2, scikit-learn-extra 0.3.0, scipy 1.14.1.

계산 흐름
---------
1. 모든 model에서 receptor/peptide Cα의 residue 대응 관계를 확인합니다.
2. 첫 PDB의 receptor Cα에 모든 complex를 alignment합니다.
3. Peptide를 추가로 fitting하지 않고 pairwise peptide Cα RMSD를 계산합니다.
4. 동일한 RMSD matrix에서 k=10, random initialization을 100회 수행합니다.
   각 실행에서는 cluster assignment와 medoid update를 수렴할 때까지 반복합니다.
5. 수렴하고 10개 cluster가 모두 비어 있지 않은 실행 중 최소 inertia를 선택합니다.
   inertia = 각 model에서 소속 medoid까지의 RMSD 총합 (RMSD²의 합이 아님).
6. Cluster size 순으로 medoid와 RMSD/silhouette를 출력하고 contact를 집계합니다.

출력 파일: 같은 이름의 기존 파일은 정상 실행 시 덮어씁니다.
  cluster_members.txt   : 구성원, medoid, 각 model의 metrics, 100회 실행 기록
  cluster_summary.csv   : cluster별 size, medoid, 평균/최대 RMSD, 평균 silhouette
  contact_frequency.csv : 전체 입력 model에서 residue-pair contact가 나타난 빈도
Medoid는 입력 PDB 중 하나입니다. 별도 PDB를 생성하거나 refinement하지 않습니다.

해석 범위
---------
k=10은 representative 수의 설정이며, 실제 binding mode가 10개라는 증거는 아닙니다.
Silhouette/RMSD는 평가용이며 model을 자동 제외하거나 최종 실행을 선택하지 않습니다.
Cluster size와 contact frequency는 입력 model의 빈도이며 binding probability가 아닙니다.
이 script는 multi-start K-medoids이며, CABS-dock의 consensus 절차 재현을 주장하지 않습니다.
"""

import csv
import glob
import os
import warnings
from collections import Counter

import MDAnalysis as mda
from MDAnalysis.analysis import align, rms
from MDAnalysis.analysis.distances import distance_array
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import silhouette_samples
from sklearn_extra.cluster import KMedoids


# 1. 설정: 보통 사용자가 수정할 부분은 이 영역입니다.
# segid는 MDAnalysis의 segment identifier입니다. PDB의 chainID와 다를 수 있으므로
# 입력의 식별자를 확인하고, 필요하면 아래 네 selection을 함께 수정하십시오.
# name CA는 Cα를 선택합니다. not name H*는 이름이 H로 시작하는 atom만 제외하는
# 원본 contact 규칙입니다. Hydrogen이 있다면 1H처럼 다른 이름이 포함되는지 확인하십시오.
recep_sel_str = "segid A and not name H*"
pep_sel_str = "segid P and not name H*"
align_sel = "segid A and name CA"
pep_ca_sel = "segid P and name CA"
CUTOFF = 4.5  # Å; atom 간 contact 판정에만 사용하며 RMSD cutoff가 아닙니다.
k = 10  # Cluster 수 = 선택할 medoid 수
N_RESTARTS = 100  # 서로 다른 initial medoid로 시작하는 독립 clustering 횟수
FIRST_SEED = 0  # 기본 seed는 0~99; 같은 입력/설정/환경에서 재현 가능하게 합니다.
MAX_ITER = 300  # 각 실행 내부의 최대 update 횟수; N_RESTARTS와 다릅니다.


def ca_signature(atoms, filename, selection):
    """Cα 중복 여부를 확인하고, model 사이의 residue 대응을 비교할 목록을 만듭니다.

    Cα 개수가 같아도 residue가 다르면 올바른 RMSD가 아닙니다.
    icode는 42A/42B처럼 같은 residue number를 구분하는 insertion code입니다.
    """
    if len(atoms) == 0:
        raise ValueError(f"{filename}: empty selection: {selection}")
    keys = [(a.segid, a.resid, a.icode) for a in atoms]
    if len(set(keys)) != len(keys):
        raise ValueError(
            f"{filename}: duplicate CA residue identifiers in {selection}; "
            "check alternate locations and residue numbering."
        )
    return [(a.segid, a.resid, a.icode, a.resname, a.name) for a in atoms]


def fit_restarts(rmsd_matrix):
    """여러 random initialization 중 최소 inertia의 유효한 결과와 실행 기록을 반환합니다."""
    best_fit = None
    best_seed = None
    run_log = []
    for seed in range(FIRST_SEED, FIRST_SEED + N_RESTARTS):
        # precomputed: 좌표 대신 이미 계산한 RMSD matrix를 전달합니다.
        # alternate: 가까운 medoid에 배정 → cluster 내부 RMSD 합이 최소인
        # model로 medoid 갱신을 반복합니다. 원본의 기본 algorithm을 유지합니다.
        candidate = KMedoids(
            n_clusters=k, metric="precomputed", method="alternate",
            init="random", random_state=seed, max_iter=MAX_ITER,
        )
        # 최대 update 횟수에 도달해도 fit()은 결과를 반환할 수 있습니다.
        # 따라서 ConvergenceWarning을 확인하여 미수렴 결과를 선택에서 제외합니다.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            candidate.fit(rmsd_matrix)

        reasons = []
        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
            reasons.append("not_converged")
        if len(np.unique(candidate.labels_)) != k:
            reasons.append("empty_final_cluster")
        if len(np.unique(candidate.medoid_indices_)) != k:
            reasons.append("duplicate_medoids")
        if not np.isfinite(candidate.inertia_):
            reasons.append("nonfinite_inertia")
        status = ";".join(reasons) if reasons else "eligible"
        run_log.append({
            "seed": seed,
            "status": status,
            "inertia": float(candidate.inertia_),
            "warnings": " | ".join(str(w.message).replace("\n", " ") for w in caught),
        })
        # inertia는 각 model의 기여를 더한 값입니다. Cluster별 평균의 단순합이 아닙니다.
        # 정확히 같은 값이면 먼저 실행한 seed를 유지합니다. 전역 최적해를 보장하지는 않습니다.
        if not reasons and (best_fit is None or candidate.inertia_ < best_fit.inertia_):
            best_fit, best_seed = candidate, seed

    if best_fit is None:
        counts = Counter(row["status"] for row in run_log)
        raise RuntimeError(
            f"No eligible clustering among {N_RESTARTS} attempts: {dict(counts)}. "
            "Check duplicate poses and input diversity; increase MAX_ITER if needed. "
            "No new output files were written."
        )
    return best_fit, best_seed, run_log


def main():
    # 현재 working directory에서 읽습니다. Script의 저장 위치와는 다를 수 있습니다.
    # Filename 정렬로 model 순서와 alignment reference를 일정하게 유지합니다.
    pdb_files = sorted(glob.glob("*.pdb"))
    if not pdb_files:
        raise ValueError("PDB files were not found in the current directory.")
    n_models = len(pdb_files)
    for name, value in (("k", k), ("N_RESTARTS", N_RESTARTS), ("MAX_ITER", MAX_ITER)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if (isinstance(FIRST_SEED, bool) or not isinstance(FIRST_SEED, (int, np.integer))
            or FIRST_SEED < 0 or FIRST_SEED + N_RESTARTS - 1 > 2**32 - 1):
        raise ValueError("All random seeds must be integers between 0 and 2**32 - 1.")
    # Silhouette는 cluster가 1개뿐이거나 모든 model이 singleton이면 정의할 수 없습니다.
    if not 2 <= k < n_models:
        raise ValueError(f"Require 2 <= k < number of models; k={k}, models={n_models}.")
    print(f"Loading {n_models} PDB files...")
    u_list = [mda.Universe(f) for f in pdb_files]
    reference_signatures = {}
    for filename, u in zip(pdb_files, u_list):
        if len(u.trajectory) != 1:
            raise ValueError(f"{filename}: expected one structure per PDB file.")
        if not np.isfinite(u.atoms.positions).all():
            raise ValueError(f"{filename}: nonfinite coordinates.")
        for selection in (recep_sel_str, pep_sel_str):
            if len(u.select_atoms(selection)) == 0:
                raise ValueError(f"{filename}: empty selection: {selection}")
        for selection in (align_sel, pep_ca_sel):
            atoms = u.select_atoms(selection)
            signature = ca_signature(atoms, filename, selection)
            if selection not in reference_signatures:
                reference_signatures[selection] = signature
            elif signature != reference_signatures[selection]:
                raise ValueError(
                    f"{filename}: CA identities/order do not match {pdb_files[0]} "
                    f"for {selection}. Check missing residues and numbering."
                )
        rec_ca = u.select_atoms(align_sel)
        xyz = rec_ca.positions.astype(float)
        if len(rec_ca) < 3 or np.linalg.matrix_rank(xyz - xyz.mean(axis=0)) < 2:
            raise ValueError(f"{filename}: receptor alignment requires >=3 noncollinear CAs.")
        if not np.isfinite(rec_ca.masses).all() or np.any(rec_ca.masses <= 0):
            raise ValueError(f"{filename}: invalid receptor CA masses for alignment.")

    # 2. Receptor alignment
    # Receptor에서 구한 rotation/translation을 peptide를 포함한 complex 전체에 적용합니다.
    # 모든 model을 같은 reference에 맞춥니다. Pair마다 receptor를 다시 fitting하는 방식은 아닙니다.
    # Flexible receptor 영역이 alignment에 영향을 준다면 공통 stable core를 align_sel로 지정합니다.
    ref = u_list[0]
    print(f"Aligning structures to {pdb_files[0]}...")
    for mobile in u_list[1:]:
        align.alignto(mobile, ref, select=align_sel, weights="mass", strict=True)

    # 3. Pairwise peptide Cα RMSD
    # D[i,j] = sqrt(sum over peptide Cα atoms of |r_i - r_j|² / number of Cα atoms).
    # Peptide를 따로 fitting하면 receptor 위의 위치와 방향 차이가 사라지므로 실시하지 않습니다.
    print("Calculating RMSD matrix...")
    pep_positions = [u.select_atoms(pep_ca_sel).positions.copy() for u in u_list]
    rmsd_matrix = np.zeros((n_models, n_models), dtype=float)
    for i in range(n_models):
        for j in range(i + 1, n_models):
            r = rms.rmsd(pep_positions[i], pep_positions[j],
                         center=False, superposition=False)
            rmsd_matrix[i, j] = rmsd_matrix[j, i] = r
    if not np.isfinite(rmsd_matrix).all() or np.any(rmsd_matrix < 0):
        raise ValueError("Invalid RMSD matrix; check coordinates and receptor alignment.")
    # 위의 대입으로 matrix의 symmetry를 유지하며, 동일 model 사이 거리인 diagonal은 0입니다.

    print(f"Running {N_RESTARTS} random initializations with k={k}...")
    kmedoids, selected_seed, run_log = fit_restarts(rmsd_matrix)
    # labels[i]는 i번째 PDB가 속한 cluster ID(0~k-1)입니다.
    # medoid_indices[c]는 cluster c의 medoid가 pdb_files에서 몇 번째인지 나타냅니다.
    # 출력의 Rank(1부터 시작)는 size 정렬 결과이며, cluster ID와 구분합니다.
    labels = kmedoids.labels_
    medoid_indices = kmedoids.medoid_indices_
    # Silhouette: s=(b-a)/max(a,b). a는 같은 cluster의 다른 model과의 평균 RMSD,
    # b는 다른 cluster별로 구한 평균 RMSD 중 최솟값입니다. Medoid까지의 거리가 아닙니다.
    # 1에 가까우면 잘 분리되고, 0 부근은 경계가 모호하며, 음수이면 다른 cluster와 평균적으로 더 가깝습니다.
    # Singleton은 scikit-learn의 규약에 따라 0입니다. 자동적인 합격/탈락 판정에 사용하지 않습니다.
    silhouettes = silhouette_samples(rmsd_matrix, labels, metric="precomputed")
    eligible_count = sum(row["status"] == "eligible" for row in run_log)

    # Mean/maximum RMSD는 각 member에서 medoid까지의 거리로 계산합니다.
    # Cluster 내부의 모든 pair 사이 RMSD가 아니며, 평균에는 medoid 자신의 0도 포함합니다.
    cluster_info = []
    for cluster_id in range(k):
        member_indices = np.flatnonzero(labels == cluster_id)
        medoid_index = int(medoid_indices[cluster_id])
        distances = rmsd_matrix[member_indices, medoid_index]
        cluster_info.append({
            "cid": cluster_id, "size": len(member_indices),
            "medoid": pdb_files[medoid_index], "indices": member_indices,
            "mean_rmsd": float(distances.mean()),
            "max_rmsd": float(distances.max()),
            "mean_silhouette": float(silhouettes[member_indices].mean()),
        })
    # 원본과 같은 size 내림차순입니다. Size가 같으면 cluster ID순이며 quality 차이를 뜻하지 않습니다.
    cluster_info.sort(key=lambda row: row["size"], reverse=True)

    # 4. 결과 저장: 상세 내용과 실행 기록은 TXT, cluster별 요약은 CSV로 저장합니다.
    with open("cluster_members.txt", "w", encoding="utf-8") as f:
        f.write("=== Cluster Member List ===\n")
        f.write(f"Models: {n_models}; k: {k}; reference: {pdb_files[0]}\n")
        f.write(f"Alignment selection: {align_sel}; peptide selection: {pep_ca_sel}\n")
        f.write("Distance: common-reference receptor-aligned peptide CA RMSD (Angstrom); no peptide fitting.\n")
        f.write(f"KMedoids: method=alternate; init=random; max_iter={MAX_ITER}\n")
        f.write(f"Random seeds: {FIRST_SEED}..{FIRST_SEED + N_RESTARTS - 1}; eligible fits: {eligible_count}/{N_RESTARTS}\n")
        f.write(f"Selected seed: {selected_seed}; minimum inertia (Angstrom): {kmedoids.inertia_:.8f}\n")
        f.write(f"Overall mean RMSD to assigned medoid (Angstrom): {kmedoids.inertia_ / n_models:.6f}\n")
        f.write(f"Overall mean silhouette (model-weighted): {silhouettes.mean():.6f}\n")
        f.write("Selection: minimum inertia among eligible fits; exact ties use the lowest seed.\n")
        f.write("Ranking: size descending. Diagnostics do not select or exclude models.\n")
        f.write("Mean/max RMSD are distances to the medoid, NOT all-pairs cluster distances.\n")
        f.write("Mean includes the medoid's zero distance. Singleton RMSD=0 and silhouette=0 by convention; this is not evidence of quality.\n")
        for rank, row in enumerate(cluster_info, 1):
            f.write(f"\nCluster Rank {rank} (Original ID: {row['cid']})\n")
            f.write(f"  - Size: {row['size']} models\n")
            f.write(f"  - Representative (Medoid): {row['medoid']}\n")
            f.write(f"  - Mean RMSD to medoid (Angstrom): {row['mean_rmsd']:.6f}\n")
            f.write(f"  - Maximum RMSD to medoid (Angstrom): {row['max_rmsd']:.6f}\n")
            f.write(f"  - Mean silhouette: {row['mean_silhouette']:.6f}\n")
            f.write("  - Members (RMSD to medoid in Angstrom; silhouette):\n")
            for idx in row["indices"]:
                d = rmsd_matrix[idx, medoid_indices[row["cid"]]]
                f.write(f"      {pdb_files[idx]}\t{d:.6f}\t{silhouettes[idx]:.6f}\n")
        f.write("\n=== Random Initialization Audit ===\n")
        f.write("Seed\tStatus\tInertia_Angstrom\tWarnings\n")
        for row in run_log:
            f.write(f"{row['seed']}\t{row['status']}\t{row['inertia']:.8f}\t{row['warnings']}\n")

    with open("cluster_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Rank", "Cluster_ID", "Size", "Medoid",
                         "Mean_RMSD_to_Medoid_Angstrom", "Max_RMSD_to_Medoid_Angstrom",
                         "Mean_Silhouette"])
        for rank, row in enumerate(cluster_info, 1):
            writer.writerow([rank, row["cid"], row["size"], row["medoid"],
                             f"{row['mean_rmsd']:.6f}", f"{row['max_rmsd']:.6f}",
                             f"{row['mean_silhouette']:.6f}"])

    # 5. Global contact frequency: clustering과 독립적인 보조 분석입니다.
    # Receptor-peptide의 atom 사이 거리가 CUTOFF 미만이면 해당 residue pair를 contact로 판정합니다.
    # 한 model에서 여러 atom pair가 contact하더라도 set으로 묶어 residue pair별로 한 번만 셉니다.
    # Frequency (%) = contact가 나타난 model 수 / 전체 입력 model 수 × 100.
    print("Analyzing Global Contact Frequencies...")
    contact_counter = Counter()
    for u in u_list:
        rec_atoms = u.select_atoms(recep_sel_str)
        pep_atoms = u.select_atoms(pep_sel_str)
        dists = distance_array(rec_atoms.positions, pep_atoms.positions)
        rec_indices, pep_indices = np.where(dists < CUTOFF)
        unique_pairs = set()
        for r_idx, p_idx in zip(rec_indices, pep_indices):
            # Insertion code가 있는 residue도 별도로 기록합니다. 일반적인 번호 표기는 유지합니다.
            r_atom, p_atom = rec_atoms[r_idx], pep_atoms[p_idx]
            r_res = f"{r_atom.resname}{r_atom.resid}{r_atom.icode.strip()}"
            p_res = f"{p_atom.resname}{p_atom.resid}{p_atom.icode.strip()}"
            unique_pairs.add((r_res, p_res))
        contact_counter.update(unique_pairs)
    with open("contact_frequency.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Rank", "Receptor_Residue", "Peptide_Residue",
                         "Frequency_Count", "Frequency_Percent(%)"])
        # Frequency가 같으면 residue 표기의 사전순으로 정렬하여 재실행 시에도 행 순서를 유지합니다.
        sorted_contacts = sorted(contact_counter.items(), key=lambda item: (-item[1], item[0]))
        for i, ((r_res, p_res), count) in enumerate(sorted_contacts):
            writer.writerow([i + 1, r_res, p_res, count, f"{count / n_models * 100:.1f}"])

    print(f"Complete: selected seed={selected_seed}, inertia={kmedoids.inertia_:.6f} Angstrom, "
          f"eligible fits={eligible_count}/{N_RESTARTS}.")
    for filename in ("cluster_members.txt", "cluster_summary.csv", "contact_frequency.csv"):
        print(os.path.abspath(filename))


if __name__ == "__main__":
    # python run_clustering.py로 실행했을 때만 main()을 호출합니다.
    main()
