"""
EXPERIMENT 3 -- gate-count and parameter-inheritance scaling.

Pure arithmetic from the manuscript's Eq. 8 plus the edge-list construction,
cross-checked against the actual edge lists the implementation builds so the
closed form and the code cannot silently disagree.

    |K_tot| = L*(n' - n + 1) + (L' - L)*n'

Reported per growth step:
  added_cry           two-qubit CRY gates introduced by this step (inert at
                      inheritance)
  inherited_cry       connecting angles carried over from all earlier stages
  total_rotations     3 * n' * L'   (rotation parameters in the child)
  inherited_rotations 3 * n  * L    (rotation parameters copied from parent)
  inherited_fraction  (inherited_rot + inherited_cry) / (total_rot + total_cry)
"""
import csv
import os
import sys
from pathlib import Path

WIGAN = Path(os.path.expanduser("~/qgan_project/wigan_project"))
sys.path.insert(0, str(WIGAN / "models"))
from inheritance_chained import build_chained_edge_list  # noqa: E402

OUT = Path(os.path.expanduser("~/host_share/qgan_results/quantum_paper/tables"))
SCHEDULE = [(4, 3), (6, 5), (8, 7), (12, 9), (16, 11)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    print("[exp3] verifying Eq. 8 against the implementation's own edge lists\n", flush=True)

    for i in range(1, len(SCHEDULE)):
        n_p, L_p = SCHEDULE[i - 1]
        n_c, L_c = SCHEDULE[i]
        label = f"{n_p}->{n_c}"

        parent_sched = SCHEDULE[:i]
        child_sched = SCHEDULE[:i + 1]
        parent_edges = build_chained_edge_list(parent_sched) if i > 1 else []
        child_edges = build_chained_edge_list(child_sched)

        inherited_cry = len(parent_edges)
        total_cry = len(child_edges)
        added_cry = total_cry - inherited_cry

        # closed form of Eq. 8 for THIS step's contribution
        eq8 = L_p * (n_c - n_p + 1) + (L_c - L_p) * n_c
        eq8_matches = (eq8 == added_cry)

        total_rot = 3 * n_c * L_c
        inherited_rot = 3 * n_p * L_p
        total_params = total_rot + total_cry
        inherited_params = inherited_rot + inherited_cry
        frac = inherited_params / total_params

        print(f"n={n_p}->n'={n_c}: added_CRY={added_cry}, inherited_CRY={inherited_cry}, "
              f"total_rot={total_rot}, inherited_rot={inherited_rot}, "
              f"inherited_fraction={frac:.2f}", flush=True)
        print(f"           Eq.8 = {eq8}  (matches edge list: {eq8_matches}); "
              f"total CRY in child = {total_cry}; total params = {total_params}",
              flush=True)

        rows.append({"growth_step": label, "n_parent": n_p, "n_child": n_c,
                     "L_parent": L_p, "L_child": L_c,
                     "added_cry": added_cry, "inherited_cry": inherited_cry,
                     "total_cry_in_child": total_cry,
                     "eq8_predicted_added_cry": eq8,
                     "eq8_matches_edge_list": eq8_matches,
                     "total_rotations": total_rot,
                     "inherited_rotations": inherited_rot,
                     "total_parameters": total_params,
                     "inherited_parameters": inherited_params,
                     "inherited_fraction": round(frac, 4)})

    with open(OUT / "table5_scaling.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n[exp3] wrote {OUT / 'table5_scaling.csv'}", flush=True)
    bad = [r["growth_step"] for r in rows if not r["eq8_matches_edge_list"]]
    print(f"[exp3] Eq.8 agrees with the implementation at every step: {not bad}"
          + (f"  (mismatches: {bad})" if bad else ""), flush=True)


if __name__ == "__main__":
    main()
