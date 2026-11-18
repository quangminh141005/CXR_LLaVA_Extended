"""Step 1.6.3: check that our CheXpert-Labeler install reproduces the official MIMIC labels.

The official mimic-cxr-2.0.0-chexpert.csv was made by running CheXpert-Labeler on one
section of each original report (impression > findings > last paragraph > comparison).
We extract the same section with the official MIMIC-CXR section parser
(eval/mimic_section_parser.py), label it with our install, and compare.
No model / GPU involved: this checks the labeler, not CXR-LLaVA.

Two sub-commands (cxreval env), with the labeler run in between (chexpert-label env):

  prepare  sample --n studies (seed 0) from the 3000 MIMIC test studies, read the original
           reports, extract the section -> reports.csv (headerless, 1 report per row)
           + keys.csv (row -> study_id, section used)
  compare  read the labeler output + official CSV -> agreement per label, official-vs-ours
           tables, disagreements.txt (with the section text, to read why)

Usage (on the server, repo root):
    python -u eval/validate_labeler.py prepare
    # in the labeler folder, env chexpert-label (the exact command is printed by prepare):
    python label.py --reports_path <out>/reports.csv --output_path <out>/labeled.csv
    python -u eval/validate_labeler.py compare 2>&1 | tee log/labeler_validation/compare.log

Output in log/labeler_validation/ (gitignored: contains MIMIC report text, PhysioNet DUA).
"""
import argparse
import csv
import os
import re
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "eval"))
from check_data import DEFAULTS  # noqa: E402  (same paths as Step 0)
import mimic_section_parser as sp  # noqa: E402

MIMIC_LABELS = ("/storage/shared/MIMIC-CXR-JPG/physionet.org/files/mimic-cxr-jpg/2.1.0/"
                "mimic-cxr-2.0.0-chexpert.csv.gz")
DEFAULT_OUT = os.path.join(REPO, "log", "labeler_validation")
DEFAULT_LABELER_DIR = "/storage/student4/nqminh/chexpert-labeler"

LABELS = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Lesion", "Lung Opacity",
          "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
          "Pleural Other", "Fracture", "Support Devices"]
MIMIC_FINDINGS = ["Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion", "Pneumonia", "Pneumothorax"]
VALUES = ["1", "0", "-1", "blank"]


# ------------------------------------------------------------------ section extraction
def list_rindex(lst, s):
    """Index of the *last* element equal to s (as in the official create_section_files.py)."""
    return len(lst) - lst[::-1].index(s) - 1


def extract_section(study, text):
    """Same rules as MIT-LCP create_section_files.py -> (section_text, section_name)."""
    custom_names, custom_idx = sp.custom_mimic_cxr_rules()
    if study in custom_idx:
        a, b = custom_idx[study]
        return text[a:b], "custom_indices"
    sections, names, _ = sp.section_text(text)
    if study in custom_names:
        sn = custom_names[study]
        return sections[list_rindex(names, sn)].strip(), f"custom:{sn}"
    for sn in ("impression", "findings", "last_paragraph", "comparison"):
        if sn in names:
            return sections[list_rindex(names, sn)].strip(), sn
    return "", "none"


# ------------------------------------------------------------------ prepare
def test_studies():
    """(study_id, report_path) for the 3000 MIMIC test images (1 image per study)."""
    rows = pd.read_csv(DEFAULTS["mimic_csv"], header=None)[0]
    out = []
    for r in rows:
        rel = r.strip().replace("\\", "/")
        rel = rel[len("files/"):] if rel.startswith("files/") else rel
        m = re.search(r"(p\d+/p\d+)/s(\d+)/", rel)
        out.append((int(m.group(2)), os.path.join(DEFAULTS["mimic_report_root"], m.group(1), f"s{m.group(2)}.txt")))
    return pd.DataFrame(out, columns=["study_id", "report_path"]).drop_duplicates("study_id")


def cmd_prepare(args):
    pool = test_studies()
    sel = pool.sample(min(args.n, len(pool)), random_state=args.seed).reset_index(drop=True)
    print(f"test studies: {len(pool)}, sampled {len(sel)} (seed {args.seed})")

    keys, texts, skipped = [], [], []
    for _, r in sel.iterrows():
        try:
            with open(r["report_path"]) as f:
                text = f.read()
        except OSError as e:
            skipped.append((r["study_id"], f"unreadable: {e}"))
            continue
        section, name = extract_section(f"s{r['study_id']}", text)
        if not section.strip():
            skipped.append((r["study_id"], f"empty section ({name})"))
            continue
        keys.append({"row": len(texts), "study_id": r["study_id"], "section": name})
        texts.append(section)

    # headerless, one report per row; csv module quotes fields containing newlines/commas
    rep_path = os.path.join(args.out, "reports.csv")
    with open(rep_path, "w", newline="") as f:
        w = csv.writer(f)
        for t in texts:
            w.writerow([t])
    pd.DataFrame(keys).to_csv(os.path.join(args.out, "keys.csv"), index=False)

    print(f"written {len(texts)} reports -> {rep_path}")
    print("section used:", pd.DataFrame(keys)["section"].value_counts().to_dict())
    for s, why in skipped:
        print(f"  skipped s{s}: {why}")
    lab_out = os.path.join(args.out, "labeled.csv")
    print(f"\nnext, in the labeler env:\n  conda activate chexpert-label\n  cd {args.labeler_dir}\n"
          f"  python label.py --reports_path {rep_path} --output_path {lab_out}")


# ------------------------------------------------------------------ compare
def as_str(v):
    """1.0 / 0.0 / -1.0 / NaN -> '1' / '0' / '-1' / 'blank'."""
    if pd.isna(v):
        return "blank"
    return str(int(float(v)))


def cmd_compare(args):
    keys = pd.read_csv(os.path.join(args.out, "keys.csv"))
    ours = pd.read_csv(os.path.join(args.out, "labeled.csv"))
    texts = pd.read_csv(os.path.join(args.out, "reports.csv"), header=None)[0]
    assert len(ours) == len(keys) == len(texts), \
        f"row count mismatch: labeled {len(ours)}, keys {len(keys)}, reports {len(texts)}"
    ours["study_id"] = keys["study_id"].values
    ours["text"] = texts.values

    off = pd.read_csv(MIMIC_LABELS)
    off = off.drop_duplicates("study_id").set_index("study_id")
    missing = set(ours["study_id"]) - set(off.index)
    assert not missing, f"{len(missing)} studies not in official CSV: {sorted(missing)[:5]}"

    rows, disagree = [], []
    for lab in LABELS:
        o = [as_str(v) for v in off.loc[ours["study_id"], lab]]
        u = [as_str(v) for v in ours[lab]]
        agree = sum(a == b for a, b in zip(o, u))
        rows.append({"label": lab, "mimic_finding": lab in MIMIC_FINDINGS, "n": len(o),
                     "agree": agree, "agreement_%": round(100 * agree / len(o), 1)})
        for sid, txt, a, b in zip(ours["study_id"], ours["text"], o, u):
            if a != b:
                disagree.append((lab, sid, a, b, txt))
        if lab in MIMIC_FINDINGS:
            tab = pd.crosstab(pd.Categorical(o, VALUES), pd.Categorical(u, VALUES),
                              rownames=["official"], colnames=["ours"], dropna=False)
            print(f"\n-- {lab}: {agree}/{len(o)} agree\n{tab.to_string()}")

    summ = pd.DataFrame(rows).sort_values(["mimic_finding", "label"], ascending=[False, True])
    summ.to_csv(os.path.join(args.out, "agreement.csv"), index=False)
    print("\n== Agreement per label (exact match of 1 / 0 / -1 / blank) ==")
    print(summ.to_string(index=False))
    m = summ[summ["mimic_finding"]]
    print(f"\n6 MIMIC findings: {m['agree'].sum()}/{m['n'].sum()} = {100 * m['agree'].sum() / m['n'].sum():.1f}%"
          f"   all 14: {summ['agree'].sum()}/{summ['n'].sum()} = {100 * summ['agree'].sum() / summ['n'].sum():.1f}%")
    n_reports = len({d[1] for d in disagree})
    print(f"reports with >=1 disagreement: {n_reports}/{len(ours)}")

    dis_path = os.path.join(args.out, "disagreements.txt")
    with open(dis_path, "w") as f:
        for lab, sid, a, b, txt in sorted(disagree, key=lambda d: (d[0] not in MIMIC_FINDINGS, d[0], d[1])):
            f.write(f"=== s{sid}  {lab}: official={a}  ours={b}\n{txt}\n\n")
    print(f"\nwritten: agreement.csv, disagreements.txt ({len(disagree)} label disagreements) in {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=DEFAULT_OUT)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--labeler_dir", default=DEFAULT_LABELER_DIR, help="only used to print the next command")
    sub.add_parser("compare")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    {"prepare": cmd_prepare, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    main()
