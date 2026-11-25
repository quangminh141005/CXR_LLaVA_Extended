"""VinDr-CXR shared definitions for the extension experiments (Exp 1 + Exp 2).

One place for: file paths, VinDr label -> CheXpert-Labeler mapping (Exp 1 rows), yes/no
questions (Exp 2), and exact box geometry (union area, part kept by the model's center crop).

VinDr label columns are prefixed "vd_" in manifests/vindr.csv, because several VinDr names
(Atelectasis, Cardiomegaly, ...) collide with the 14 CheXpert columns that hold the mapped GT.
"""
import os

VINDR_ROOT = "/storage/shared/VinDr-CXR"
VINDR_PNG_DIR = os.path.join(VINDR_ROOT, "processed_images", "png")  # verified = DICOM conversion (Q5)
VINDR_LABELS_TEST = os.path.join(VINDR_ROOT, "annotations", "image_labels_test.csv")
VINDR_BOXES_TEST = os.path.join(VINDR_ROOT, "annotations", "annotations_test.csv")
VINDR_LABELS_TRAIN = os.path.join(VINDR_ROOT, "annotations", "image_labels_train.csv")

NON_LABEL_COLS = {"image_id", "rad_id"}


def vd(name):
    """manifest column of a VinDr label"""
    return "vd_" + name


# ---------------------------------------------------------------- Exp 1: report-scorable rows
# row name -> (VinDr labels OR-ed into the row GT, CheXpert-Labeler class that scores it)
# Only VinDr findings with a CheXpert-Labeler class (Q3: unmapped findings are not scored from reports).
EXP1_ROWS = {
    "Aortic enlargement": (["Aortic enlargement"], "Enlarged Cardiomediastinum"),  # approximate mapping
    "Atelectasis": (["Atelectasis"], "Atelectasis"),
    "Cardiomegaly": (["Cardiomegaly"], "Cardiomegaly"),
    "Consolidation": (["Consolidation"], "Consolidation"),
    "Edema": (["Edema"], "Edema"),
    "Lung Opacity": (["Lung Opacity"], "Lung Opacity"),
    "Nodule/Mass": (["Nodule/Mass"], "Lung Lesion"),
    "Lung tumor": (["Lung tumor"], "Lung Lesion"),
    "Pleural effusion": (["Pleural effusion"], "Pleural Effusion"),
    "Pleural thickening": (["Pleural thickening"], "Pleural Other"),
    "Pneumothorax": (["Pneumothorax"], "Pneumothorax"),
    "Pneumonia": (["Pneumonia"], "Pneumonia"),
    "Rib/Clavicle fracture": (["Rib fracture", "Clavicle fracture"], "Fracture"),
}
# boxes are drawn around the heart / aorta, not a lesion -> left out of the pooled size analysis
NOT_LESIONS = {"Cardiomegaly", "Aortic enlargement"}

# mapped CheXpert GT columns written to the manifest: CheXpert class -> VinDr labels OR-ed
CHEXPERT_FROM_VINDR = {}
for _row, (_vlabs, _cx) in EXP1_ROWS.items():
    CHEXPERT_FROM_VINDR.setdefault(_cx, [])
    CHEXPERT_FROM_VINDR[_cx] += [v for v in _vlabs if v not in CHEXPERT_FROM_VINDR[_cx]]
CHEXPERT_FROM_VINDR["No Finding"] = ["No finding"]  # inverted logic not needed: VinDr has the label itself

# ---------------------------------------------------------------- Exp 2: yes/no questions
QUESTION = "Is there {} in this chest X-ray? Answer yes or no."
# VinDr label -> phrase inserted in QUESTION ("Other lesion", "Other disease(s)", "No finding" not asked)
YESNO_FINDINGS = {
    "Aortic enlargement": "aortic enlargement",
    "Atelectasis": "atelectasis",
    "Calcification": "calcification",
    "Cardiomegaly": "cardiomegaly",
    "Clavicle fracture": "a clavicle fracture",
    "Consolidation": "consolidation",
    "Edema": "pulmonary edema",
    "Emphysema": "emphysema",
    "Enlarged PA": "an enlarged pulmonary artery",
    "ILD": "interstitial lung disease",
    "Infiltration": "infiltration",
    "Lung Opacity": "lung opacity",
    "Lung cavity": "a lung cavity",
    "Lung cyst": "a lung cyst",
    "Mediastinal shift": "mediastinal shift",
    "Nodule/Mass": "a nodule or mass",
    "Pleural effusion": "pleural effusion",
    "Pleural thickening": "pleural thickening",
    "Pneumothorax": "pneumothorax",
    "Pulmonary fibrosis": "pulmonary fibrosis",
    "Rib fracture": "a rib fracture",
    "COPD": "chronic obstructive pulmonary disease",
    "Lung tumor": "a lung tumor",
    "Pneumonia": "pneumonia",
    "Tuberculosis": "tuberculosis",
}
# Priming: guided prompt (content) and a length-matched neutral preamble (control for "more text")
GUIDED = {
    "Pneumothorax": "Look carefully at the lung apices and pleural edges for a thin visceral pleural line "
                    "with no lung markings beyond it.",
    "Nodule/Mass": "Look carefully at both lungs, including the apices, the hila and the areas behind the "
                   "heart and diaphragm, for any round or oval opacity.",
}
CONTROL = {
    "Pneumothorax": "Take your time and review this whole chest X-ray image once more before you answer the "
                    "question below.",
    "Nodule/Mass": "Take your time and review this whole chest X-ray image once more before you answer the "
                   "question below, please.",
}


def slug(label):
    return "".join(c if c.isalnum() else "_" for c in label.lower()).strip("_")


def questions():
    """qid -> question text. qid = <prompt>__<slug(VinDr label)>, prompt in plain / guided / control."""
    q = {f"plain__{slug(f)}": QUESTION.format(p) for f, p in YESNO_FINDINGS.items()}
    for f, pre in GUIDED.items():
        q[f"guided__{slug(f)}"] = pre + " " + QUESTION.format(YESNO_FINDINGS[f])
        q[f"control__{slug(f)}"] = CONTROL[f] + " " + QUESTION.format(YESNO_FINDINGS[f])
    return q


# ---------------------------------------------------------------- box geometry (exact, no raster)
def union_area(boxes):
    """Exact area of the union of axis-aligned boxes [(x0, y0, x1, y1), ...] (coordinate compression)."""
    boxes = [b for b in boxes if b[2] > b[0] and b[3] > b[1]]
    if not boxes:
        return 0.0
    xs = sorted({b[0] for b in boxes} | {b[2] for b in boxes})
    area = 0.0
    for xa, xb in zip(xs[:-1], xs[1:]):
        ivs = sorted((b[1], b[3]) for b in boxes if b[0] <= xa and b[2] >= xb)
        covered, cur_s, cur_e = 0.0, None, None
        for s, e in ivs:
            if cur_e is None or s > cur_e:
                if cur_e is not None:
                    covered += cur_e - cur_s
                cur_s, cur_e = s, e
            else:
                cur_e = max(cur_e, e)
        if cur_e is not None:
            covered += cur_e - cur_s
        area += covered * (xb - xa)
    return area


def clip(boxes, region):
    x0, y0, x1, y1 = region
    return [(max(b[0], x0), max(b[1], y0), min(b[2], x1), min(b[3], y1)) for b in boxes]


def crop_box(w, h):
    """Region kept by the model's preprocessing: resize shortest edge + center crop (square of side min(w, h)).
    Same as eval/check_preprocessing.py crop_box."""
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    return left, top, left + s, top + s


SIZE_BINS = [(0.0, 0.01, "<1%"), (0.01, 0.05, "1-5%"), (0.05, 1.01, ">5%")]


def size_bin(frac):
    for lo, hi, name in SIZE_BINS:
        if lo <= frac < hi:
            return name
    return None


if __name__ == "__main__":  # tiny self-test of the geometry
    assert union_area([(0, 0, 2, 2), (1, 1, 3, 3)]) == 7
    assert union_area([(0, 0, 1, 1), (0, 0, 1, 1)]) == 1
    assert union_area([(0, 0, 4, 1), (1, 0, 2, 5)]) == 8
    assert union_area(clip([(0, 0, 10, 10)], crop_box(10, 20))) == 50  # crop (0,5,10,15)
    qs = questions()
    print(f"geometry ok; {len(qs)} questions, e.g.\n  " + "\n  ".join(f"{k}: {v}" for k, v in list(qs.items())[-4:]))
