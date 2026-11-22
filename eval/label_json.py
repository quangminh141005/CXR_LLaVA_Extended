"""Parse CXR-LLaVA label-prompt output (CheXpert JSON) into 14 labels: 1 / 0 / None.

The model may wrap the JSON in a report (gemini prompt), use single quotes or trailing
commas, or skip findings. Strategy:
  1. take the last {...} block that json-loads (after light cleanup) -> key/value pairs
  2. otherwise fall back to every  "Finding": 0|1  /  'Finding': 0|1  pattern in the text
Keys are matched case/space-insensitively to the 14 CheXpert names. A finding that is
absent or has a value other than 0/1 (or true/false) stays None -> blank (not scored,
counted as a parse failure). Nothing is guessed.
"""
import json
import re

FINDINGS = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Lesion", "Lung Opacity",
            "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
            "Pleural Other", "Fracture", "Support Devices"]


def _norm(s):
    return re.sub(r"[^a-z]", "", str(s).lower())


_BY_NORM = {_norm(f): f for f in FINDINGS}


def _value(v):
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)) and v in (0, 1):
        return int(v)
    if isinstance(v, str):
        return {"0": 0, "1": 1, "true": 1, "false": 0}.get(v.strip().lower())
    return None


def _from_json(text):
    for block in reversed(re.findall(r"\{[^{}]*\}", text, flags=re.S)):
        b = re.sub(r",\s*}", "}", block.replace("'", '"'))
        try:
            obj = json.loads(b)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def parse_labels(text):
    """-> (labels dict finding -> 1/0/None, method 'json' | 'regex' | 'none')."""
    labels = {f: None for f in FINDINGS}
    text = text or ""
    obj, method = _from_json(text), "json"
    if obj is None:
        obj = {m.group(1): m.group(2) for m in
               re.finditer(r"[\"']([A-Za-z ]+)[\"']\s*:\s*[\"']?(0|1|true|false)\b", text, flags=re.I)}
        method = "regex" if obj else "none"
    for k, v in obj.items():
        f = _BY_NORM.get(_norm(k))
        if f is not None:
            labels[f] = _value(v)
    return labels, method


if __name__ == "__main__":  # quick self-check
    t = 'Report: ... {"No Finding": 0, "Cardiomegaly": 1, \'Edema\': "0", "Pleural Effusion": 1,}'
    lab, m = parse_labels(t)
    assert m == "json" and lab["Cardiomegaly"] == 1 and lab["Edema"] == 0 and lab["Atelectasis"] is None, (lab, m)
    lab, m = parse_labels('"Cardiomegaly": 1, "Edema": 0 and then garbage {')
    assert m == "regex" and lab["Cardiomegaly"] == 1 and lab["Edema"] == 0, (lab, m)
    assert parse_labels("no json here")[1] == "none"
    print("ok")
