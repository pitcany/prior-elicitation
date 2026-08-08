"""Fetch and convert the risk-frequency-judgment datasets collated by
Pachur (2024), "The perception of dramatic risks: Biased media, but
unbiased minds", Cognition 246:105736, https://doi.org/10.1016/j.cognition.2024.105736

Source repository: https://osf.io/u4d7g  (Data and code / Frequency
Judgments and Media Coverage / Individual data sets). Each file holds,
per cause of death, the actual annual frequency and the aggregate
(geometric mean or median) judged frequency from one study:

  Lichtenstein1978   US,          41 causes, N = 74 participants
  Morgan1983         US,          41 causes
  Benjamin2001       US,          41 causes
  Hertwig2005        Germany,     37 causes, N = 45
  HakesViscusi2004   US,          22 causes
  Armantier2006      US, 3 treatments x 35 causes
  Pachur2012         Switzerland, 41 causes, N = 85
  LacourDavis2020    US, 2 experiments

Output: risk_judgments.csv with columns dataset, source_file, risk,
actual, estimate -- rows with missing or nonpositive values dropped.

Run: python fetch_convert.py   (requires pandas + openpyxl)
"""

from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

FILES = {
    "Lichtenstein1978": "https://osf.io/download/7zs5g/",
    "Hertwig2005": "https://osf.io/download/a3dvs/",
    "Pachur2012": "https://osf.io/download/4h3mx/",
    "Morgan1983": "https://osf.io/download/64xnh/",
    "Benjamin2001": "https://osf.io/download/x29c4/",
    "HakesViscusi2004": "https://osf.io/download/6565e8f679d42826e43e86a9/",
    "Armantier2006": "https://osf.io/download/fc4vz/",
    "LacourDavis2020": "https://osf.io/download/764dh/",
}


def main() -> None:
    rows = []
    for name, url in FILES.items():
        with urllib.request.urlopen(url) as resp:
            df = pd.read_excel(io.BytesIO(resp.read()))
        for idx, r in df.iterrows():
            study = str(df["Study"][idx]) if "Study" in df.columns else name
            if study == "7":  # HakesViscusi2004 stores a numeric study code
                study = name
            rows.append({"dataset": study, "source_file": name, "risk": r["Risk"],
                         "actual": r["ActualFrequencies"], "estimate": r["Estimates"]})
    out = pd.DataFrame(rows)
    out = out[np.isfinite(out["actual"]) & np.isfinite(out["estimate"])
              & (out["actual"] > 0) & (out["estimate"] > 0)]
    dest = Path(__file__).parent / "risk_judgments.csv"
    out.to_csv(dest, index=False)
    print(f"wrote {dest}: {len(out)} rows, {out.dataset.nunique()} datasets")


if __name__ == "__main__":
    main()
