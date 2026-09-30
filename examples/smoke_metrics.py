"""Run the released selection metrics on synthetic eyes; no study data needed."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from optimize_protocol_aligned_gate import (  # noqa: E402
    coverage_by_group,
    operating_metrics,
    restricted_aurc,
)


def main() -> None:
    frame = pd.DataFrame(
        {
            "y_true": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            "y_pred": [0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
            "dr_grade": [0, 0, 0, 0, 4, 4, 4, 4, 4, 4],
            "selection_risk": [0.01, 0.02, 0.03, 0.04, 0.1, 0.2, 0.3, 0.4, 0.9, 1.0],
        }
    )
    metrics = operating_metrics(frame, "selection_risk", coverage=0.8)
    retention = coverage_by_group(frame, "selection_risk", "dr_grade", coverage=0.8)
    aurc = restricted_aurc(
        frame["y_true"].ne(frame["y_pred"]), frame["selection_risk"]
    )
    assert metrics["accepted_n"] == 8
    assert metrics["accepted_error_rate"] == 0.0
    assert np.isfinite(aurc)
    assert {row["group"]: row["coverage"] for row in retention} == {0: 1.0, 4: 4 / 6}
    print("Selection-metric smoke test passed (synthetic data).")


if __name__ == "__main__":
    main()
