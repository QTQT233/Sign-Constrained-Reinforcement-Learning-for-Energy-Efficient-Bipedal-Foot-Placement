import ast
import csv
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "four_link" / "selector" / "paper_four_link_passive_sign_selector_v22_3_grid.py"
DATASET = ROOT / "data" / "four_link" / "selector" / "paper_four_link_passive_sign_selector_v22_3_grid_dataset.csv"


def source_constants() -> dict:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    result = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                try:
                    result[target.id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
    return result


class SelectorLabelProtocolTest(unittest.TestCase):
    def test_source_declares_archived_valid_cmt_filter(self) -> None:
        constants = source_constants()
        self.assertEqual(constants["MIN_NONZERO_TORQUE_STEPS"], 1)
        self.assertEqual(constants["MIN_COM_DISPLACEMENT_M"], 0.006)
        self.assertIs(constants["REQUIRE_NONZERO_TORQUE_FOR_CMT"], True)
        self.assertIs(constants["REQUIRE_MIN_COM_DISPLACEMENT_FOR_CMT"], True)
        self.assertEqual(constants["CMT_TIE_MARGIN"], 0.01)

    def test_archived_label_counts(self) -> None:
        counts = Counter()
        with DATASET.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            if row["label"] == "":
                continue
            counts[(row["split"], row["label"])] += 1

        self.assertEqual(len(rows), 75_600)
        self.assertEqual(counts[("train", "0")], 8_729)
        self.assertEqual(counts[("train", "1")], 8_795)
        self.assertEqual(counts[("validation", "0")], 1_417)
        self.assertEqual(counts[("validation", "1")], 1_463)


if __name__ == "__main__":
    unittest.main()
