import json
import tempfile
import unittest
from pathlib import Path
from src.state_reader import load


class InputContractTests(unittest.TestCase):
    def test_primary_material_is_readable(self):
        self.assertGreater(len(load(Path(__file__).resolve().parents[1] / 'fixtures/observations.csv')), 0)

    def test_material_inventory_matches_bytes(self):
        root = Path(__file__).resolve().parents[1]
        import hashlib
        for item in load(root / "material-index.json"):
            path = root / item["path"]
            with self.subTest(path=item["path"]):
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), item["sha256"])
                self.assertIsInstance(load(path), (list, dict))

    def test_blank_is_not_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "values.csv"
            path.write_text("id,value\na,\nb,0\n", encoding="utf-8")
            result = load(path)
            self.assertEqual(result[0]["value"], "")
            self.assertEqual(result[1]["value"], "0")

    def test_bad_record_is_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text('{"key":1}\n{bad}\n', encoding="utf-8")
            with self.assertRaises(json.JSONDecodeError):
                load(path)

    def test_absent_is_not_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                load(Path(directory) / "missing.csv")
