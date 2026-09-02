import ast
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


def load_functions(*names):
    source = Path(__file__).parents[1].joinpath("app.py").read_text()
    tree = ast.parse(source)
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"pd": pd, "np": np}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "app.py", "exec"), namespace)
    return namespace


class PortfolioLogicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.logic = load_functions("apply_trade", "compute_advanced_indicators")

    def test_buy_averages_existing_position(self):
        portfolio = pd.DataFrame({"Quantity": [2.0], "PurchasePrice": [10.0]}, index=["ABC"])
        updated = self.logic["apply_trade"](portfolio, "ABC", "Buy", 2.0, 14.0)
        self.assertEqual(updated.loc["ABC", "Quantity"], 4.0)
        self.assertEqual(updated.loc["ABC", "PurchasePrice"], 12.0)

    def test_sell_removes_position_when_quantity_reaches_zero(self):
        portfolio = pd.DataFrame({"Quantity": [2.0], "PurchasePrice": [10.0]}, index=["ABC"])
        updated = self.logic["apply_trade"](portfolio, "ABC", "Sell", 2.0, 12.0)
        self.assertNotIn("ABC", updated.index)

    def test_indicators_include_expected_values(self):
        values = np.linspace(100, 160, 220)
        index = pd.date_range("2025-01-01", periods=len(values), freq="D")
        history = pd.DataFrame({
            "Close": values,
            "High": values + 2,
            "Low": values - 2,
            "Volume": np.full(len(values), 1000),
        }, index=index)
        indicators = self.logic["compute_advanced_indicators"](history)
        self.assertIn("rsi", indicators)
        self.assertIn("sma200", indicators)
        self.assertGreater(indicators["sma20"], indicators["sma50"])
        self.assertEqual(indicators["trend"], "Strong Uptrend")


if __name__ == "__main__":
    unittest.main()