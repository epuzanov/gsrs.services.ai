import unittest

import _bootstrap  # noqa: F401

from gsrs.model import ChemicalSubstance, Product, Substance
from gsrs.services.ai import SubstanceChunker


class ImportTests(unittest.TestCase):
    def test_imports_exposed(self):
        self.assertIsNotNone(ChemicalSubstance)
        self.assertIsNotNone(Substance)
        self.assertIsNotNone(Product)
        self.assertIsNotNone(SubstanceChunker)


if __name__ == "__main__":
    unittest.main()


