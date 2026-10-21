"""
Master test runner for the complete ET-Pfam-R codebase.
Executes all unit and integration test suites:
1. tests/test_plm_abstraction.py (ESM-2, baseline pLM, shape compatibility [1280, L])
2. tests/test_uncertainty_calibration.py (Conformal prediction & Random Neighbor Score OOD)
3. tests/test_retrieval_evidence.py (FAISS index & top-5 homology search)
4. tests/test_sequence_explainability.py (Gradient saliency, domain attribution, motifs)
5. tests/test_predict_et_pfam_r.py (Unified CLI pipeline integration)
"""
import sys
import unittest

from tests.test_plm_abstraction import TestPLMAbstraction
from tests.test_uncertainty_calibration import TestUncertaintyCalibration
from tests.test_retrieval_evidence import TestRetrievalEvidence
from tests.test_sequence_explainability import TestSequenceExplainability
from tests.test_predict_et_pfam_r import TestPredictETPfamR


def run_all_tests():
    print("=" * 80)
    print("  ET-Pfam-R: Executing Master Comprehensive Test Suite")
    print("=" * 80)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suites = [
        ("PLM Abstraction Layer (ESM-2 & Baseline)", loader.loadTestsFromTestCase(TestPLMAbstraction)),
        ("Uncertainty Calibration & OOD (Conformal & RNS)", loader.loadTestsFromTestCase(TestUncertaintyCalibration)),
        ("Retrieval-Augmented Evidence (FAISS)", loader.loadTestsFromTestCase(TestRetrievalEvidence)),
        ("Sequence Explainability & Domain Attribution", loader.loadTestsFromTestCase(TestSequenceExplainability)),
        ("Unified Pipeline (predict_et_pfam_r)", loader.loadTestsFromTestCase(TestPredictETPfamR)),
    ]

    total_tests = 0
    total_failures = 0
    total_errors = 0

    for name, s in suites:
        print(f"\n[Running Suite] {name}")
        print("-" * 75)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(s)

        total_tests += result.testsRun
        total_failures += len(result.failures)
        total_errors += len(result.errors)

        if not result.wasSuccessful():
            print(f"FAILED: {name}")
        else:
            print(f"PASSED: {name}")

    print("\n" + "=" * 80)
    print(f"TEST SUMMARY: {total_tests} Tests Run | {total_failures} Failures | {total_errors} Errors")
    print("=" * 80)

    success = (total_failures == 0 and total_errors == 0)
    return success


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
