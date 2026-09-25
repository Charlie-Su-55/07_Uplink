"""Accounting tests plus native Sionna checks when the target environment exists."""

import importlib.metadata
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from link_level.sionna_ebno import effective_payload_rate, ebno_to_noise, require_sionna2


def has_sionna2():
    try:
        return importlib.metadata.version("sionna").split(".")[0] == "2"
    except importlib.metadata.PackageNotFoundError:
        return False


class AccountingTests(unittest.TestCase):
    def test_payload_rate_excludes_crc_and_uses_actual_lengths(self):
        self.assertEqual(effective_payload_rate(3496, 10752), 3496 / 10752)
        self.assertNotEqual(effective_payload_rate(3496, 10752), 340 / 1024)
        for values in ((0, 10), (11, 10), (1.5, 10), (True, 10)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                effective_payload_rate(*values)

    def test_bad_grids_fail_before_dependency_import(self):
        for rg in (None, SimpleNamespace(num_streams_per_tx=16),
                   SimpleNamespace(num_streams_per_tx=1, num_data_symbols=191)):
            with self.subTest(grid=rg), self.assertRaises(ValueError):
                ebno_to_noise(0, 4, 240, 768, rg)

    def test_old_environment_has_no_fallback(self):
        with patch("link_level.sionna_ebno.importlib.metadata.version", return_value="0.15.1"):
            with self.assertRaisesRegex(RuntimeError, "GPU server"):
                require_sionna2()


@unittest.skipUnless(has_sionna2(), "Requires server Sionna 2.x; no local environment changes")
class NativeEbnoTests(unittest.TestCase):
    def grid(self, users=16, cp=14):
        from sionna.phy.ofdm import ResourceGrid
        return ResourceGrid(14, 192, 30000, num_tx=users, num_streams_per_tx=1,
                            cyclic_prefix_length=cp, pilot_pattern="empty", device="cpu")

    def test_native_formula_cp_and_per_ue_power(self):
        for qm, payload in ((2, 1728), (4, 3496), (6, 8192)):
            for users in (1, 16):
                for cp in (0, 14):
                    with self.subTest(qm=qm, users=users, cp=cp):
                        rg = self.grid(users, cp)
                        n = 2688 * qm
                        actual = float(ebno_to_noise(3.0, qm, payload, n, rg).item())
                        expected = (1 + cp / 192) / (10 ** .3 * qm * (payload / n))
                        self.assertAlmostEqual(actual, expected, delta=expected * 2e-6)

    def test_three_db_ratio_and_finite_guard(self):
        rg = self.grid()
        n0 = ebno_to_noise(0, 4, 3496, 10752, rg).item()
        n3 = ebno_to_noise(3, 4, 3496, 10752, rg).item()
        self.assertAlmostEqual(n0 / n3, 10 ** .3, places=5)
        for value in (math.nan, math.inf):
            with self.assertRaises(ValueError):
                ebno_to_noise(value, 4, 3496, 10752, rg)


if __name__ == "__main__":
    unittest.main()
