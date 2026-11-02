import unittest
from unittest.mock import Mock

import torch

from models.r3p5_reference6000_adapter import R3P5Reference6000Adapter


class AdapterContractTests(unittest.TestCase):
    def test_shared_ce_whitening_and_full_grid_order(self):
        adapter = object.__new__(R3P5Reference6000Adapter)
        adapter.device = torch.device("cpu")
        output = torch.ones((1, 12, 192, 16, 4), dtype=torch.float32)
        adapter.receiver = Mock()
        adapter.receiver.return_value = output
        adapter.receiver.last_audit = [{"seed": 7}]
        y = torch.ones((1, 12, 192, 256), dtype=torch.complex64)
        h = torch.ones((1, 12, 192, 256, 16), dtype=torch.complex64)
        err = torch.full_like(h.real, .25)
        llr, audit = adapter.predict(y, h, err, torch.tensor([1.]), seed=7)
        args, kwargs = adapter.receiver.call_args
        torch.testing.assert_close(args[0], torch.full_like(y, 1 / (5 ** .5)))
        torch.testing.assert_close(args[1], torch.full_like(h, 1 / (5 ** .5)))
        torch.testing.assert_close(args[2], torch.eye(256, dtype=torch.complex64)[None])
        self.assertEqual(kwargs, {"seed": 7})
        self.assertEqual(tuple(llr.shape), (1, 12, 192, 16, 4))
        self.assertEqual(audit["scale"], "sqrt(N0+sum(err_var)); Ruu=I")
        self.assertNotIn("bits", adapter.predict.__code__.co_varnames)

    def test_rejects_partial_window_and_bad_uncertainty(self):
        adapter = object.__new__(R3P5Reference6000Adapter)
        adapter.device = torch.device("cpu")
        adapter.receiver = Mock()
        y = torch.ones((1, 1, 8, 256), dtype=torch.complex64)
        h = torch.ones((1, 1, 8, 256, 16), dtype=torch.complex64)
        with self.assertRaises(ValueError):
            adapter.predict(y, h, torch.ones_like(h.real), torch.tensor([1.]), seed=1)
        y = torch.ones((1, 12, 192, 256), dtype=torch.complex64)
        h = torch.ones((1, 12, 192, 256, 16), dtype=torch.complex64)
        with self.assertRaises(ValueError):
            adapter.predict(y, h, torch.full_like(h.real, -1), torch.tensor([1.]), seed=1)
        adapter.receiver.assert_not_called()


if __name__ == "__main__":
    unittest.main()
