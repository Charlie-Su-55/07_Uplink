"""Batch/UE/RE/bit ordering; small CPU native codec identity on the server."""

import importlib.metadata
import unittest

try:
    import torch
except ImportError:
    torch = None

from link_level.nr_codec import NRTransportBlockCodec, symbols_to_grid_input, grid_llrs_to_codewords


def has_sionna2():
    try:
        return importlib.metadata.version("sionna").split(".")[0] == "2"
    except importlib.metadata.PackageNotFoundError:
        return False


@unittest.skipIf(torch is None, "Requires small CPU PyTorch tensors")
class OrderingTests(unittest.TestCase):
    def test_batch_user_re_and_bit_axes_preserve_distinct_tags(self):
        tags = torch.arange(2 * 3 * 5 * 4).reshape(2, 3, 5, 4)
        # Native APP packs adjacent Qm bits for each RE within each UE.
        native_llr = tags.reshape(2, 3, 1, 20)
        self.assertTrue(torch.equal(grid_llrs_to_codewords(native_llr, 3, 20), tags.flatten(2)))
        symbols = tags[..., 0]  # intentionally non-contiguous
        self.assertFalse(symbols.is_contiguous())
        grid = symbols_to_grid_input(symbols)
        for b in range(2):
            for k in range(3):
                for re in range(5):
                    self.assertEqual(grid[b, k, 0, re].item(), tags[b, k, re, 0].item())
        with self.assertRaises(ValueError):
            grid_llrs_to_codewords(native_llr.transpose(1, 2), 3, 20)


@unittest.skipUnless(torch is not None and has_sionna2(), "Requires native Sionna 2.x TB codec")
class NativeCodecTests(unittest.TestCase):
    def test_mapper_app_tb_identity_multiple_batches_users_modulations(self):
        from sionna.phy.mapping import Demapper
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for table, index, qm in ((1, 4, 2), (1, 10, 4), (1, 17, 6), (2, 10, 4)):
                with self.subTest(table=table, mcs=index):
                    torch.manual_seed(12)
                    codec = NRTransportBlockCodec(192, 3, table=table, index=index, device="cpu")
                    self.assertEqual(codec.qm, qm)
                    bits = torch.randint(0, 2, (2, 3, codec.info_bits), dtype=torch.float32)
                    coded, symbols = codec.encode(bits)
                    llr = Demapper("app", "qam", qm, device="cpu")(symbols, 0.001)
                    self.assertTrue(torch.equal(llr[:, :, 0] > 0, coded.bool()))
                    decoded, crc = codec.decode(llr)
                    self.assertTrue(torch.equal(decoded, bits))
                    self.assertTrue(crc.all().item())
                    self.assertEqual(codec.rate, bits.shape[-1] / coded.shape[-1])
        finally:
            torch.set_num_threads(old_threads)


if __name__ == "__main__":
    unittest.main()
