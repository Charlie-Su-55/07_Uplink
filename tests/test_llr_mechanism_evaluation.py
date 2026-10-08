"""Light streaming/identity checks; no real confirmation channel is generated locally."""

import copy
import importlib.metadata
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from evaluation import evaluate_llr_mechanism_controls as evaluation
from evaluation import llr_mechanism_common as common
from evaluation import report_llr_mechanism_controls as reporting
from models.llr_mechanism_controls import make_model
from tests import test_paper_soft_cache as cache_fixtures
from tests.test_cached_llr_evaluation import fixture as cached_fixture, normalization, ToyCodec


def config(mode="smoke"):
    value = common.load_config(mode=mode)
    value["runtime"]["device"] = "cpu"
    value["receiver"]["re_chunk"] = 2
    value["evaluation_chunk"] = 2
    return value


def observation(seed=81000024, ebno=-10.5):
    channel = cached_fixture()
    channel.update(seed=seed, ebno_db=ebno, n0=torch.tensor(.4), native_data_ordinals=torch.arange(5))
    channel["methods"]["sionna_lmmse"] = copy.deepcopy(channel["methods"]["ep5"])
    return channel


class CapturingScale(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, nodes, edges=None):
        self.inputs.append(nodes.clone())
        return torch.ones(*nodes.shape[:2], 4, dtype=torch.float32, device=nodes.device)


def receiver_fixture(settings):
    normalizer = normalization()
    noise = dict(alpha_min=.05, alpha_max=8., interpolation="linear_log_alpha_log_n0_endpoint_hold",
                 knots=[dict(n0=noise, ebno_db=ebno, alpha=[2.] * 4) for noise, ebno in ((.2, -9), (.4, -10.5), (.8, -12))])
    result = [dict(method=method, candidate=method, training_seed=None, checkpoint_type="fixed", step=None,
                   parameter_count=count) for method, count in zip(evaluation.BASELINES, (0, 0, 1, 4, 12))]
    result[1]["alpha"] = [1.]
    result[2]["alpha"] = [2.]
    result[3]["alpha"] = [2.] * 4
    result[4]["noise_fit"] = noise
    for candidate in settings["candidates"]:
        for seed in settings["training"]["seeds"]:
            model = make_model(candidate, settings).eval()
            result.append(dict(method=f"{candidate}_seed{seed}_best", candidate=candidate, training_seed=seed,
                               checkpoint_type="best", step=2, parameter_count=sum(parameter.numel() for parameter in model.parameters()),
                               model=model, normalization=normalizer))
    for seed in (42, 43, 44):
        original = dict(normalizer, mean=[1.] * 6)
        result.append(dict(method=f"frozen_reference_seed{seed}", candidate="frozen_reference", training_seed=seed,
                           checkpoint_type="frozen_best", step=1000, parameter_count=1412,
                           model=CapturingScale(), normalization=original))
    return result


class StreamingLink(cache_fixtures.TinyLink):
    def __init__(self):
        super().__init__()
        self.codec.decode = Mock(wraps=self.codec.decode)
        self.native_inputs = None

    def equalizer(self, received, estimate, uncertainty, n0):
        self.native_inputs = (received, estimate, uncertainty, n0)
        return torch.zeros(1, 2, 1, 5, dtype=torch.complex64), torch.ones(1, 2, 1, 5)

    def demapper(self, symbols, noise):
        return (2 * self.coded - 1).unsqueeze(2)


def timing_fixture():
    return dict(transmit_seconds=.01, ce_seconds=.02, native_equalizer_app_seconds=.03,
                native_decode_seconds=.04, ep_decode_seconds=.05, ep_seconds=.06, frontend_seconds=.07)


def write_report_stub(directory, summary, rows, settings):
    names = ["summary.csv", "decision_summary.json", "targets.json", "parameters_added_time.csv"]
    names += [f"plots/{name}.{extension}" for name in ("bler_curves", "feature_ablation", "parameters_added_time")
              for extension in ("pdf", "svg")]
    artifacts = []
    for name in names:
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("unit-test report fixture\n", encoding="utf-8")
        artifacts.append(dict(file=name, sha256=evaluation.file_hash(path), bytes=path.stat().st_size))
    return artifacts


class MechanismEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.previous_threads)

    def test_one_transmit_one_actual_ce_native_errvar_and_shared_ep_frontend(self):
        settings, link = config(), StreamingLink()
        captures = []
        ep = cache_fixtures.TinyReceiver(captures)
        with patch.object(evaluation, "make_detector_batch", side_effect=cache_fixtures.CacheFeatureTests.make_batch):
            channel, timing = evaluation.stream_observation(link, ep, 90000000, -10.5, settings)
        self.assertEqual(link.transmit_calls, 1)
        self.assertEqual(link.estimation_calls, 1)
        self.assertEqual(link.codec.decode.call_count, 2)
        self.assertEqual(len(captures), 3)
        self.assertTrue((link.native_inputs[2] > 0).all())
        self.assertEqual(set(channel["methods"]), {"sionna_lmmse", "ep5"})
        self.assertFalse({"y", "h_true", "h_hat", "noise"}.intersection(channel))
        self.assertEqual(channel["methods"]["ep5"]["llr"].shape, (1, 5, 2, 4))
        self.assertTrue(all(value >= 0 for value in timing.values()))
        labels = evaluation.coded_labels(channel)
        self.assertTrue(torch.equal(evaluation.llrs_to_native_order(labels).squeeze(2), channel["coded_bits"]))

    def test_baselines_decoded_once_scalers_share_ep_and_frozen_normalizer_is_original(self):
        settings, channel = config(), observation()
        receivers = receiver_fixture(settings)
        codec = ToyCodec()
        codec.decode = Mock(wraps=codec.decode)
        timing = timing_fixture()
        rows = evaluation.evaluate_observation(channel, receivers, codec, settings, normalization(), timing)
        self.assertEqual(len(rows), 13)
        self.assertEqual(codec.decode.call_count, 11)
        self.assertTrue(all(row["positive_scale_hard_decisions_unchanged"] for row in rows if row["method"] != "native_lmmse"))
        raw_nodes, _ = evaluation.feature_tensors(channel, settings)
        reference = receivers[-1]
        expected = evaluation.normalize_nodes(raw_nodes, reference["normalization"], settings)
        self.assertTrue(torch.equal(torch.cat(reference["model"].inputs), expected))
        self.assertFalse(torch.equal(expected, evaluation.normalize_nodes(raw_nodes, normalization(), settings)))
        self.assertEqual(rows[4]["alpha"]["per_bit_mean"], [2.] * 4)
        self.assertEqual(rows[4]["noise_conditioning"]["position"], "interior")
        for row in rows[1:]:
            self.assertEqual(sum(row["alpha"]["histogram_counts"]), row["alpha"]["count"])
            self.assertEqual(row["alpha"]["histogram_edges"], settings["scale_distribution"]["bin_edges"])
        self.assertEqual(rows[0]["decode_seconds"], .04)
        self.assertEqual(rows[1]["decode_seconds"], .05)

    def test_smoke_fresh_anchor_exact_full_codeword_crc_identity_and_mismatch_stop(self):
        settings, channel = config(), observation()
        qualification = evaluation.qualify_smoke(channel, ToyCodec(), settings, normalization())
        self.assertEqual(qualification["candidates"], {candidate: True for candidate in settings["candidates"]})
        self.assertTrue(qualification["decoded_payload_crc_identity"])
        channel["methods"]["ep5"]["crc_ok"] = ~channel["methods"]["ep5"]["crc_ok"]
        with self.assertRaisesRegex(ValueError, "baseline mismatch"):
            evaluation.qualify_smoke(channel, ToyCodec(), settings, normalization())

    def test_atomic_confirmation_manifest_before_waveform_complete_reuse_and_vector_coverage(self):
        self.run_fixture("confirmation")

    def test_smoke_reads_only_old_validation_and_never_constructs_phy(self):
        self.run_fixture("smoke")

    def test_interrupted_budget_keeps_first_committed_case_and_refuses_partial_reuse(self):
        self.run_fixture("confirmation", fail_after_first=True)

    def run_fixture(self, mode, fail_after_first=False):
        settings = config(mode)
        settings["training"]["seeds"] = [42]
        channel = observation(81000024 if mode == "smoke" else 90000000)
        records = [dict(seed=channel["seed"], ebno_db=-10.5, split="calibration" if mode == "smoke" else "confirmation", file="old.pt")]
        if fail_after_first:
            records.append(dict(records[0], seed=90000001))
        receivers = receiver_fixture(settings)
        trained = dict(normalization=normalization())
        with tempfile.TemporaryDirectory() as temporary:
            paths = {key: Path(temporary) / key for key in ("training", "results", "logs")}
            paths["training"].mkdir()
            (paths["training"] / "manifest.json").write_text("{}", encoding="utf-8")
            artifacts = [dict(file="manifest.json", sha256=evaluation.file_hash(paths["training"] / "manifest.json"))]
            context = dict(run_identity_id="mechanism-run", run_identity={"fixture": True}, code_identity={}, environment={},
                           cache_dir=Path("unused-cache"), cache_manifest=dict(identity_id="old-cache", identity=dict(reference={"fixture": True})),
                           frozen_references=[], config=settings)

            def generate(*args, **kwargs):
                manifest = json.loads((paths["results"] / "evaluation_manifest.json").read_text())
                self.assertEqual(manifest["status"], "running")
                self.assertEqual(manifest["planned_cases"], records)
                self.assertTrue(manifest["training_artifacts"])
                if fail_after_first and manifest["cases"]:
                    raise RuntimeError("simulated second-channel GPU failure")
                return copy.deepcopy(channel), timing_fixture()

            args = ["--run-id", "unit", "--mode", mode]
            with patch.object(common, "load_config", return_value=settings), \
                    patch.object(common, "prepare", return_value=context), \
                    patch.object(common, "records_for", return_value=records) as plan, \
                    patch.object(common, "run_paths", return_value=paths), \
                    patch.object(evaluation, "load_receivers", return_value=(trained, receivers, artifacts)), \
                    patch.object(evaluation, "link_and_ep", return_value=(SimpleNamespace(codec=ToyCodec()), object())) as factory, \
                    patch.object(evaluation, "stream_observation", side_effect=generate) as streaming, \
                    patch("evaluation.paper_soft_common.load_channel", return_value=copy.deepcopy(channel)) as old_loader, \
                    patch.object(evaluation, "codec_from_cache", return_value=ToyCodec()), \
                    patch.object(reporting, "summarize", return_value={}), \
                    patch.object(reporting, "write_outputs", side_effect=write_report_stub):
                if fail_after_first:
                    with self.assertRaisesRegex(RuntimeError, "second-channel GPU failure"):
                        evaluation.main(args)
                    manifest = json.loads((paths["results"] / "evaluation_manifest.json").read_text())
                    self.assertEqual(manifest["status"], "failed")
                    self.assertEqual(len(manifest["cases"]), 1)
                    case = manifest["cases"][0]
                    self.assertEqual(evaluation.file_hash(paths["results"] / case["file"]), case["sha256"])
                    self.assertEqual(streaming.call_count, 2)
                    with self.assertRaises(FileExistsError):
                        evaluation.main(args + ["--reuse-complete"])
                    self.assertEqual(streaming.call_count, 2)
                    return
                self.assertEqual(evaluation.main(args), 0)
                if mode == "smoke":
                    self.assertEqual(plan.call_args.args[1], "validation")
                    factory.assert_not_called()
                    streaming.assert_not_called()
                    self.assertEqual(old_loader.call_count, 1)
                else:
                    self.assertEqual(streaming.call_count, 1)
                    old_loader.assert_not_called()
                streaming.reset_mock()
                self.assertEqual(evaluation.main(args + ["--reuse-complete"]), 0)
                self.assertEqual(evaluation.main(args + ["--check-complete"]), 0)
                streaming.assert_not_called()
                summary_path = paths["results"] / "summary.json"
                summary = json.loads(summary_path.read_text())
                self.assertEqual(summary["independent_channels"], 1)
                self.assertTrue(summary["all_positive_scale_hard_decisions_unchanged"])
                case_path = paths["results"] / evaluation.case_name(channel["seed"], -10.5)
                case = json.loads(case_path.read_text())
                self.assertEqual(case["input_identity"]["coded_bits"], evaluation.tensor_identity(channel["coded_bits"]))
                self.assertFalse({"y", "h_hat", "h_true", "llr", "features"}.intersection(case))
                self.assertEqual(case["ep5_llr_identity"], evaluation.tensor_identity(channel["methods"]["ep5"]["llr"]))
                self.assertEqual(case["inference_feature_identity"]["shape"], [5, 3, 6])
                summary["artifacts"] = [artifact for artifact in summary["artifacts"] if artifact["file"] != "plots/bler_curves.svg"]
                summary["content_sha256"] = evaluation.digest({key: value for key, value in summary.items() if key != "content_sha256"})
                summary_path.write_text(json.dumps(summary), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "vector artifacts"):
                    evaluation.main(args + ["--check-complete"])
                with patch.object(evaluation, "load_receivers", side_effect=ValueError("incomplete training")):
                    with self.assertRaisesRegex(ValueError, "incomplete training"):
                        evaluation.main(args)
                streaming.assert_not_called()


def native_available():
    try:
        return importlib.metadata.version("sionna") == "2.0.1"
    except importlib.metadata.PackageNotFoundError:
        return False


@unittest.skipUnless(native_available(), "Native Sionna 2.0.1 absent; CPU mocks do not qualify the NR link")
class NativeMechanismCodecTests(unittest.TestCase):
    def test_five_fresh_controls_preserve_complete_native_tb_and_crc(self):
        from link_level.nr_codec import NRTransportBlockCodec

        settings = config()
        codec = NRTransportBlockCodec(64, 2, device="cpu")
        info = torch.randint(2, (1, 2, codec.info_bits), generator=torch.Generator().manual_seed(351)).float()
        coded, _ = codec.encode(info)
        channel = dict(seed=81000024, ebno_db=-10.5, n0=torch.tensor(.4), coded_bits=coded, info_bits=info,
                       data_indices=torch.arange(64), native_data_ordinals=torch.arange(64))
        llr = (2 * evaluation.coded_labels(channel) - 1) * 20
        native = evaluation.llrs_to_native_order(llr)
        decoded, crc = codec.decode(native)
        self.assertTrue(torch.equal(decoded, info))
        self.assertTrue(crc.all())
        channel["methods"] = dict(ep5=evaluation.tb_record(llr, native, decoded, crc, info))
        channel["features"] = dict(ep5_mean=torch.zeros(1, 64, 2, dtype=torch.complex64),
                                   ep5_variance=torch.ones(1, 64, 2), residual=torch.ones(1, 64), eta=torch.zeros(1, 64, 2),
                                   gram=torch.eye(2, dtype=torch.complex64).expand(1, 64, -1, -1).clone())
        result = evaluation.qualify_smoke(channel, codec, settings, normalization())
        self.assertEqual(result["candidates"], {candidate: True for candidate in settings["candidates"]})


if __name__ == "__main__":
    unittest.main()
