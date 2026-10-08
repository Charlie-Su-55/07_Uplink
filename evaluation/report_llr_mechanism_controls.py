"""Paired mechanism reports and measured-bracket BLER targets; no model selection."""

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import torch

from evaluation.evaluate_paper_soft_output import aggregate, paired_bootstrap


BASELINES = ("native_lmmse", "ep5_raw", "global_train24", "per_bit_train24", "noise_per_bit_train24")
CONTROLS = ("mlp_without_residual", "mlp_without_eta", "mlp_without_both", "affine_all_features")
DESCRIPTORS = ("method", "candidate", "training_seed", "checkpoint_type", "parameter_count")


def target_crossing(points, target, settings):
    """Use measured adjacent points only; retain every bracket when unresolved."""
    if settings["interpolation"] != "linear_ebno_log10_bler" or not 0 < target < 1:
        raise ValueError("Require the predeclared log10-BLER interpolation and a target in (0,1)")
    points = sorted((float(ebno), float(bler)) for ebno, bler in points)
    if (not points or len({ebno for ebno, _ in points}) != len(points)
            or any(not math.isfinite(ebno) or not math.isfinite(bler) or not 0 <= bler <= 1 for ebno, bler in points)):
        raise ValueError("Target interpolation needs distinct finite measured Eb/N0 points and valid BLER")
    exact = [ebno for ebno, bler in points if bler == target]
    brackets = []
    for left, right in zip(points, points[1:]):
        if min(left[1], right[1]) <= target <= max(left[1], right[1]):
            direction = "descending" if left[1] > right[1] else "ascending" if left[1] < right[1] else "flat"
            estimate = None
            if left[1] != right[1] and min(left[1], right[1]) > 0:
                weight = (math.log10(target) - math.log10(left[1])) / (math.log10(right[1]) - math.log10(left[1]))
                estimate = left[0] + weight * (right[0] - left[0])
            brackets.append(dict(left_ebno_db=left[0], left_bler=left[1], right_ebno_db=right[0], right_bler=right[1],
                                 direction=direction, interpolated_ebno_db=estimate))
    result = dict(target_bler=target, interpolation=settings["interpolation"], measured_points=[list(point) for point in points],
                  brackets=brackets, exact_measured_ebno_dbs=exact, extrapolated=False, required_ebno_db=None)
    if any(right[1] > left[1] for left, right in zip(points, points[1:])):
        return dict(result, status="unresolved", reason="nonmonotonic_measured_curve")
    if len(exact) > 1 or any(item["direction"] == "flat" for item in brackets):
        return dict(result, status="unresolved", reason="multiple_exact_points_or_plateau")
    if len(exact) == 1:
        return dict(result, status="resolved", reason="exact_measured_target", required_ebno_db=exact[0])
    if len(brackets) > 1:
        return dict(result, status="unresolved", reason="multiple_measured_brackets")
    if not brackets:
        return dict(result, status="unresolved", reason="target_not_bracketed_by_measured_points")
    if brackets[0]["interpolated_ebno_db"] is None:
        return dict(result, status="unresolved", reason="zero_endpoint_cannot_interpolate_log_bler")
    return dict(result, status="resolved", reason="adjacent_measured_bracket",
                required_ebno_db=brackets[0]["interpolated_ebno_db"])


def _paired(rows, baseline, config):
    bler = paired_bootstrap(rows, baseline, config["bootstrap"])
    bce = paired_bootstrap([dict(row, blocks=row["coded_bits"], block_errors=row["bce_sum_nats"]) for row in rows],
                           [dict(row, blocks=row["coded_bits"], block_errors=row["bce_sum_nats"]) for row in baseline],
                           config["bootstrap"])
    bce["delta_bce_nats"] = bce.pop("delta_bler")
    for statistics in (bler, bce):
        statistics["limitation"] = "Channel-cluster percentile interval conditional on saved trained models; no multiple-comparison correction."
    lookup = {(row["seed"], row["ebno_db"]): row for row in baseline}
    rescue = harm = 0
    for row in rows:
        other = lookup[(row["seed"], row["ebno_db"])]
        errors = torch.as_tensor(row["per_ue"]["block_error"], dtype=torch.bool).reshape(-1)
        previous = torch.as_tensor(other["per_ue"]["block_error"], dtype=torch.bool).reshape(-1)
        if errors.shape != previous.shape or errors.numel() != row["blocks"]:
            raise ValueError("Paired per-UE outcomes do not match TB denominators")
        rescue += int((previous & ~errors).sum())
        harm += int((~previous & errors).sum())
    return dict(bler=bler, bce=bce, rescue=rescue, harm=harm, delta_sign="candidate minus baseline; negative is improvement")


def _histogram(alphas, config):
    edges = config["scale_distribution"]["bin_edges"]
    if not alphas:
        return dict(alpha_histogram_edges=None, alpha_histogram_counts=None, alpha_histogram_fractions=None)
    for item in alphas:
        counts = item.get("histogram_counts", [])
        if (item.get("histogram_edges") != edges or len(counts) != len(edges) - 1
                or any(not isinstance(count, int) or isinstance(count, bool) or count < 0 for count in counts)
                or sum(counts) != item["count"] or item["count"] < 1):
            raise ValueError("Scale histogram must match the fixed bin edges and complete scale count")
    counts = [sum(item["histogram_counts"][index] for item in alphas) for index in range(len(edges) - 1)]
    total = sum(counts)
    return dict(alpha_histogram_edges=list(edges), alpha_histogram_counts=counts,
                alpha_histogram_fractions=[count / total for count in counts])


def _validate_rows(rows, config):
    if not rows:
        raise ValueError("Cannot report an empty experiment")
    grouped, descriptors = {}, {}
    for row in rows:
        method = row["method"]
        descriptor = {name: row[name] for name in DESCRIPTORS}
        descriptor["step"] = row.get("step")
        candidate = descriptor["candidate"]
        feature_counts = dict(full_mlp=6, mlp_without_residual=5, mlp_without_eta=5,
                              mlp_without_both=4, affine_all_features=6, frozen_reference=6)
        descriptor["effective_input_features"] = feature_counts.get(candidate, 1 if method == "noise_per_bit_train24" else 0)
        descriptor["nominal_input_features"] = 6 if candidate in feature_counts else descriptor["effective_input_features"]
        if method in descriptors and descriptors[method] != descriptor:
            raise ValueError("Receiver identity changes between confirmation channels")
        descriptors[method] = descriptor
        grouped.setdefault(method, []).append(row)
        if row["checkpoint_type"] == "last":
            raise ValueError("Primary confirmation reports only preselected best checkpoints")
        if any(not math.isfinite(float(row[name])) or row[name] < 0 for name in ("scale_forward_seconds", "decode_seconds")):
            raise ValueError("Added processing times must be finite and nonnegative")
    expected_methods = set(BASELINES)
    expected_methods.update(f"{candidate}_seed{seed}_best" for candidate in config["candidates"] for seed in config["training"]["seeds"])
    expected_methods.update(f"frozen_reference_seed{seed}" for seed in (42, 43, 44))
    if set(grouped) != expected_methods:
        raise ValueError("Missing or unexpected receiver in the prespecified primary confirmation pool")
    reference = grouped["ep5_raw"]
    cases = {(row["seed"], row["ebno_db"]): row for row in reference}
    if len(cases) != len(reference):
        raise ValueError("Duplicate channel/EbNo observations")
    if config.get("mode") == "confirmation":
        seeds = {seed for seed, _ in cases}
        expected = {(seed, ebno) for seed in seeds for ebno in config["confirmation"]["ebno_dbs"]}
        if len(seeds) != config["confirmation"]["channels"] or set(cases) != expected:
            raise ValueError("Confirmation report cannot stop before its fixed complete channel/EbNo budget")
    for method, selected in grouped.items():
        keys = [(row["seed"], row["ebno_db"]) for row in selected]
        if len(keys) != len(set(keys)) or set(keys) != set(cases):
            raise ValueError("All receivers must use the same complete channel/EbNo realizations")
        for row in selected:
            other = cases[(row["seed"], row["ebno_db"])]
            if any(row[name] != other[name] for name in ("coded_bits", "payload_bits", "blocks", "n0")):
                raise ValueError("Paired receivers disagree on waveform/noise/bit denominators")
    return grouped, descriptors


def summarize(rows, config):
    grouped, descriptors = _validate_rows(rows, config)
    points = [None] + sorted({row["ebno_db"] for row in rows})
    metrics, comparisons, seed_summaries = [], [], []
    point_groups = {(method, point): [row for row in selected if point is None or row["ebno_db"] == point]
                    for method, selected in grouped.items() for point in points}
    for method, descriptor in descriptors.items():
        for point in points:
            selected = point_groups[(method, point)]
            alphas = [row["alpha"] for row in selected if row.get("alpha") is not None]
            alpha_count = sum(item["count"] for item in alphas)
            scale_seconds = sum(row["scale_forward_seconds"] for row in selected)
            decode_seconds = sum(row["decode_seconds"] for row in selected)
            metrics.append(dict(**descriptor, ebno_db=point, **aggregate(selected),
                                alpha_mean=sum(item["sum"] for item in alphas) / alpha_count if alpha_count else None,
                                alpha_min=min(item["min"] for item in alphas) if alphas else None,
                                alpha_max=max(item["max"] for item in alphas) if alphas else None,
                                **_histogram(alphas, config),
                                alpha_lower_fraction=sum(item["lower_count"] for item in alphas) / alpha_count if alpha_count else None,
                                alpha_upper_fraction=sum(item["upper_count"] for item in alphas) / alpha_count if alpha_count else None,
                                scale_forward_seconds=scale_seconds, decode_seconds=decode_seconds,
                                scale_forward_ms_per_channel=1000 * scale_seconds / len(selected),
                                decode_ms_per_channel=1000 * decode_seconds / len(selected)))
    pairs = []
    for method, descriptor in descriptors.items():
        if descriptor["candidate"] != "full_mlp":
            continue
        training_seed = descriptor["training_seed"]
        requested = [(baseline, "full_mlp_vs_" + baseline) for baseline in BASELINES[2:]]
        requested += [(f"{control}_seed{training_seed}_best", "full_mlp_vs_" + control) for control in CONTROLS]
        requested += [(f"frozen_reference_seed{training_seed}", "full_mlp_vs_frozen_reference")]
        for baseline, category in requested:
            if baseline not in grouped:
                raise ValueError(f"Missing prespecified paired control: {baseline}")
            pairs.append((method, baseline, category, training_seed))
            for point in points:
                comparisons.append(dict(method=method, baseline=baseline, category=category, training_seed=training_seed,
                                        ebno_db=point, **_paired(point_groups[(method, point)], point_groups[(baseline, point)], config)))
    for candidate in [*config["candidates"], "frozen_reference"]:
        for point in points:
            selected = [item for item in metrics if item["candidate"] == candidate and item["ebno_db"] == point]
            if not selected:
                continue
            seeds = [item["training_seed"] for item in selected]
            if len(seeds) != len(set(seeds)):
                raise ValueError("Duplicate checkpoint/model in training-seed aggregation")
            summary = dict(candidate=candidate, ebno_db=point, training_seeds=sorted(seeds),
                           parameter_count=selected[0]["parameter_count"], effective_input_features=selected[0]["effective_input_features"],
                           nominal_input_features=selected[0]["nominal_input_features"], independent_channels=selected[0]["independent_channels"],
                           channel_ebno_observations_per_training_seed=selected[0]["channel_ebno_observations"],
                           note="Same channel set reused for each training seed; count is not multiplied by the number of models")
            for field in ("bler", "bce_nats", "coded_ber", "payload_ber", "fixed_scale_gmi_proxy", "block_errors",
                          "crc_failures", "undetected_block_errors", "crc_failures_without_payload_error", "rescue_vs_raw", "harm_vs_raw",
                          "alpha_mean", "alpha_min", "alpha_max", "alpha_lower_fraction", "alpha_upper_fraction",
                          "scale_forward_ms_per_channel", "decode_ms_per_channel"):
                values = [item[field] for item in selected]
                summary[field] = dict(mean=sum(values) / len(values), min=min(values), max=max(values))
            seed_summaries.append(summary)
    targets = []
    for method in descriptors:
        curve = [(item["ebno_db"], item["bler"]) for item in metrics if item["method"] == method and item["ebno_db"] is not None]
        for target in config["targets"]["bler"]:
            targets.append(dict(curve=method, aggregation="individual_model", **target_crossing(curve, target, config["targets"])))
    for candidate in [*config["candidates"], "frozen_reference"]:
        curve = [(item["ebno_db"], item["bler"]["mean"]) for item in seed_summaries
                 if item["candidate"] == candidate and item["ebno_db"] is not None]
        if curve:
            for target in config["targets"]["bler"]:
                targets.append(dict(curve=candidate + "_training_seed_mean", aggregation="training_seed_mean",
                                    **target_crossing(curve, target, config["targets"])))
    target_lookup = {(item["curve"], item["target_bler"]): item for item in targets}
    mean_pairs = [("full_mlp_training_seed_mean", baseline, category, None) for baseline, category in
                  [(name, "full_mlp_vs_" + name) for name in BASELINES[2:]]]
    mean_pairs += [("full_mlp_training_seed_mean", control + "_training_seed_mean", "full_mlp_vs_" + control, None)
                   for control in [*CONTROLS, "frozen_reference"]]
    gains = []
    for method, baseline, category, training_seed in pairs + mean_pairs:
        for target in config["targets"]["bler"]:
            if (method, target) not in target_lookup or (baseline, target) not in target_lookup:
                continue
            candidate_crossing, baseline_crossing = target_lookup[(method, target)], target_lookup[(baseline, target)]
            resolved = candidate_crossing["status"] == baseline_crossing["status"] == "resolved"
            gains.append(dict(method=method, baseline=baseline, category=category, training_seed=training_seed,
                              target_bler=target, status="resolved" if resolved else "unresolved",
                              gain_db=baseline_crossing["required_ebno_db"] - candidate_crossing["required_ebno_db"] if resolved else None,
                              candidate_status=candidate_crossing["status"], baseline_status=baseline_crossing["status"],
                              sign="baseline required Eb/N0 minus candidate required Eb/N0; positive is candidate gain"))
    decisions = []
    for category in sorted({item["category"] for item in comparisons}):
        for point in points:
            selected = [item for item in comparisons if item["category"] == category and item["ebno_db"] == point]
            decisions.append(dict(comparison=category, ebno_db=point,
                                  training_seed_count=len(selected), training_seeds=[item["training_seed"] for item in selected],
                                  bler_improved_seeds=sum(item["bler"]["delta_bler"] < 0 for item in selected),
                                  bler_worsened_seeds=sum(item["bler"]["delta_bler"] > 0 for item in selected),
                                  bce_improved_seeds=sum(item["bce"]["delta_bce_nats"] < 0 for item in selected),
                                  all_seed_bler_intervals_below_zero=all(item["bler"]["interval"][1] < 0 for item in selected),
                                  per_training_seed=[dict(training_seed=item["training_seed"], delta_bler=item["bler"]["delta_bler"],
                                                          bler_interval=item["bler"]["interval"], delta_bce_nats=item["bce"]["delta_bce_nats"],
                                                          bce_interval=item["bce"]["interval"], rescue=item["rescue"], harm=item["harm"],
                                                          bce_reduced_without_bler_reduction=item["bce"]["delta_bce_nats"] < 0 and item["bler"]["delta_bler"] >= 0)
                                                     for item in selected]))
    return dict(metrics=metrics, paired_comparisons=comparisons, training_seed_summary=seed_summaries,
                targets=targets, target_gains=gains,
                timing_note="Added scale and decoder times exclude shared PHY/CE/EP/frontend/features; not a full-receiver speedup claim.",
                decision_summary=dict(comparisons=decisions, targets=targets, target_gains=gains,
                    primary_checkpoint="best selected on the fixed validation split before confirmation; last saved outside primary pool",
                    protocol="Fixed confirmation seed/EbNo budget; no performance stopping, refitting, checkpoint choice or best-training-seed selection.",
                    limitations=["Channel-cluster intervals do not quantify training randomness; all training seeds and their mean/range are reported.",
                                 "Multiple predeclared comparisons have no multiplicity correction; BCE and BLER findings are distinct.",
                                 "Target dB gains require resolved measured adjacent brackets for both curves; no extrapolation or favorable crossing selection.",
                                 "Smoke uses old validation data only and is not independent confirmation."],
                    related_work=config.get("related_work", {})))


def _write_new(path, writer):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite report artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        writer(temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _plots(root, summary, config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics, means = summary["metrics"], summary["training_seed_summary"]
    seeds = sorted({item["training_seed"] for item in metrics if item["candidate"] == "full_mlp"})
    colors = {name: plt.get_cmap("tab10")(index) for index, name in enumerate([*BASELINES, *config["candidates"], "frozen_reference"])}
    created = []

    def curve(axis, points, label, color, spread=None):
        points = sorted(points)
        if not points:
            return
        horizontal, vertical = zip(*points)
        visible = [max(value, 1e-5) for value in vertical]
        axis.plot(horizontal, visible, marker="o", markersize=3, label=label, color=color, linewidth=1.3)
        zero = [index for index, value in enumerate(vertical) if value == 0]
        if zero:
            axis.scatter([horizontal[index] for index in zero], [1e-5] * len(zero), marker="v", color=color)
        if spread:
            axis.fill_between(horizontal, [max(item[0], 1e-5) for item in spread],
                              [max(item[1], 1e-5) for item in spread], color=color, alpha=.15)

    def finish(figure, name):
        figure.tight_layout(rect=(0, .045, 1, .96))
        for extension in ("pdf", "svg"):
            path = root / "plots" / f"{name}.{extension}"
            _write_new(path, lambda temporary: figure.savefig(temporary, format=extension, bbox_inches="tight"))
            created.append(path)
        plt.close(figure)

    for name, candidates, baselines in (("bler_curves", ["full_mlp", "frozen_reference"], BASELINES),
                                        ("feature_ablation", config["candidates"], ())):
        figure, panels = plt.subplots(2, 2, figsize=(13, 9), squeeze=False)
        for axis, training_seed in zip(panels.flat, seeds[:3]):
            for baseline in baselines:
                selected = [item for item in metrics if item["method"] == baseline and item["ebno_db"] is not None]
                curve(axis, [(item["ebno_db"], item["bler"]) for item in selected], baseline, colors[baseline])
            for candidate in candidates:
                selected = [item for item in metrics if item["candidate"] == candidate and item["training_seed"] == training_seed and item["ebno_db"] is not None]
                curve(axis, [(item["ebno_db"], item["bler"]) for item in selected], candidate, colors[candidate])
            axis.set_title(f"Training seed {training_seed}; validation-selected best")
        for axis in list(panels.flat)[len(seeds[:3]):3]:
            axis.set_visible(False)
        axis = panels[1, 1]
        for baseline in baselines:
            selected = [item for item in metrics if item["method"] == baseline and item["ebno_db"] is not None]
            curve(axis, [(item["ebno_db"], item["bler"]) for item in selected], baseline, colors[baseline])
        for candidate in candidates:
            selected = sorted((item for item in means if item["candidate"] == candidate and item["ebno_db"] is not None), key=lambda item: item["ebno_db"])
            curve(axis, [(item["ebno_db"], item["bler"]["mean"]) for item in selected], candidate, colors[candidate],
                  [(item["bler"]["min"], item["bler"]["max"]) for item in selected])
        axis.set_title("Training-seed mean and range; same channel set")
        for axis in panels.flat:
            if axis.get_visible():
                axis.set(xlabel="Eb/N0 [dB]", ylabel="Full-TB BLER", yscale="log")
                axis.grid(True, which="both", alpha=.25)
                axis.legend(fontsize=6.5)
        figure.suptitle(f"{config.get('mode', 'confirmation')}: {name.replace('_', ' ')}")
        figure.text(.02, .01, "v marks zero observed BLER at plotting floor 1e-5; original zeros remain unchanged in target interpolation.", fontsize=8)
        finish(figure, name)
    figure, axis = plt.subplots(figsize=(13, 6.8))
    axis.axis("off")
    table_rows = []
    for baseline in BASELINES:
        item = next(item for item in metrics if item["method"] == baseline and item["ebno_db"] is None)
        table_rows.append([baseline, item["parameter_count"], item["effective_input_features"], f"{item['scale_forward_ms_per_channel']:.4g}",
                           "fixed", f"{item['decode_ms_per_channel']:.4g}"])
    for candidate in [*config["candidates"], "frozen_reference"]:
        selected = [item for item in means if item["candidate"] == candidate and item["ebno_db"] is None]
        if selected:
            item = selected[0]
            scale, decode = item["scale_forward_ms_per_channel"], item["decode_ms_per_channel"]
            table_rows.append([candidate, item["parameter_count"], item["effective_input_features"], f"{scale['mean']:.4g}",
                               f"{scale['min']:.4g} - {scale['max']:.4g}", f"{decode['mean']:.4g}"])
    table = axis.table(cellText=table_rows, colLabels=["Receiver/readout", "Parameters", "Inputs", "Scale ms/channel", "Training-seed range", "Decode ms/channel"],
                       loc="center", cellLoc="left", colWidths=[.27, .09, .07, .18, .21, .18])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.7)
    figure.suptitle("Parameter count and measured added processing time")
    shared = summary.get("shared_pipeline_timing", {})
    if shared.get("case_count") and "totals_seconds" in shared:
        timing_text = "; ".join(f"{key}: {1000 * value / shared['case_count']:.4g} ms/channel" for key, value in shared["totals_seconds"].items())
        figure.text(.02, .07, "Shared pipeline recorded once per realization:\n" + timing_text, fontsize=7, wrap=True)
    figure.text(.02, .01, summary["timing_note"], fontsize=8, wrap=True)
    finish(figure, "parameters_added_time")
    return created


def write_outputs(directory, summary, rows, config):
    """Write derived artifacts only; caller owns summary.json, channel log and status."""
    root = Path(directory)
    paths = []
    csv_path = root / "summary.csv"
    def write_csv(path):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary["metrics"][0]))
            writer.writeheader()
            writer.writerows(summary["metrics"])
    _write_new(csv_path, write_csv)
    paths.append(csv_path)
    parameter_path = root / "parameters_added_time.csv"
    def write_parameters(path):
        fields = [*DESCRIPTORS, "effective_input_features", "nominal_input_features", "scale_forward_ms_per_channel", "decode_ms_per_channel", "independent_channels", "channel_ebno_observations"]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows({name: row[name] for name in fields} for row in summary["metrics"] if row["ebno_db"] is None)
    _write_new(parameter_path, write_parameters)
    paths.append(parameter_path)
    for filename, value in (("decision_summary.json", summary["decision_summary"]),
                            ("targets.json", dict(targets=summary["targets"], target_gains=summary["target_gains"]))):
        path = root / filename
        _write_new(path, lambda temporary, content=value: temporary.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8"))
        paths.append(path)
    paths.extend(_plots(root, summary, config))
    artifacts = []
    for path in paths:
        with path.open("rb") as handle:
            sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
        artifacts.append(dict(file=path.relative_to(root).as_posix(), sha256=sha256, bytes=path.stat().st_size))
    return artifacts
