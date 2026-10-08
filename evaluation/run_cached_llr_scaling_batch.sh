#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: bash evaluation/run_cached_llr_scaling_batch.sh smoke|development --run-id NAME [--config PATH] [--reuse-complete]

Reads the existing complete Mode-B cache. Missing cache is an error.
Runs preflight, lightweight tests, training/decode smoke, then development.
Use a fresh run ID; --reuse-complete only permits validated complete artifacts.
Set PYTHON to the executable in the existing server environment.
USAGE
}

if [[ $# -eq 0 ]]; then usage >&2; exit 2; fi
if [[ "$1" == --help || "$1" == -h ]]; then usage; exit 0; fi
mode=$1
shift
case "$mode" in smoke|development) ;; *) usage >&2; exit 2 ;; esac
run_id=
config=configs/evaluation/cached_llr_scaling.yaml
reuse_complete=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-id|--config)
            if [[ $# -lt 2 || "$2" == --* ]]; then usage >&2; exit 2; fi
            if [[ "$1" == --run-id ]]; then run_id=$2; else config=$2; fi
            shift 2
            ;;
        --reuse-complete) reuse_complete=true; shift ;;
        --help|-h) usage; exit 0 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done
if [[ ! "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
    printf 'Require --run-id with a leading letter/digit and only letters, digits, underscore, dot, or hyphen.\n' >&2
    exit 2
fi
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd -- "$repo_root"
python=${PYTHON:-python}
run_ids=$run_id
if [[ "$mode" == development ]]; then run_ids="$run_id,${run_id}_smoke"; fi
path_output=$("$python" -c '
import sys
from evaluation.cached_llr_common import load_config, run_paths
config = load_config(sys.argv[1], mode=sys.argv[3])
for run_id in sys.argv[2].split(","):
    paths = run_paths(config, run_id)
    for name in ("training", "results", "logs"):
        path = str(paths[name])
        if "\n" in path or "\r" in path:
            raise ValueError("Output paths must not contain line breaks")
        print(path)
' "$config" "$run_ids" "$mode")
mapfile -t output_paths <<< "$path_output"
expected_paths=3
if [[ "$mode" == development ]]; then expected_paths=6; fi
if [[ ${#output_paths[@]} -ne $expected_paths ]]; then
    printf 'Could not resolve all experiment output paths.\n' >&2
    exit 2
fi
if [[ "$reuse_complete" == false ]]; then
    for output_path in "${output_paths[@]}"; do
        if [[ -e "$output_path" ]]; then
            printf 'Output exists: %s. Choose a new run ID or use --reuse-complete.\n' "$output_path" >&2
            exit 2
        fi
    done
fi
log_dir=${output_paths[2]}
mkdir -p -- "$log_dir"
lock_dir="$log_dir/.runner.lock"
if ! mkdir -- "$lock_dir" 2>/dev/null; then
    printf 'Runner lock exists: %s. Inspect runner.pid and runner_status.txt before retrying.\n' "$lock_dir" >&2
    exit 2
fi
log_file="$log_dir/runner.log"
status_file="$log_dir/runner_status.txt"
stage_file="$log_dir/stages.tsv"
printf '%s\n' "$$" > "$log_dir/runner.pid"
started=$(date +%s)
started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
current_stage=initializing
stage_started=$started
state=running

write_status() {
    local exit_code=$1 now
    now=$(date +%s)
    printf 'run_id=%s\nmode=%s\npid=%s\nstate=%s\nstage=%s\nstarted_utc=%s\nelapsed_seconds=%s\nstage_elapsed_seconds=%s\nexit_code=%s\nupdated_utc=%s\n' \
        "$run_id" "$mode" "$$" "$state" "$current_stage" "$started_utc" "$((now - started))" \
        "$((now - stage_started))" "$exit_code" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$status_file.tmp.$$"
    mv -- "$status_file.tmp.$$" "$status_file"
}

finish() {
    local exit_code=$?
    trap - EXIT
    if [[ $exit_code -eq 0 ]]; then state=complete; else state=failed; fi
    write_status "$exit_code" || true
    rmdir -- "$lock_dir" || true
    exit "$exit_code"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
write_status ''
if [[ ! -e "$stage_file" ]]; then
    printf 'run_id\tpid\tstage\tstate\telapsed_seconds\texit_code\tutc\n' > "$stage_file"
fi

run_stage() {
    current_stage=$1
    shift
    stage_started=$(date +%s)
    write_status ''
    printf '[%s] START %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$current_stage" | tee -a "$log_file"
    local exit_code=0 pipeline_codes elapsed
    if "$@" 2>&1 | tee -a "$log_file"; then
        exit_code=0
    else
        pipeline_codes=("${PIPESTATUS[@]}")
        if [[ ${pipeline_codes[0]} -ne 0 ]]; then exit_code=${pipeline_codes[0]}; else exit_code=${pipeline_codes[1]}; fi
    fi
    elapsed=$(($(date +%s) - stage_started))
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$run_id" "$$" "$current_stage" \
        "$([[ $exit_code -eq 0 ]] && printf complete || printf failed)" "$elapsed" "$exit_code" \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$stage_file"
    printf '[%s] END %s elapsed=%ss exit=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        "$current_stage" "$elapsed" "$exit_code" | tee -a "$log_file"
    return "$exit_code"
}

reuse_options=()
if [[ "$reuse_complete" == true ]]; then reuse_options+=(--reuse-complete); fi
run_stage preflight "$python" -m training.train_cached_llr_scaling \
    --config "$config" --run-id "$run_id" --mode "$mode" --preflight
run_stage lightweight_tests "$python" -m unittest discover -s tests -p 'test_cached_llr_*.py' -v
smoke_id=$run_id
if [[ "$mode" == development ]]; then smoke_id="${run_id}_smoke"; fi
run_stage smoke_train "$python" -m training.train_cached_llr_scaling \
    --config "$config" --run-id "$smoke_id" --mode smoke "${reuse_options[@]}"
run_stage smoke_evaluate "$python" -m evaluation.evaluate_cached_llr_scaling \
    --config "$config" --run-id "$smoke_id" --mode smoke "${reuse_options[@]}"
run_stage smoke_check "$python" -m evaluation.evaluate_cached_llr_scaling \
    --config "$config" --run-id "$smoke_id" --mode smoke --check-complete
if [[ "$mode" == development ]]; then
    run_stage development_train "$python" -m training.train_cached_llr_scaling \
        --config "$config" --run-id "$run_id" --mode development "${reuse_options[@]}"
    run_stage development_evaluate "$python" -m evaluation.evaluate_cached_llr_scaling \
        --config "$config" --run-id "$run_id" --mode development "${reuse_options[@]}"
    run_stage development_check "$python" -m evaluation.evaluate_cached_llr_scaling \
        --config "$config" --run-id "$run_id" --mode development --check-complete
fi
run_stage summary printf 'Batch complete.\nNew scale training: %s\nResults and summary: %s\nLogs and status: %s\n' \
    "${output_paths[0]}" "${output_paths[1]}" "${output_paths[2]}"
