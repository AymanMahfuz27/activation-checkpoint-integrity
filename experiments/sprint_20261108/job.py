"""Execute one source-bound development study, retaining failures and outputs.

Input is a frozen contract admitted by the existing BA/BB native gates. Install
qualification precedes numerical imports, then ordered fixtures precede training.
The output is execution evidence; independent review owns scientific acceptance.
"""

import importlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import tarfile
import time
import traceback
import unittest


def interrupted(signum, frame):
    raise KeyboardInterrupt("Allocated science cutoff; retain partial evidence")


def fixture_ids(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from fixture_ids(item)
        else:
            yield item.id()


def main():
    checkout = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "_sprint_runtime", checkout / "experiments/bb_20261108/copy_runtime.py"
    )
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    common = runtime._common
    job = os.environ.get("SLURM_JOB_ID", "")
    if not job.isdecimal() or "BB_NUMERICAL_ADMISSION" in os.environ:
        raise RuntimeError("Fresh numerical admission required")
    out = common.ROOT / "artifacts" / job / "bb-selector"
    out.mkdir(parents=True, exist_ok=False)
    for signum in (signal.SIGALRM, signal.SIGUSR1, signal.SIGTERM):
        signal.signal(signum, interrupted)
    contract, status = None, 1
    try:
        contract = common.admit(checkout, Path(os.environ["BA_CONTRACT_PATH"]), out)
        runtime.qualify(checkout, out, contract)
        common.write(out / "contract.json", contract)
        common.write(out / "execution-release.json", json.loads(
            Path(os.environ["BA_RELEASE_PATH"]).read_text()))
        with tarfile.open(out / "exact-source.tar.gz", "x:gz") as archive:
            for name in sorted(contract["files"]):
                archive.add(checkout / name, arcname=name, recursive=False)
        for item in contract["inputs"]:
            source = Path(item["path"])
            common.verify_file(source, item["sha256"])
            if source.stat().st_size != item["bytes"]:
                raise RuntimeError("Input byte count differs")
        common.write(out / "verified-inputs.json", contract["inputs"])
        os.environ["BB_NUMERICAL_ADMISSION"] = "PASS:" + job
        sys.path.insert(0, str(checkout))
        sys.path.insert(0, str(checkout / "src"))
        sys.path.insert(0, str(checkout / "experiments/bb_20261108"))
        sys.path.insert(0, str(checkout / "experiments/sprint_20261108/selection"))
        # Real physical/source/environment/input admission has completed above.
        torch = importlib.import_module("torch")
        if not torch.cuda.is_available():
            raise RuntimeError("A real CUDA device is required")
        if (torch.cuda.device_count() != contract["visible_cuda_devices"]
                or "A100" not in torch.cuda.get_device_name(0)
                or list(torch.cuda.get_device_capability(0)) != [8, 0]):
            raise RuntimeError("Frozen A100 hardware scope differs")
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        common.write(out / "hardware.json", {
            "python": sys.version, "torch": str(torch.__version__),
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "visible_devices": torch.cuda.device_count(),
            "driver": common.command(out, "nvidia-smi", ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"]),
            "deterministic": torch.are_deterministic_algorithms_enabled(),
        })
        # First actual backend use includes lazy native compilation. Fixtures
        # warm that cache, so preserve this cold cost separately for all arms.
        torch.cuda.synchronize()
        cold_started = time.perf_counter()
        fingerprint = importlib.import_module("ac_integrity.capture.fingerprint")
        fingerprint.fingerprint_tensor(torch.zeros(2048, device="cuda"),
            include_sketch=False, backend="full")
        torch.cuda.synchronize()
        cold_seconds = time.perf_counter() - cold_started
        os.environ["SPRINT_CHECKER_STARTUP_SECONDS"] = str(cold_seconds)
        common.write(out / "checker-cold-start.json", {
            "elapsed_seconds": cold_seconds,
            "boundary": "first fingerprint import, lazy native build, CUDA allocation/hash/synchronization",
            "charged_separately_from_warm_training": True})
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromNames(contract["fixture_modules"])
        loaded = list(fixture_ids(suite))
        common.write(out / "fixture-census.json", {
            "expected": contract["fixture_ids"], "loaded": loaded})
        if loader.errors or not loaded or loaded != contract["fixture_ids"]:
            raise RuntimeError("Declared fixture modules failed to load")
        fixture_started = time.perf_counter()
        with (out / "fixtures.log").open("x") as stream:
            result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        common.write(out / "fixture-result.json", {
            "run": result.testsRun, "failures": len(result.failures),
            "errors": len(result.errors), "skips": len(result.skipped),
            "expected_failures": len(result.expectedFailures),
            "unexpected_successes": len(result.unexpectedSuccesses),
            "successful": result.wasSuccessful(),
            "elapsed_seconds": time.perf_counter() - fixture_started,
        })
        if not result.wasSuccessful() or result.skipped or result.expectedFailures:
            raise RuntimeError("Development fixtures did not all pass normally")
        if contract.get("calibrate_target", False):
            calibration = importlib.import_module("experiments.sprint_20261108.calibrate_target")
            prior = next(item for item in contract["inputs"] if item["role"] == "prior-clean-calibration")
            calibration.derive(prior["path"], out / "derived-target.json")
        if contract.get("projection_probe", False):
            probe = importlib.import_module("experiments.sprint_20261108.selection.projection_probe")
            probe.run_probe(out / "projection-probe.json")
        module = importlib.import_module(contract["training_module"])
        config = json.loads((checkout / contract["training_config"]).read_text())
        training_out = out / "training"
        training_out.mkdir()
        returned = module.run_study(config, training_out)
        analysis = None
        figures = None
        if contract.get("run_analysis", False):
            reducer = importlib.import_module("experiments.sprint_20261108.analyze")
            analysis = reducer.analyze(returned, config, out / "analysis")
        if contract.get("plot_curves", False):
            plotter = importlib.import_module("experiments.sprint_20261108.plot_training")
            figures = plotter.plot_curves(training_out, config, out / "figures")
        common.write(out / "results.json", {
            "execution_complete": True, "status": "PENDING_INDEPENDENT_ACTUAL_REVIEW",
            "scientific_acceptance": False, "study": contract["campaign_id"],
            "training_return": returned,
            "analysis": analysis, "figures": figures,
        })
        status = 0
    except BaseException as error:
        common.write(out / "failure.json", {
            "type": type(error).__name__, "message": str(error),
            "traceback": traceback.format_exc(), "scientific_acceptance": False,
        })
        traceback.print_exc()
    finally:
        signal.alarm(0)
        for signum in (signal.SIGUSR1, signal.SIGTERM):
            signal.signal(signum, signal.SIG_IGN)
        common.write(out / "terminal.json", {"exit_code": status,
            "scientific_acceptance": False})
        if contract is not None:
            common.retain(out, contract["campaign_id"], contract["output_max_bytes"])
    return status


if __name__ == "__main__":
    raise SystemExit(main())
