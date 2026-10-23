"""Translate three 16-rank, 4-GiB examples using CPU memory only.

Run from the repository root:
  python3 -B flagcx/tools/examples_4g.py --output-dir algo_output/dsl_v8_4g
"""
import argparse
import json
from pathlib import Path

try:
    from .builders_grouped import build_groupedar
    from .builders_fused import build_allgather_gemm, build_gemm_reducescatter
    from .dsl_validation import validate_workflow
    from .utils import DataType, RedOp
except ImportError:
    from builders_grouped import build_groupedar
    from builders_fused import build_allgather_gemm, build_gemm_reducescatter
    from dsl_validation import validate_workflow
    from utils import DataType, RedOp


def examples():
    # 4 GiB / sizeof(float32); REDSLICESIZE is an ELEMENT count.
    yield "groupedar", build_groupedar(
        world_size=16, count=(4 << 30) // 4, group_size=8,
        datatype=DataType.float32, red_op=RedOp.sum,
        num_slices=4, num_local_slices=2, num_red_slices=0,
        red_slice_size="1M",
    )
    # Global M/K for AG: each rank supplies A[M/16,K], B[K,N].
    yield "allgather_gemm", build_allgather_gemm(
        world_size=16, m=262144, n=4096, k=4096,
        group_size=8, num_slices=4, num_k_slices=1,
    )
    # Global M/K for RS: each rank supplies A[M,K/16], B[K/16,N].
    yield "gemm_reducescatter", build_gemm_reducescatter(
        world_size=16, m=262144, n=4096, k=4096,
        group_size=8, num_slices=4, num_k_slices=1,
    )


def export_examples(output_dir):
    output = Path(output_dir).resolve()
    # Preserve previous generated inputs. Select a new directory to regenerate.
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "format_version": 8, "world_size": 16, "datatype": "float32",
        "logical_bytes": 4 << 30, "gpu_execution": False, "examples": {},
    }
    common_env = {
        "FLAGCX_USE_HETERO_COMM": "1", "FLAGCX_UNIRUNNER_GROUPSIZE": "8",
        "FLAGCX_UNIRUNNER_NSLICES": "4",
        "FLAGCX_UNIRUNNER_MAX_THREADS": "256",
        "FLAGCX_UNIRUNNER_MAX_BLOCKS": "32",
        "FLAGCX_UNIRUNNER_NREDSLICES": "0",
        "FLAGCX_UNIRUNNER_REDSLICESIZE": "1M",
    }
    for name, workflow in examples():
        report = validate_workflow(workflow)
        folder = output / name
        paths = workflow.write_rank_files(str(folder))
        environment = dict(common_env, FLAGCX_UNIRUNNER_ALGO_PATH=str(folder))
        if name == "groupedar":
            environment.update(
                FLAGCX_UNIRUNNER_USE_GROUPEDAR="1",
                FLAGCX_UNIRUNNER_LOCAL_NSLICES="2",
            )
        else:
            selector = ("FLAGCX_UNIRUNNER_FUSED_AG_GEMM_ALGO" if
                        name == "allgather_gemm" else
                        "FLAGCX_UNIRUNNER_FUSED_GEMM_RS_ALGO")
            environment.update({
                selector: "1", "FLAGCX_UNIRUNNER_GEMM_KSLICES": "1",
                "FLAGCX_UNIRUNNER_GEMM_WORKERS_PER_STEP": "128",
                "FLAGCX_UNIRUNNER_MAX_BLOCKS": "128",
            })
        report["algorithm_hash"] = str(workflow.cache_key(0).algorithm_hash())
        report["files"] = [str(Path(path).relative_to(output)) for path in paths]
        report["runtime_environment"] = environment
        manifest["examples"][name] = report
        rank0 = report["ranks"][0]
        print(f"{name}: {len(paths)} files, {rank0['nodes']} nodes/rank, "
              f"scratch={report['bindings']['scratch']} bytes/rank")
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    export_examples(args.output_dir)
