import json
import subprocess
import unittest
import json
import os
import subprocess
import tempfile
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


class PpuCiRegressionTest(unittest.TestCase):
    def test_legacy_unit_skip_keeps_other_p2p_phases_and_failures(self):
        source = (REPO_ROOT / '.github/scripts/ci/run_unit_test.sh').read_text()
        case = source[source.index('    p2p)\n'):source.index('    rma)\n')]
        helpers = source[source.index('flagcx_ci_report_skipped_suite() {'):
                         source.index('\nif [[ "$(basename "$SET_ENV_SCRIPT" .sh)" == kunlunxin ]]')]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recorder = root / 'record'
            recorder.write_text('''#!/usr/bin/env python3
import json, os, sys
label = os.getenv('FLAGCX_CI_MPI_LABEL', os.getenv('FLAGCX_CI_TEST_LABEL', 'build'))
with open(os.environ['CAPTURE'], 'a') as out:
    out.write(json.dumps(label) + '\\n')
sys.exit(17 if label == os.getenv('FAIL_LABEL') else 0)
''')
            recorder.chmod(0o755)
            harness = '''set -euo pipefail
source "$1"
PROJECT_ROOT=$2
SUITE=p2p
suite_dir=/fake/p2p
args=(TEST_ONLY=1)
TEST_RUNNER=$3
MPI_RUNNER=/fake/mpi
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
nproc() { echo 2; }
make() { "$TEST_RUNNER" "$@"; }
''' + helpers + '''
run_suite() {
  case "$SUITE" in
''' + case + '''  esac
}
run_suite
'''
            legacy = 'p2p unit tests'
            phases = ['p2p Engine WRITE MPI tests', 'p2p shared-engine unit tests',
                      'p2p shared-engine WRITE MPI tests']
            for platform in ('ppu', 'cuda', 'hygon'):
                for failure in ('', legacy, phases[1], phases[2]):
                    with self.subTest(platform=platform, failure=failure):
                        capture = root / 'calls'
                        capture.write_text('')
                        result = subprocess.run(
                            ['bash', '-c', harness, 'bash',
                             str(REPO_ROOT / f'.github/scripts/set_env/{platform}.sh'),
                             str(REPO_ROOT), str(recorder)],
                            env={**os.environ, 'CAPTURE': str(capture),
                                 'FAIL_LABEL': failure, 'MPI_HOME': '/fake/mpi',
                                 'FLAGCX_CI_IGNORE_TEST': json.dumps(yaml.safe_load(
                                     (REPO_ROOT / f'.github/configs/{platform}.yml').read_text()
                                 )['ignore_test'])},
                            capture_output=True, text=True, timeout=15,
                        )
                        labels = [json.loads(line) for line in capture.read_text().splitlines()]
                        expected_failure = bool(failure) and not (platform == 'ppu' and failure == legacy)
                        self.assertEqual(result.returncode, int(expected_failure), result.stderr)
                        self.assertEqual(legacy in labels, platform != 'ppu')
                        for label in phases:
                            self.assertIn(label, labels)
                        self.assertEqual('SKIPPED p2p/p2p-legacy-unit' in result.stdout,
                                         platform == 'ppu')

    def test_runner_uses_builtin_kernels_for_all_operations_and_preserves_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake = root / "runner"
            fake.write_text(r"""#!/usr/bin/env python3
import json, os, sys

args = sys.argv[1:]
label = os.environ.get("FLAGCX_CI_MPI_LABEL", os.environ.get("FLAGCX_CI_TEST_LABEL"))
assert os.environ["PCCL_ENABLE_EXT_KERNEL"] == "0"
assert os.environ["GTEST_FILTER"] == "*-CoverageSupplemental.*"
if args[0] != "make":
    exports = dict(arg.split("=", 1) for i, arg in enumerate(args)
                   if i > 0 and args[i - 1] == "-x")
    assert exports["PCCL_ENABLE_EXT_KERNEL"] == "0"
    assert "./build/bin/runner_mpi_tests" in args
    if label == "runner PPU PXN four-rank":
        assert args[-1] == "--gtest_filter=FlagCXCollTest.SendRecv:FlagCXCollTest.AllReduce:FlagCXCollTest.AlltoAll"
    else:
        assert not any("gtest_filter" in arg for arg in args)
with open(os.environ["CALL_LOG"], "a") as out:
    out.write(json.dumps({"label": label, "args": args}) + "\n")
if label == os.environ["FAIL_LABEL"]:
    sys.exit(139)
""")
            fake.chmod(0o755)
            for failure in ("none", "runner unit tests", "runner default",
                            "runner PPU PXN four-rank"):
                with self.subTest(failure=failure):
                    calls = root / (failure.replace(" ", "-") + ".jsonl")
                    env = dict(os.environ, TEST_RUNNER=str(fake), MPI_RUNNER=str(fake),
                               CALL_LOG=str(calls), FAIL_LABEL=failure,
                               MPI_HOME=str(root / "mpi-home"), PROJECT_ROOT=str(REPO_ROOT),
                               FLAGCX_CI_ENABLE_PXN="1", PCCL_ENABLE_EXT_KERNEL="1",
                               NCCL_ENABLE_EXT_KERNEL="1", GTEST_FILTER="FlagCXCollTest.AllReduce")
                    result = subprocess.run([
                        "bash", "-euc",
                        'source "$1"; flagcx_ci_run_suite_override runner "$2" '
                        'USE_PPU=1 USE_ACCL_BAREX=1',
                        "bash", str(REPO_ROOT / ".github/scripts/set_env/ppu.sh"), str(root),
                    ], env=env, text=True, capture_output=True)
                    self.assertEqual(result.returncode, 0 if failure == "none" else 1,
                                     result.stdout + result.stderr)
                    invoked = [json.loads(line) for line in calls.read_text().splitlines()]
                    self.assertEqual([case["label"] for case in invoked], [
                        "runner unit tests", "runner default",
                        "runner BAREX heterogeneous",
                        "runner BAREX forced NET", "runner PPU PXN four-rank",
                    ])
                    self.assertEqual([case["args"][1] for case in invoked[1:]],
                                     ["4", "4", "4", "4"])

    def test_barex_main_adaptor_exposes_runtime_multichannel_transport(self):
        barex = (
            REPO_ROOT / "flagcx/adaptor/net/barex_adaptor.cc"
        ).read_text()
        runtime_api = (
            REPO_ROOT / "flagcx/adaptor/include/barex_runtime.h"
        ).read_text()
        transport_test = (
            REPO_ROOT / "test/unittest/p2p/test_p2p_engine_transport.cpp"
        ).read_text()
        legacy_engine = (
            REPO_ROOT / "flagcx/core/flagcx_p2p.cc"
        ).read_text()
        legacy_barex_engine = (
            REPO_ROOT / "flagcx/core/flagcx_p2p_accl.cc"
        ).read_text()
        shared_engine = (
            REPO_ROOT / "flagcx/core/flagcx_p2p_shared.cc"
        ).read_text()
        makefile = (REPO_ROOT / "Makefile").read_text()
        ppu_env = (
            REPO_ROOT / ".github/scripts/set_env/ppu.sh"
        ).read_text()
        unit_runner = (
            REPO_ROOT / ".github/scripts/ci/run_unit_test.sh"
        ).read_text()
        test_make = (REPO_ROOT / "test/make.inc").read_text()
        read_test = (
            REPO_ROOT / "test/unittest/p2p/test_p2p_engine_read.cpp"
        ).read_text()
        rpc_test = (
            REPO_ROOT / "test/unittest/p2p/test_p2p_engine_rpc.cpp"
        ).read_text()

        self.assertIn("std::vector<XChannel *> channels", barex)
        self.assertIn("flagcxBarexRuntimeEncodeHelloGeometry", barex)
        self.assertIn("flagcxBarexRuntimeSelectLane", barex)
        self.assertIn("flagcxBarexRuntimeResetConnect", barex)
        self.assertIn("flagcxBarexRuntimeDeferMr", barex)
        self.assertIn("flagcxBarexRuntimeDrainDeferredMrs", barex)
        self.assertIn("FLAGCX_NET_SUBMIT_INDEPENDENT", barex)
        self.assertNotIn("rr.fetch_add", barex)
        self.assertIn("FLAGCX_BAREX_RUNTIME_MAX_CHANNELS = 8", runtime_api)
        self.assertIn(
            "BarexOrderedDomainsUseStablePhysicalLanes", transport_test
        )
        self.assertIn(
            "DescriptorPreservesSparsePhysicalNicKeys", transport_test
        )
        self.assertIn(
            "useBarex ? &flagcxNetBarex : &flagcxNetIb", shared_engine
        )
        self.assertIn("flagcxP2pMrSplitPair", shared_engine)
        self.assertIn(
            "SplitsPairedRangesAtEitherProviderBoundary", transport_test
        )
        self.assertIn("adaptor = getNetAdaptor(RDMA)", read_test)
        self.assertIn("adaptor = getNetAdaptor(RDMA)", rpc_test)
        self.assertIn("flagcxNetIbP2p", legacy_engine)
        self.assertIn("return flagcxAcclEngineCreate()", legacy_engine)
        self.assertIn("flagcxP2pIsAccl", legacy_engine)
        self.assertTrue(
            (REPO_ROOT / "flagcx/adaptor/net/ibrc_p2p_adaptor.cc").exists()
        )
        self.assertTrue(
            (REPO_ROOT / "flagcx/core/flagcx_p2p_accl.cc").exists()
        )
        self.assertTrue(
            (REPO_ROOT / "flagcx/core/include/flagcx_p2p_accl.h").exists()
        )
        self.assertIn("acclRetryConflictingDeregs", legacy_barex_engine)
        self.assertIn(
            "if (!acclRetryConflictingDeregs(engine, data, size, dtype))",
            legacy_barex_engine,
        )
        self.assertLess(
            legacy_barex_engine.index(
                "if (!acclRetryConflictingDeregs(engine, data, size, dtype))"
            ),
            legacy_barex_engine.index(
                "const uint32_t reservedMrId = engine->nextMrId++"
            ),
        )
        self.assertIn("current.size = csize", legacy_barex_engine)
        self.assertIn("MSG_NOSIGNAL | MSG_DONTWAIT", legacy_barex_engine)
        self.assertIn("std::chrono::steady_clock::now()", legacy_barex_engine)
        self.assertIn("hasAnyRemoteKey", legacy_barex_engine)
        self.assertIn("remoteMrInfo.rkeys[peerNic] == 0", barex)
        self.assertNotIn("wire.rkeys[key] == 0", legacy_barex_engine)
        self.assertNotIn("region.info.rkeys[key] == 0", shared_engine)
        self.assertIn("if (!replyMrQuery(engine, peer.fd, query))",
                      legacy_barex_engine)
        self.assertIn("USE_SHARED_P2P_ENGINE ?= 0", makefile)
        self.assertIn("ifeq ($(USE_SHARED_P2P_ENGINE), 1)", makefile)
        self.assertIn("flagcx/core/flagcx_p2p_shared.cc", makefile)
        self.assertIn("flagcx/adaptor/net/ibrc_p2p_adaptor.cc", makefile)
        self.assertNotIn("flagcxP2pIsAccl", shared_engine)
        self.assertIn("FLAGCX_CI_ENABLE_SHARED_P2P_ENGINE=1", ppu_env)
        self.assertIn("USE_SHARED_P2P_ENGINE=1", unit_runner)
        self.assertIn('build-p2p-shared', unit_runner)
        self.assertIn("USE_ACCL_BAREX     ?= 0", test_make)
        self.assertIn("CXXFLAGS += -DUSE_ACCL_BAREX", test_make)
        self.assertIn("CXXFLAGS += -DUSE_SHARED_P2P_ENGINE", test_make)

    def test_ppu_symmem_requires_vmm_local_and_barex_vmm_net(self):
        unit_runner = (
            REPO_ROOT / ".github/scripts/ci/run_unit_test.sh"
        ).read_text()
        symmem_runner = (
            REPO_ROOT / "test/script/symmem_test.sh"
        ).read_text()
        barex = (
            REPO_ROOT / "flagcx/adaptor/net/barex_adaptor.cc"
        ).read_text()

        self.assertIn('FLAGCX_CI_MPI_LABEL="symmem VMM local"', symmem_runner)
        self.assertIn(
            'FLAGCX_CI_MPI_LABEL="symmem VMM + NET route union"', unit_runner
        )
        self.assertIn(
            "FLAGCX_CI_ALLOW_VMM_NET_UNSUPPORTED=1", unit_runner
        )
        self.assertIn(
            'FLAGCX_CI_MPI_LABEL="symmem remote without NET"', unit_runner
        )
        self.assertIn(
            'FLAGCX_CI_MPI_LABEL="symmem local IPC + NET fallback"',
            unit_runner,
        )
        self.assertIn('symmem_dmabuf_enable=${FLAGCX_DMABUF_ENABLE:-1}', unit_runner)
        self.assertIn('-x FLAGCX_DMABUF_ENABLE="$symmem_dmabuf_enable"', unit_runner)
        self.assertIn("FLAGCX_CI_REQUIRE_VMM_ROUTE_UNION=1", unit_runner)
        self.assertIn("SymMemTest.VmmNetRouteCapabilityUnion", unit_runner)
        ppu_config = yaml.safe_load(
            (Path(__file__).resolve().parents[2] / ".github/configs/ppu.yml").read_text()
        )["ignore_test"]
        self.assertTrue(any(
            entry["suite"] == "symmem" and entry["name"] == "vmm-net-data"
            for entry in ppu_config["phases"]
        ))
        self.assertIn(
            "if ((symmem_run_vmm_net_data != 0)); then", unit_runner
        )
        self.assertIn(
            "SymMemTest.RemotePeersWithoutNetworkDoNotPublishWindow",
            unit_runner,
        )
        self.assertIn(
            "SymMemTest.RankLocalIpcFailureUsesNetworkMrFallback",
            unit_runner,
        )
        self.assertIn("expected_adaptor=BAREX", unit_runner)
        self.assertIn(
            "SymMemTest.VmmLsaMapFailureDoesNotPublishNetworkOnlyWindow", unit_runner
        )
        self.assertIn(
            "SymMemTest.DirectGdrVmmPreservesMrRoute", unit_runner
        )
        self.assertIn(
            "SymMemTest.CrossNodeLocalCleanupFailureRetries",
            unit_runner,
        )
        self.assertIn(
            "SymMemTest.VmmRollbackFailureConvergesBeforeMrMetadataExchange",
            unit_runner,
        )
        self.assertIn(
            "SymMemTest.SignalRegistrationUsesAllocationProvenance",
            unit_runner,
        )
        self.assertNotIn("barexnet::barexRegMrDmaBuf", barex)
        self.assertIn(
            "ACCL currently has no API that consumes a DMA-BUF fd", barex
        )
        self.assertIn("FLAGCX_VMM_MR_CAP_NONE", barex)
        self.assertIn("barexnet::barexRegMr", barex)
        self.assertNotIn("Requires FLAGCX_VMM_ENABLE=0", barex)

    def test_ppu_runner_executes_barex_heterogeneous_variants(self):
        source = (
            REPO_ROOT / ".github/scripts/set_env/ppu.sh"
        ).read_text()

        self.assertNotIn("Skipping PPU runner heterogeneous", source)
        self.assertNotIn("runner BAREX heterogeneous two-rank", source)
        self.assertIn("runner BAREX heterogeneous", source)
        self.assertIn("runner BAREX forced NET", source)
        self.assertGreaterEqual(source.count("FLAGCX_P2P_TRANSPORT=accl"), 3)
        self.assertNotIn("NCCL_P2P_DISABLE", source)
        self.assertNotIn("NCCL_SHM_DISABLE", source)
        self.assertEqual(source.count("FLAGCX_USE_HETERO_COMM=1"), 0)
        self.assertIn("FLAGCX_CI_EXPECT_RUNNER_MODE=HOMO", source)
        self.assertGreaterEqual(
            source.count("FLAGCX_CI_EXPECT_RUNNER_MODE=HYBRID"), 3
        )
        self.assertGreaterEqual(
            source.count("FLAGCX_CI_EXPECT_PEER_TRANSPORT=P2P"), 1
        )
        self.assertIn("FLAGCX_CI_EXPECT_PEER_TRANSPORT=NET", source)
        self.assertIn("FLAGCX_CI_EXPECT_COLL_MULTICHANNEL=1", source)
        full_heterogeneous = source[
            source.index('FLAGCX_CI_MPI_LABEL="runner BAREX heterogeneous"'):
        ]
        self.assertNotIn("FLAGCX_USE_HETERO_COMM=1", full_heterogeneous)
        self.assertEqual(
            full_heterogeneous.count(
                "env -u FLAGCX_USE_HOST_COMM -u FLAGCX_USE_HETERO_COMM"
            ),
            3,
        )
        default_runner = source[source.index('FLAGCX_CI_MPI_LABEL="runner default"'):]
        default_runner = default_runner[:default_runner.index('FLAGCX_CI_MPI_LABEL="runner BAREX')]
        for variable in (
            "FLAGCX_USE_HETERO_COMM",
            "FLAGCX_CLUSTER_SPLIT_LIST",
            "FLAGCX_MEM_ENABLE",
            "FLAGCX_P2P_TRANSPORT",
            "FLAGCX_P2P_DISABLE",
        ):
            self.assertIn(f"-u {variable}", default_runner)
        self.assertNotIn("-u FLAGCX_VMM_ENABLE", default_runner)

    def test_ppu_perf_uses_supported_collectives_for_each_runner(self):
        ppu_source = (
            REPO_ROOT / ".github/scripts/ci/run_ppu_workload.sh"
        ).read_text()
        source = (
            REPO_ROOT / ".github/scripts/ci/run_host_perf_suite.sh"
        ).read_text()
        homogeneous_operations = (
            "alltoall",
            "alltoallv",
            "sendrecv",
            "allreduce",
            "allgather",
            "reducescatter",
            "broadcast",
            "gather",
            "scatter",
            "reduce",
        )
        heterogeneous_operations = (
            "alltoall",
            "alltoallv",
            "sendrecv",
            "allgather",
            "broadcast",
            "gather",
            "scatter",
        )

        homogeneous_suite = source[source.index("homogeneous)\n"):]
        homogeneous_suite = homogeneous_suite[:homogeneous_suite.index(";;")]
        heterogeneous_suite = source[source.index("heterogeneous)\n"):]
        heterogeneous_suite = heterogeneous_suite[:heterogeneous_suite.index(";;")]
        for operation in homogeneous_operations:
            self.assertIn(operation, homogeneous_suite)
        for operation in heterogeneous_operations:
            self.assertIn(operation, heterogeneous_suite)
        for operation in ("allreduce", "reducescatter", "reduce"):
            self.assertNotIn(operation, heterogeneous_suite)
        self.assertIn('"$perf_runner" homogeneous "$perf_bin"', ppu_source)
        self.assertIn('"$perf_runner" heterogeneous "$perf_bin"', ppu_source)
        self.assertIn("FLAGCX_USE_HETERO_COMM=1", source)
        self.assertIn("FLAGCX_P2P_TRANSPORT=accl", ppu_source)

        self.assertIn("FLAGCX_MEM_ENABLE=1", heterogeneous_suite)
        self.assertNotIn("FLAGCX_CLUSTER_SPLIT_LIST", heterogeneous_suite)

        clean_mode = source[source.index("clean_mode_env=("):]
        clean_mode = clean_mode[:clean_mode.index("mpi_args=(")]
        for variable in (
            "FLAGCX_USE_HETERO_COMM",
            "FLAGCX_CLUSTER_SPLIT_LIST",
            "FLAGCX_MEM_ENABLE",
            "FLAGCX_VMM_ENABLE",
            "FLAGCX_P2P_TRANSPORT",
            "FLAGCX_P2P_DISABLE",
        ):
            self.assertIn(f"-u {variable}", clean_mode)

    def test_ppu_torch_heterogeneous_mode_uses_hybrid_runner(self):
        source = (
            REPO_ROOT / ".github/scripts/ci/run_ppu_workload.sh"
        ).read_text()
        torch_case = source[source.index("  torch-api)\n"):]
        torch_case = torch_case[:torch_case.index("    ;;")]

        self.assertIn("unset FLAGCX_USE_HETERO_COMM", torch_case)
        self.assertNotIn("export FLAGCX_USE_HETERO_COMM=1", torch_case)
        self.assertIn("export FLAGCX_CLUSTER_SPLIT_LIST=2", torch_case)
        self.assertIn("export FLAGCX_MEM_ENABLE=1", torch_case)
        self.assertIn("export FLAGCX_P2P_TRANSPORT=accl", torch_case)

    def test_ppu_jobs_are_present_in_public_workflows(self):
        perf_workflow = (
            REPO_ROOT / ".github/workflows/test.yml"
        ).read_text()
        torch_workflow = (
            REPO_ROOT / ".github/workflows/torch-api-test.yml"
        ).read_text()

        self.assertIn("perf-test-ppu:", perf_workflow)
        self.assertIn("name: perf-test (ppu)", perf_workflow)
        self.assertIn("run_hardware_container.sh\"", perf_workflow)
        self.assertIn("ppu perf", perf_workflow)
        self.assertIn("perf-test-experimental-ppu:", perf_workflow)
        self.assertIn("ppu perf-test-experimental", perf_workflow)
        self.assertIn("PPU_EXPERIMENTAL_RESULT:", perf_workflow)
        self.assertIn("torch-api-test-ppu:", torch_workflow)
        self.assertIn("name: torch-api-test (ppu)", torch_workflow)
        self.assertIn("run_hardware_container.sh\"", torch_workflow)
        self.assertIn("ppu torch-api", torch_workflow)

        # BAREX needs host networking. GitHub Actions job containers reject
        # --network, so the PPU jobs must launch Docker explicitly.
        self.assertNotIn("container", yaml.safe_load(perf_workflow)["jobs"]["perf-test-ppu"])
        self.assertNotIn("container", yaml.safe_load(torch_workflow)["jobs"]["torch-api-test-ppu"])

        container_runner = (
            REPO_ROOT / ".github/scripts/ci/run_hardware_container.sh"
        ).read_text()
        ppu_config = (REPO_ROOT / ".github/configs/ppu.yml").read_text()
        self.assertIn("docker container run --rm", container_runner)
        self.assertNotIn("docker run --rm", container_runner)
        self.assertNotIn("ruby ", container_runner)
        for variable in (
            "FLAGCX_CI_IMAGE",
            "FLAGCX_CI_DOCKER_ARGS",
            "FLAGCX_CI_SET_ENV",
        ):
            self.assertIn(variable, container_runner)
            self.assertIn(variable, perf_workflow)
            self.assertIn(variable, torch_workflow)
        self.assertIn("--network=host", ppu_config)

if __name__ == "__main__":
    unittest.main()
