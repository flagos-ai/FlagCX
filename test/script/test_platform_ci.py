import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
METAX_ENV = REPO_ROOT / ".github/scripts/set_env/metax.sh"


class PlatformCiRegressionTest(unittest.TestCase):
    def test_reference_platform_coverage_is_explicit(self):
        perf_workflow = (REPO_ROOT / ".github/workflows/test.yml").read_text()
        torch_workflow = (
            REPO_ROOT / ".github/workflows/torch-api-test.yml"
        ).read_text()

        hygon_perf = perf_workflow[perf_workflow.index("  perf-test-hygon:") :]
        hygon_perf = hygon_perf[: hygon_perf.index("  perf-test-cuda:")]
        metax_perf = perf_workflow[perf_workflow.index("  perf-test-metax:") :]
        metax_perf = metax_perf[: metax_perf.index("  perf-test-ppu:")]
        hygon_handler = (
            REPO_ROOT / ".github/scripts/ci/run_hygon_workload.sh"
        ).read_text()
        self.assertIn("hygon perf", hygon_perf)
        self.assertIn('"$perf_runner" homogeneous "$perf_bin"', hygon_handler)
        self.assertIn('"$perf_runner" heterogeneous "$perf_bin"', hygon_handler)

        self.assertIn("Perf tests (MetaX uniRunner 8-chip)", metax_perf)
        self.assertIn("run_host_perf_suite.sh heterogeneous", metax_perf)

        hygon_torch = torch_workflow[
            torch_workflow.index("  torch-api-test-hygon:") :
        ]
        hygon_torch = hygon_torch[: hygon_torch.index("  torch-api-test-cuda:")]
        metax_torch = torch_workflow[
            torch_workflow.index("  torch-api-test-metax:") :
        ]
        metax_torch = metax_torch[: metax_torch.index("  torch-api-test-ppu:")]
        self.assertIn("hygon torch-api", hygon_torch)
        self.assertIn("unset FLAGCX_SKIP_HETERO", hygon_handler)
        self.assertNotIn("FLAGCX_USE_HETERO_COMM", metax_torch)

    def test_common_launcher_and_static_rdma_preflight_cover_all_platforms(self):
        unit_workflow = (
            REPO_ROOT / ".github/workflows/unit_tests_common.yml"
        ).read_text()
        unit_runner = (
            REPO_ROOT / ".github/scripts/ci/run_unit_test.sh"
        ).read_text()
        self.assertIn("run_hardware_container.sh\"", unit_workflow)
        self.assertIn('unittest "${{ matrix.suite }}"', unit_workflow)
        self.assertIn("export FLAGCX_DEBUG=INFO", unit_runner)
        self.assertIn("export FLAGCX_DEBUG_SUBSYS=ALL", unit_runner)
        self.assertIn("export FLAGCX_VMM_ENABLE=0", unit_runner)

        for platform in ("cuda", "metax", "hygon", "ppu"):
            source = (
                REPO_ROOT / f".github/scripts/set_env/{platform}.sh"
            ).read_text()
            self.assertIn("rdma_static_preflight.sh", source)
            self.assertIn("flagcx_ci_validate_rdma_static", source)

    def test_perf_collectives_fail_fast_on_flagcx_errors(self):
        perf_dir = REPO_ROOT / "test/perf/host_api"
        operations = (
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

        for operation in operations:
            source = (perf_dir / f"test_{operation}.cpp").read_text()
            self.assertIn("PERF_CHECK(flagcx", source, operation)

    def test_rdma_integration_suites_fail_fast_without_hardware_adaptor(self):
        adaptor_test = (
            REPO_ROOT / "test/unittest/adaptor/test_net_adaptor.cpp"
        ).read_text()
        p2p_test = (
            REPO_ROOT / "test/unittest/p2p/test_p2p_adaptor.cpp"
        ).read_text()
        runner_test = (
            REPO_ROOT / "test/unittest/runner/main_mpi.cpp"
        ).read_text()
        unit_runner = (
            REPO_ROOT / ".github/scripts/ci/run_unit_test.sh"
        ).read_text()
        ppu_env = (
            REPO_ROOT / ".github/scripts/set_env/ppu.sh"
        ).read_text()

        self.assertIn("ASSERT_GT(nDevs_, 0)", adaptor_test)
        device_requirement = p2p_test[
            p2p_test.index("TEST_F(P2pAdaptorTest, DevicesReturnsPositive)") :
        ]
        device_requirement = device_requirement[:
            device_requirement.index("TEST_F(P2pAdaptorTest, InitIsIdempotent)")
        ]
        self.assertIn("ASSERT_EQ(initResult, flagcxSuccess)", device_requirement)
        self.assertRegex(device_requirement, r"(?:ASSERT|EXPECT)_GT\(nDevs, 0\)")
        self.assertNotIn("GTEST_SKIP", device_requirement)

        self.assertIn("FLAGCX_CI_EXPECT_NET_ADAPTOR", runner_test)
        forced_net = unit_runner[
            unit_runner.index('FLAGCX_CI_MPI_LABEL="runner forced NET"') :
        ]
        forced_net = forced_net[:forced_net.index(";;")]
        self.assertIn("FLAGCX_CI_EXPECT_NET_ADAPTOR=IB", forced_net)

        ppu_forced_net = ppu_env[
            ppu_env.index('FLAGCX_CI_MPI_LABEL="runner BAREX forced NET"') :
        ]
        ppu_forced_net = ppu_forced_net[:ppu_forced_net.index("return")]
        self.assertIn("FLAGCX_CI_EXPECT_NET_ADAPTOR=BAREX", ppu_forced_net)

    def test_p2p_read_diagnostics_use_fresh_qp_and_mtu_processes(self):
        unit_runner = (
            REPO_ROOT / ".github/scripts/ci/run_unit_test.sh"
        ).read_text()
        p2p_runner = unit_runner[unit_runner.index("    p2p)") :]
        p2p_runner = p2p_runner[: p2p_runner.index("    rma)")]

        self.assertIn("read_diagnostic_filter", p2p_runner)
        self.assertIn("ReadsWholeRegisteredGpuBuffer", p2p_runner)
        self.assertIn("TwoIndependent2KiBReadsCover4KiBBuffer", p2p_runner)
        self.assertIn("ReadsWholeRegisteredHostBuffer", p2p_runner)
        self.assertIn(
            "FLAGCX_P2P_QPS_PER_CONN=1 FLAGCX_P2P_MTU=4096",
            p2p_runner,
        )
        self.assertIn(
            "FLAGCX_P2P_QPS_PER_CONN=1 FLAGCX_P2P_MTU=2048",
            p2p_runner,
        )
        self.assertIn("single_qp_status", p2p_runner)
        self.assertIn("mtu_2048_status", p2p_runner)

    def test_rma_transport_mode_uses_common_ib_and_p2p_switches(self):
        rma_makefile = (
            REPO_ROOT / "test/unittest/rma/Makefile"
        ).read_text()
        rma_fixture = (
            REPO_ROOT / "test/unittest/rma/rma_test.cpp"
        ).read_text()

        self.assertIn(
            "IPC_ENV := $(HETERO_ENV) -x FLAGCX_IB_DISABLE=1 "
            "-x FLAGCX_P2P_DISABLE=0",
            rma_makefile,
        )
        self.assertIn(
            "NET_ENV := $(HETERO_ENV) -x FLAGCX_IB_DISABLE=0 "
            "-x FLAGCX_P2P_DISABLE=1",
            rma_makefile,
        )
        self.assertIn('std::getenv("FLAGCX_IB_DISABLE")', rma_fixture)
        self.assertIn('std::getenv("FLAGCX_P2P_DISABLE")', rma_fixture)

        for path in (
            REPO_ROOT / "flagcx/core/flagcx_hetero.cc",
            REPO_ROOT / "test/unittest/rma/Makefile",
            REPO_ROOT / "test/unittest/rma/rma_test.cpp",
            REPO_ROOT / ".github/scripts/set_env/metax.sh",
            REPO_ROOT / ".github/scripts/set_env/hygon.sh",
            REPO_ROOT / ".github/scripts/set_env/cuda.sh",
            REPO_ROOT / ".github/scripts/set_env/ppu.sh",
        ):
            source = path.read_text()
            self.assertNotIn("FLAGCX_RMA_FORCE_NET", source, str(path))
            self.assertNotIn("FLAGCX_RMA_TEST_REQUIRE_IPC", source, str(path))

        for platform in ("cuda", "metax", "hygon", "ppu"):
            source = (
                REPO_ROOT / f".github/scripts/set_env/{platform}.sh"
            ).read_text()
            configure = source[source.index("flagcx_ci_configure_suite() {") :]
            configure = configure[: configure.index("\n}\n") + 3]
            self.assertNotIn(
                "FLAGCX_P2P_DISABLE=1", configure, platform
            )


class MetaXEnvironmentTest(unittest.TestCase):
    def run_metax_shell(self, body, *, extra_env=None):
        environment = os.environ.copy()
        if extra_env:
            environment.update(extra_env)
        return subprocess.run(
            ["bash", "-c", 'set -euo pipefail; source "$1"; eval "$2"',
             "bash", str(METAX_ENV), body],
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_selector_uses_linux_rdma_sysfs_path(self):
        source = METAX_ENV.read_text()
        selector = source[source.index("flagcx_ci_select_rdma() {") :]
        selector = selector[: selector.index("flagcx_ci_validate_rdma() {")]

        self.assertIn("/sys/class/infiniband/bnxt_roce*", selector)
        self.assertIn("/sys/class/infiniband/bnxt_re_bond*", selector)
        self.assertIn("export FLAGCX_IB_HCA", selector)

    def test_prepare_does_not_select_rdma_hca(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            for command in ("mpirun", "mxcc"):
                executable = fake_bin / command
                executable.write_text("#!/usr/bin/env bash\nexit 0\n")
                executable.chmod(0o755)

            result = self.run_metax_shell(
                'unset FLAGCX_IB_HCA; flagcx_ci_prepare symmem >/dev/null; '
                'printf "%s\\n" "${FLAGCX_IB_HCA-<unset>}"',
                extra_env={"PATH": f"{fake_bin}:{os.environ['PATH']}"},
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), "<unset>")

    def test_validator_propagates_static_preflight_failure(self):
        result = self.run_metax_shell(
            'flagcx_ci_select_rdma() { '
            'export FLAGCX_IB_HCA=bnxt_roce0; '
            'FLAGCX_CI_METAX_RDMA_PATTERN=/sys/class/infiniband/bnxt_roce\\*; '
            '}; '
            'flagcx_ci_validate_rdma_static() { return 17; }; '
            'unset FLAGCX_IB_HCA; set +e; '
            'flagcx_ci_validate_rdma rma; status=$?; set -e; '
            'printf "%s\\n" "$status"'
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "17")


if __name__ == "__main__":
    unittest.main()
