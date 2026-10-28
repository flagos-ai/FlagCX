import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
METAX_ENV = REPO_ROOT / ".github/scripts/set_env/metax.sh"
HYGON_ENV = REPO_ROOT / ".github/scripts/set_env/hygon.sh"


class PlatformCiRegressionTest(unittest.TestCase):
    def test_gpu_ci_world_uses_four_ranks(self):
        for platform in ("cuda", "metax", "hygon", "ppu"):
            with self.subTest(platform=platform):
                source = (
                    REPO_ROOT / f".github/scripts/set_env/{platform}.sh"
                ).read_text()
                self.assertIn("FLAGCX_CI_INTRA_NP=4", source)
                self.assertIn("FLAGCX_CI_NODE_NP=2", source)
                self.assertIn("FLAGCX_CI_RUNNER_NP=4", source)
                self.assertIn("export NP=4", source)

        perf_workflow = (REPO_ROOT / ".github/workflows/test.yml").read_text()
        torch_script = (REPO_ROOT / "test/script/torch_api_test.sh").read_text()
        device_makefile = (
            REPO_ROOT / "test/unittest/device_api/Makefile"
        ).read_text()
        self.assertNotIn("mpirun -np 8", perf_workflow)
        self.assertNotIn("-n 8", device_makefile)
        self.assertNotIn("-n 4 $(HETERO_ENV)", device_makefile)
        self.assertIn("--nproc_per_node 4", torch_script)
        self.assertIn("-n 2 $(HETERO_ENV)", device_makefile)

    def test_device_kernel_sources_use_the_selected_platform(self):
        for platform, adaptor in (
            ("nvidia", "nvidia_adaptor.h"),
            ("du", "du_adaptor.h"),
            ("iluvatar", "iluvatar_adaptor.h"),
        ):
            suffix = "hip" if platform == "du" else "cu"
            for kernel in (f"device_api.{suffix}", f"device_ir.{suffix}"):
                source = (REPO_ROOT / "test/kernel" / platform / kernel).read_text()
                self.assertIn(f'#include "{adaptor}"', source)
                self.assertNotIn('../nvidia/', source)
                self.assertNotIn('USE_ILUVATAR_ADAPTOR', source)

        corex_api = (REPO_ROOT / "test/kernel/iluvatar/device_api.cu").read_text()
        self.assertIn("flagcxIluvatarAtomicContractKernel", corex_api)
        self.assertIn("flagcxIluvatarUnsupportedCoopKernel", corex_api)

    def test_hygon_device_api_build_and_suite_configuration(self):
        suite_dir = REPO_ROOT / "test/unittest/device_api"
        result = subprocess.run(
            [
                "make", "-n", "-B", "-C", str(suite_dir), "mpi",
                "USE_DU=1", "USE_SHCA=1",
                "DEVICE_HOME=/opt/dtk",
                "CCL_HOME=/opt/dtk",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        for target in (
            "test_device_api_intra", "test_device_api_inter",
            "test_device_ir_intra", "test_device_ir_inter",
            "test_device_ir_unified_intra", "test_device_ir_unified_inter",
        ):
            self.assertIn(target, result.stdout)
        self.assertIn("-DUSE_DU_ADAPTOR", result.stdout)
        for kernel in ("device_api.hip", "device_ir.hip"):
            compile_line = next(
                line for line in result.stdout.splitlines()
                if "/bin/hipcc " in line and kernel in line
            )
            self.assertNotIn("-Xcompiler -fPIC", compile_line)
        for target in (
            "test_device_api_intra", "test_device_api_inter",
            "test_device_ir_intra", "test_device_ir_inter",
            "test_device_ir_unified_intra", "test_device_ir_unified_inter",
        ):
            compile_line = next(
                line for line in result.stdout.splitlines()
                if line.startswith("/opt/dtk/bin/hipcc ")
                and f"/obj/{target}.o " in line
            )
            self.assertIn(" -c ", compile_line)
            self.assertIn(f"{target}.cpp", compile_line)
            link_line = next(
                line for line in result.stdout.splitlines()
                if line.startswith("/opt/dtk/bin/hipcc ") and f"/bin/{target} " in line
            )
            self.assertIn("-no-pie", link_line)
            self.assertIn(f"/obj/{target}.o", link_line)
            self.assertNotIn(".cpp", link_line)

        rma_result = subprocess.run(
            [
                "make", "-n", "-B", "-C", str(REPO_ROOT / "test/unittest/rma"),
                "USE_DU=1", "USE_SHCA=1",
                "DEVICE_HOME=/opt/dtk", "CCL_HOME=/opt/dtk",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        for source in (
            "test_rma_post_result.cpp",
            "test_signal_registration_ownership.cpp",
        ):
            compile_line = next(
                line for line in rma_result.stdout.splitlines()
                if source in line and " -c " in line
            )
            self.assertTrue(compile_line.startswith("g++ "))
            self.assertNotIn("-fgpu-rdc", compile_line)
        unit_link_line = next(
            line for line in rma_result.stdout.splitlines()
            if "/bin/rma_unit_tests " in line
        )
        self.assertTrue(unit_link_line.startswith("g++ "))

        kernel_result = subprocess.run(
            [
                "make", "-n", "-B", "-C", str(REPO_ROOT / "test/unittest/kernel"),
                "all", "USE_DU=1", "USE_SHCA=1",
                "DEVICE_HOME=/opt/dtk", "CCL_HOME=/opt/dtk",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        kernel_unit_link = next(
            line for line in kernel_result.stdout.splitlines()
            if "/bin/kernel_unit_tests " in line
        )
        self.assertTrue(kernel_unit_link.startswith("g++ "))
        kernel_mpi_link = next(
            line for line in kernel_result.stdout.splitlines()
            if "/bin/kernel_mpi_tests " in line
        )
        self.assertTrue(kernel_mpi_link.startswith("/opt/dtk/bin/hipcc "))
        self.assertIn("-fgpu-rdc", kernel_mpi_link)
        self.assertIn("-no-pie", kernel_mpi_link)
        self.assertIn("/du/build/obj/device_api.o", kernel_mpi_link)
        self.assertNotIn(".cpp", kernel_mpi_link)

        perf_result = subprocess.run(
            [
                "make", "-n", "-B", "-C", str(REPO_ROOT / "test/perf/device_api"),
                "all", "USE_DU=1", "USE_SHCA=1",
                "DEVICE_HOME=/opt/dtk", "CCL_HOME=/opt/dtk",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        for source in sorted((REPO_ROOT / "test/perf/device_api").glob("test_*.cpp")):
            target = source.stem.removeprefix("test_")
            compile_line = next(
                line for line in perf_result.stdout.splitlines()
                if f"/obj/perf_{target}.o " in line and " -c " in line
            )
            self.assertTrue(compile_line.startswith("g++ "))
            link_line = next(
                line for line in perf_result.stdout.splitlines()
                if f"/bin/perf_{target} " in line
            )
            self.assertTrue(link_line.startswith("/opt/dtk/bin/hipcc "))
            self.assertIn("-fgpu-rdc", link_line)
            self.assertIn("-no-pie", link_line)
            self.assertIn(f"/obj/perf_{target}.o", link_line)
            self.assertIn("/du/build/obj/device_api.o", link_line)
            self.assertNotIn(".cpp", link_line)

        for suite in ("device_api", "device_api_unified_ir"):
            configured = subprocess.run(
                [
                    "bash", "-c",
                    'source "$1"; flagcx_ci_configure_suite "$2"; '
                    'flagcx_ci_build_suite_override "$2" ""; '
                    'build_handled=$FLAGCX_CI_BUILD_SUITE_OVERRIDE_HANDLED; '
                    'flagcx_ci_run_suite_override "$2" ""; '
                    'printf "%s %s %s %s %s %s %s" "$FLAGCX_CI_NODE_NP" '
                    '"$FLAGCX_CI_INTRA_NP" "$HIP_VISIBLE_DEVICES" '
                    '"$FLAGCX_IB_HCA" '
                    '"${FLAGCX_CI_PROJECT_MAKE_ARGS[*]}" '
                    '"$build_handled" "$FLAGCX_CI_RUN_SUITE_OVERRIDE_HANDLED"',
                    "bash", str(HYGON_ENV), suite,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("2 4 0,1,6,7 shca_0,shca_3", configured.stdout)
            self.assertIn("COMPILE_KERNEL=1", configured.stdout)
            self.assertTrue(configured.stdout.endswith(" 0 0"))

    def _run_platform_matrix(
        self, event_name, changed_files="", configs_dir=None
    ):
        script = REPO_ROOT / ".github/scripts/ci/resolve_ci_matrix.rb"
        config_path = configs_dir or REPO_ROOT / ".github/configs"
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            context_path = temp_path / "context.json"
            output_path = temp_path / "github-output"
            context_path.write_text(
                json.dumps(
                    {
                        "event_name": event_name,
                        "files": changed_files.splitlines() if changed_files else [],
                        "repository": "owner/FlagCX",
                        "ref": "HEAD",
                    }
                )
            )
            env = os.environ.copy()
            env["GITHUB_OUTPUT"] = str(output_path)
            subprocess.run(
                ["ruby", str(script), str(config_path), str(context_path)],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            outputs = dict(
                line.split("=", 1) for line in output_path.read_text().splitlines()
            )
            return json.loads(outputs["platforms"])

    def test_unified_ci_filters_platform_changes_and_preserves_shared_fallback(self):
        all_platforms = ["cuda", "hygon", "iluvatar", "metax", "ppu"]
        self.assertEqual(
            self._run_platform_matrix(
                "pull_request", "flagcx/adaptor/device/cuda_adaptor.cc"
            ),
            ["cuda"],
        )
        self.assertEqual(
            self._run_platform_matrix(
                "pull_request", "flagcx/adaptor/device/duhip_adaptor.cc"
            ),
            ["hygon"],
        )
        for changed_files in (
            "flagcx/flagcx.cc",
            ".github/workflows/ci.yml",
            "flagcx/adaptor/device/cuda_adaptor.cc\nflagcx/flagcx.cc",
        ):
            self.assertEqual(
                self._run_platform_matrix("pull_request", changed_files),
                all_platforms,
            )
        self.assertEqual(
            self._run_platform_matrix("push", "flagcx/flagcx.cc"),
            all_platforms,
        )

    def test_unified_ci_uses_enabled_registry_for_matrix(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            configs_dir = Path(temp_dir)
            (configs_dir / "platforms.yml").write_text(
                "platforms:\n"
                "  cuda:\n"
                "    enabled: true\n"
                "  hygon:\n"
                "    enabled: false\n"
            )
            (configs_dir / "cuda.yml").write_text(
                "hardware_name: cuda\n"
                "display_name: CUDA\n"
                "paths_watch:\n"
                "  - flagcx/adaptor/device/cuda_adaptor.cc\n"
            )
            (configs_dir / "hygon.yml").write_text(
                "hardware_name: hygon\n"
                "display_name: Hygon\n"
                "paths_watch:\n"
                "  - flagcx/adaptor/device/duhip_adaptor.cc\n"
            )
            self.assertEqual(
                self._run_platform_matrix(
                    "push", configs_dir=configs_dir
                ),
                ["cuda"],
            )
            self.assertEqual(
                self._run_platform_matrix(
                    "pull_request",
                    "flagcx/adaptor/device/duhip_adaptor.cc",
                    configs_dir,
                ),
                ["cuda"],
            )

    def test_main_unit_tests_have_one_automatic_entry_point(self):
        unit_workflow = (REPO_ROOT / ".github/workflows/unit-test.yml").read_text()
        unit_trigger = unit_workflow[: unit_workflow.index("\njobs:")]

        self.assertIn("      - main", unit_trigger)
        self.assertIn("      - devops", unit_trigger)
        self.assertIn("  pull_request:", unit_trigger)
        self.assertIn("  issue_comment:", unit_trigger)
        self.assertIn("name: FlagCX CI", unit_trigger)

        pipeline = unit_workflow[unit_workflow.index("\n  unit_tests:") :]
        self.assertIn("needs: prepare", pipeline)
        self.assertIn("needs.prepare.outputs.has_platforms == 'true'", pipeline)

    def test_hygon_builds_ibrc_with_shca_abi(self):
        hygon_env = HYGON_ENV.read_text()
        makefile = (REPO_ROOT / "Makefile").read_text()
        compat = (
            REPO_ROOT / "flagcx/service/include/ibv_compat.h"
        ).read_text()

        self.assertIn("USE_SHCA=1", hygon_env)
        self.assertIn("USE_SHCA ?= 0", makefile)
        self.assertIn("NET_ADAPTOR_FLAG += -DUSE_SHCA", makefile)
        self.assertIn("<infiniband/verbs.h>", compat)
        self.assertIn("<infiniband/shca_17b_types.h>", compat)

    def test_shca_ibrc_excludes_ud_ah_srq_provider_operations(self):
        common_retrans = (
            REPO_ROOT / "flagcx/adaptor/net/ib_retrans.cc"
        ).read_text()
        ud_retrans = (
            REPO_ROOT / "flagcx/adaptor/net/ib_retrans_ud.cc"
        ).read_text()
        symbols = (
            REPO_ROOT / "flagcx/service/ibvsymbols.cc"
        ).read_text()
        ibvwrap = (
            REPO_ROOT / "flagcx/service/include/ibvwrap.h"
        ).read_text()

        self.assertNotIn("ops.create_ah", common_retrans)
        self.assertNotIn("ops.destroy_ah", common_retrans)
        self.assertNotIn("flagcxWrapIbvPostSrqRecv", common_retrans)
        self.assertIn("#ifndef USE_SHCA", ud_retrans)
        self.assertNotIn("defined(USE_IBUC)", ud_retrans)
        self.assertIn(
            "flagcxIbRetransUdSupported(void) { return false; }", ud_retrans
        )
        self.assertIn("ibvSymbols->ibv_internal_create_ah = NULL", symbols)
        self.assertIn("ibvSymbols->ibv_internal_destroy_ah = NULL", symbols)
        self.assertNotIn("ops.create_ah", ibvwrap)
        self.assertNotIn("ops.destroy_ah", ibvwrap)
        self.assertIn(
            'LOAD_SYM(ibvhandle, "ibv_create_ah"', symbols
        )
        self.assertIn(
            'LOAD_SYM(ibvhandle, "ibv_destroy_ah"', symbols
        )

    def test_shca_scope_does_not_extend_ibuc(self):
        ibuc = (
            REPO_ROOT / "flagcx/adaptor/net/ibuc_adaptor.cc"
        ).read_text()
        ud_retrans = (
            REPO_ROOT / "flagcx/adaptor/net/ib_retrans_ud.cc"
        ).read_text()

        self.assertNotIn("USE_SHCA", ibuc)
        self.assertNotIn("flagcxIbPortLid", ibuc)
        self.assertNotIn("flagcxIbSetAhDlid", ibuc)
        self.assertNotIn("flagcxIbUseGlobalRoute", ibuc)
        self.assertNotIn("defined(USE_IBUC)", ud_retrans)

    def test_hygon_service_checks_ibuc_with_standard_verbs_abi(self):
        service_dir = REPO_ROOT / "test/unittest/service"
        result = subprocess.run(
            [
                "make",
                "-n",
                "-C",
                str(service_dir),
                "compile-ibuc",
                "USE_DU=1",
                "USE_SHCA=1",
                "DEVICE_HOME=/opt/dtk",
                "CCL_HOME=/opt/dtk",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        compile_command = next(
            line
            for line in result.stdout.splitlines()
            if "ibuc_adaptor.cc -fsyntax-only" in line
        )

        self.assertIn("-DUSE_IBUC", compile_command)
        self.assertNotIn("-DUSE_SHCA", compile_command)

    def test_shca_ibrc_uses_gid_global_route(self):
        ibrc = (
            REPO_ROOT / "flagcx/adaptor/net/ibrc_adaptor.cc"
        ).read_text()
        route_selector = (
            REPO_ROOT / "flagcx/adaptor/include/ib_common.h"
        ).read_text()
        route_selector = route_selector[
            route_selector.index("flagcxIbUseGlobalRoute") :
        ]
        self.assertIn("#ifdef USE_SHCA", route_selector)
        self.assertIn("return true", route_selector)
        self.assertNotIn("IB_SHCA_USE_GID", ibrc)
        self.assertIn("qpAttr.ah_attr.is_global = 1", ibrc)
        self.assertIn("qpAttr.ah_attr.grh.dgid.global.subnet_prefix", ibrc)
        self.assertIn("flagcxIbSetAhDlid(&qpAttr.ah_attr, info->lid)", ibrc)
        self.assertIn("flagcxIbUseGlobalRoute(devInfo->linkLayer)", ibrc)
        self.assertIn("flagcxIbRtrQp(qp->qp", ibrc)

    def test_hygon_rdma_suites_use_connected_shca_fabric(self):
        hygon_env = HYGON_ENV.read_text()

        self.assertIn(
            "FLAGCX_CI_HYGON_CONNECTED_HCAS=shca_0,shca_3", hygon_env
        )
        self.assertIn(
            "FLAGCX_CI_HYGON_TWO_GPU_DEVICES=0,7", hygon_env
        )
        self.assertIn(
            "FLAGCX_CI_HYGON_FOUR_GPU_DEVICES=0,1,6,7", hygon_env
        )
        self.assertIn("FLAGCX_CI_PXN_VISIBLE_DEVICES=2,0,6,7", hygon_env)
        self.assertIn("FLAGCX_CI_PXN_HCAS=shca_0,shca_3", hygon_env)
        self.assertIn("FLAGCX_CI_RUNNER_NP=4", hygon_env)
        self.assertIn("export NP=4", hygon_env)

        configure = hygon_env[hygon_env.index("flagcx_ci_configure_suite() {") :]
        configure = configure[: configure.index("flagcx_ci_prepare() {")]
        self.assertIn("p2p|rma)", configure)
        self.assertIn(
            'export HIP_VISIBLE_DEVICES="$FLAGCX_CI_HYGON_TWO_GPU_DEVICES"',
            configure,
        )
        self.assertIn(
            'export FLAGCX_IB_HCA="$FLAGCX_CI_HYGON_CONNECTED_HCAS"',
            configure,
        )
        self.assertIn('runner|symmem)', configure)
        self.assertIn(
            'export HIP_VISIBLE_DEVICES="$FLAGCX_CI_HYGON_FOUR_GPU_DEVICES"',
            configure,
        )
        self.assertNotIn("FLAGCX_CI_RUNNER_NP=8", configure)

    def test_cuda_runs_ibuc_after_ibrc_in_adaptor_suite(self):
        cuda_config = (REPO_ROOT / ".github/configs/cuda.yml").read_text()
        cuda_env = (
            REPO_ROOT / ".github/scripts/set_env/cuda.sh"
        ).read_text()
        self.assertIn("  - adaptor", cuda_config)
        self.assertNotIn("  - ibuc", cuda_config)
        self.assertNotIn("ibuc)", cuda_env)
        self.assertIn("FLAGCX_CI_ENABLE_IBUC=1", cuda_env)

        metax_env = METAX_ENV.read_text()
        self.assertNotIn("FLAGCX_CI_ENABLE_IBUC=1", metax_env)

        ppu_env = (REPO_ROOT / ".github/scripts/set_env/ppu.sh").read_text()
        self.assertNotIn("FLAGCX_CI_ENABLE_IBUC=1", ppu_env)

        unit_runner = (
            REPO_ROOT / ".github/scripts/ci/run_unit_test.sh"
        ).read_text()
        adaptor_case = unit_runner[unit_runner.index("    adaptor)") :]
        adaptor_case = adaptor_case[: adaptor_case.index("    core|service)")]
        self.assertLess(
            adaptor_case.index('FLAGCX_CI_TEST_LABEL="$SUITE unit tests"'),
            adaptor_case.index('FLAGCX_CI_TEST_LABEL="IBUC net adaptor tests"'),
        )
        self.assertIn('BUILDDIR="$ibuc_project_build"', adaptor_case)
        self.assertIn('BUILDDIR="$ibuc_test_build"', adaptor_case)
        self.assertIn('FLAGCX_LIB="$ibuc_project_build/lib"', adaptor_case)
        self.assertIn(
            'LD_LIBRARY_PATH="$ibuc_project_build/lib:$LD_LIBRARY_PATH"',
            adaptor_case,
        )
        self.assertIn("USE_IBUC=1", adaptor_case)
        self.assertIn("FLAGCX_CI_ENABLE_IBUC", adaptor_case)
        self.assertIn('env "${FLAGCX_CI_IBUC_ENV[@]}"', adaptor_case)
        self.assertIn("IbucRetransmissionTest.*", adaptor_case)
        self.assertIn("FLAGCX_IB_QPS_PER_CONNECTION=2", adaptor_case)
        self.assertIn("FLAGCX_IBUC_SPLIT_DATA_ON_QPS=1", adaptor_case)
        self.assertIn("FLAGCX_CI_EXPECT_NET_ADAPTOR=IBUC", unit_runner)
        self.assertNotIn('FLAGCX_CI_MPI_LABEL="IBUC forced NET runner"', unit_runner)

        hygon_env = HYGON_ENV.read_text()
        self.assertNotIn("FLAGCX_CI_ENABLE_IBUC=1", hygon_env)
        self.assertNotIn("FLAGCX_CI_IBUC_ENV=(", hygon_env)

    def test_ibuc_retransmission_is_separate_from_data_receive_queues(self):
        common = (
            REPO_ROOT / "flagcx/adaptor/include/ib_common.h"
        ).read_text()
        common_impl = (
            REPO_ROOT / "flagcx/adaptor/net/ib_common.cc"
        ).read_text()
        ibuc = (
            REPO_ROOT / "flagcx/adaptor/net/ibuc_adaptor.cc"
        ).read_text()
        self.assertIn("#define FLAGCX_IB_SRQ_SIZE 1024", common)
        self.assertIn("#define FLAGCX_IBUC_RETRANS_RECV_DEPTH 16", common)
        self.assertIn("retransQpn", common)
        self.assertIn("flagcxIbucPostDataRecv", ibuc)
        self.assertIn("MAX_REQUESTS; credit++", ibuc)
        self.assertIn(
            "flagcxIbucAbortAccept(lComm, rComm, postResult)", ibuc
        )
        self.assertIn("flagcxIbucPostRetransRecv", ibuc)
        self.assertIn("flagcxIbucCreateQpWithTypeCq", ibuc)
        self.assertIn("&commDev->retransCq", ibuc)
        self.assertIn("flagcxIbucPostAckRecv", ibuc)
        self.assertIn("flagcxIbucSendAckRc", ibuc)
        ack_sender = ibuc[
            ibuc.index("static flagcxResult_t flagcxIbucSendAckRc") :
            ibuc.index("static flagcxResult_t flagcxIbucAckSequence")
        ]
        self.assertIn("IBV_SEND_INLINE | IBV_SEND_SIGNALED", ack_sender)
        self.assertIn("pendingDataEvents", ibuc)
        self.assertIn("target->dataEvents[devIndex]--", ibuc)
        self.assertIn("FLAGCX_IBUC_RETRANS_RECV_DEPTH", ibuc)
        self.assertNotIn("flagcxIbCreateSrq(", ibuc)

        fifo = common[
            common.index("struct flagcxIbSendFifo {") :
            common.index("struct flagcxIbRequest {")
        ]
        metadata = common[
            common.index("struct flagcxIbConnectionMetadata {") :
            common.index("struct flagcxIbNetCommDevBase {")
        ]
        send_comm_dev = common[
            common.index("struct flagcxIbSendCommDev {") :
            common.index("struct alignas(32) flagcxIbNetCommBase {")
        ]
        send_comm = common[
            common.index("struct flagcxIbSendComm {") :
            common.index("struct flagcxIbGpuFlush {")
        ]
        recv_comm_dev = common[
            common.index("struct alignas(16) flagcxIbRecvCommDev {") :
            common.index("struct alignas(32) flagcxIbRecvComm {")
        ]
        self.assertNotIn("#ifdef USE_IBUC", fifo)
        self.assertNotIn("#ifdef USE_IBUC", metadata)
        self.assertNotIn("#ifdef USE_IBUC", send_comm_dev)
        self.assertNotIn("#ifdef USE_IBUC", send_comm)
        self.assertNotIn("#ifdef USE_IBUC", recv_comm_dev)
        self.assertIn("struct flagcxIbQp retransQp", send_comm_dev)
        self.assertIn("struct ibv_cq *retransCq", send_comm_dev)
        self.assertIn("struct ibv_mr *retransHdrMr", send_comm_dev)
        self.assertIn("bool retransUsesRc", send_comm)
        self.assertIn("struct flagcxIbQp retransQp", recv_comm_dev)
        self.assertNotIn("#ifdef USE_IBUC", common_impl)
        self.assertLess(fifo.index("requestSlot"), fifo.index("uint64_t idx"))
        self.assertLess(fifo.index("generation"), fifo.index("uint64_t idx"))
        self.assertIn("sizeof(struct flagcxIbSendFifo) == 64", common)
        self.assertIn("offsetof(struct flagcxIbSendFifo, idx) == 56", common)

    def test_automatic_hardware_ci_covers_all_platforms_and_suites(self):
        matrix_loader = REPO_ROOT / ".github/scripts/ci/load_platform_matrix.rb"
        result = subprocess.run(
            [
                "ruby",
                str(matrix_loader),
                str(REPO_ROOT / ".github/configs"),
                "all",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.stdout.strip(),
            '{"include":[{"platform":"cuda","display_name":"CUDA Tests"},'
            '{"platform":"hygon","display_name":"Hygon DCU Tests"},'
            '{"platform":"iluvatar","display_name":"Iluvatar CoreX Tests"},'
            '{"platform":"metax","display_name":"MetaX Tests"},'
            '{"platform":"ppu","display_name":"T-Head PPU Tests"}]}',
        )
        unit_workflow = (
            REPO_ROOT / ".github/workflows/unit-test.yml"
        ).read_text()
        self.assertIn(
            "EVENT_SELECTION: ${{ inputs.platform || 'all' }}",
            unit_workflow,
        )
        self.assertNotIn("pull_request.number == 643", unit_workflow)
        self.assertNotIn("pull_request.number == 648", unit_workflow)
        self.assertNotIn("pull_request.number == 658", unit_workflow)
        self.assertNotIn("FLAGCX_CI_PR_NUMBER", unit_workflow)

        expected_suites = {
            "cuda": [
                "adaptor",
                "core",
                "device_api",
                "device_api_host",
                "device_api_unified_ir",
                "p2p",
                "rma",
                "runner",
                "service",
                "symmem",
            ],
            "hygon": [
                "adaptor",
                "core",
                "device_api",
                "device_api_host",
                "device_api_unified_ir",
                "p2p",
                "rma",
                "runner",
                "service",
                "symmem",
            ],
            "iluvatar": [
                "adaptor",
                "core",
                "device_api",
                "device_api_host",
                "device_api_unified_ir",
                "p2p",
                "rma",
                "runner",
                "service",
                "symmem",
            ],
            "metax": [
                "adaptor",
                "core",
                "p2p",
                "rma",
                "runner",
                "service",
                "symmem",
            ],
            "ppu": [
                "adaptor",
                "core",
                "p2p",
                "rma",
                "runner",
                "service",
                "symmem",
            ],
        }
        for platform, expected in expected_suites.items():
            config = (
                REPO_ROOT / f".github/configs/{platform}.yml"
            ).read_text()
            suites = config[config.index("unit_test_suites:") :]
            self.assertEqual(
                [
                    line.strip().removeprefix("- ")
                    for line in suites.splitlines()[1:]
                    if line.strip().startswith("- ")
                ],
                expected,
                platform,
            )

        for workflow_name in (
            "test.yml",
            "torch-api-test.yml",
            "format-check.yml",
        ):
            workflow = (
                REPO_ROOT / f".github/workflows/{workflow_name}"
            ).read_text()
            trigger = workflow[: workflow.index("\njobs:")]
            self.assertIn("workflow_call:", trigger)
            self.assertNotIn("pull_request.number != 648", workflow)
            self.assertNotIn("pull_request.number != 658", workflow)

    def test_reference_platform_coverage_is_explicit(self):
        perf_workflow = (REPO_ROOT / ".github/workflows/test.yml").read_text()
        torch_workflow = (
            REPO_ROOT / ".github/workflows/torch-api-test.yml"
        ).read_text()
        hygon_handler = (
            REPO_ROOT / ".github/scripts/ci/run_hygon_workload.sh"
        ).read_text()
        metax_handler = (
            REPO_ROOT / ".github/scripts/ci/run_metax_workload.sh"
        ).read_text()
        self.assertIn("hygon perf", perf_workflow)
        self.assertIn('"$perf_runner" homogeneous "$perf_bin"', hygon_handler)
        self.assertIn('"$perf_runner" heterogeneous "$perf_bin"', hygon_handler)
        self.assertIn("metax perf", perf_workflow)
        self.assertIn('run_host_perf_suite.sh" heterogeneous "$perf_bin"', metax_handler)

        self.assertIn("hygon torch-api", torch_workflow)
        self.assertIn("unset FLAGCX_SKIP_HETERO", hygon_handler)
        self.assertIn("metax torch-api", torch_workflow)
        self.assertNotIn("export FLAGCX_USE_HETERO_COMM", metax_handler)

    def test_perf_and_torch_jobs_use_platform_configured_images(self):
        perf_workflow = (REPO_ROOT / ".github/workflows/test.yml").read_text()
        torch_workflow = (
            REPO_ROOT / ".github/workflows/torch-api-test.yml"
        ).read_text()
        for workflow, workload in (
            (perf_workflow, "perf"),
            (torch_workflow, "torch-api"),
        ):
            self.assertNotIn("container:", workflow)
            self.assertNotIn("image: harbor.baai.ac.cn", workflow)
            for platform in ("cuda", "metax", "hygon", "ppu"):
                with self.subTest(workload=workload, platform=platform):
                    self.assertIn(
                        f"ruby .github/scripts/ci/load_platform_config.rb {platform}",
                        workflow,
                    )
                    self.assertIn(
                        f"FLAGCX_CI_IMAGE: "
                        f"${{{{ needs.load-hardware-container-configs.outputs.{platform}_ci_image }}}}",
                        workflow,
                    )
                    self.assertIn(f"{platform} {workload}", workflow)
                    self.assertIn(
                        f"runs-on: "
                        f"${{{{ fromJSON(needs.load-hardware-container-configs.outputs.{platform}_runs_on) }}}}",
                        workflow,
                    )

        self.assertIn("  perf-test:\n", perf_workflow)
        self.assertIn("cuda perf-legacy", perf_workflow)
        self.assertNotIn("cuda_legacy_perf_runs_on", perf_workflow)
        legacy_perf_job = perf_workflow.split("  perf-test:\n", 1)[1].split(
            "\n  perf-test-hygon:", 1
        )[0]
        self.assertIn(
            "runs-on: ${{ fromJSON(needs.load-hardware-container-configs.outputs.cuda_runs_on) }}",
            legacy_perf_job,
        )
        self.assertIn("timeout-minutes: 240", legacy_perf_job)
        self.assertNotIn("\n  torch-api-test:\n", torch_workflow)

        for platform in ("cuda", "metax", "hygon", "ppu"):
            handler = (
                REPO_ROOT / f".github/scripts/ci/run_{platform}_workload.sh"
            ).read_text()
            self.assertIn('source "$project_root/$FLAGCX_CI_SET_ENV"', handler)
        cuda_handler = (
            REPO_ROOT / ".github/scripts/ci/run_cuda_workload.sh"
        ).read_text()
        self.assertIn(
            'source "$project_root/.github/scripts/ci/run_cuda_legacy_perf.sh"',
            cuda_handler,
        )

    def test_p2p_perf_keeps_legacy_write_and_runs_shared_read(self):
        perf_workflow = (REPO_ROOT / ".github/workflows/test.yml").read_text()
        legacy_perf = (
            REPO_ROOT / ".github/scripts/ci/run_cuda_legacy_perf.sh"
        ).read_text()
        self.assertIn("cuda perf-legacy", perf_workflow)
        self.assertIn('run_host_perf_suite.sh" homogeneous "$perf_bin"', legacy_perf)
        self.assertIn('"$perf_bin/perf_allreduce"', legacy_perf)
        self.assertIn("for operation in put get", legacy_perf)
        self.assertIn('"$perf_bin/perf_$operation"', legacy_perf)
        self.assertIn('"$device_api_bin/perf_allreduce_intranode"', legacy_perf)
        self.assertIn('"$device_api_bin/perf_internode_onesided"', legacy_perf)
        self.assertIn("FLAGCX_P2P_PERF_OP=write", legacy_perf)
        self.assertIn("USE_SHARED_P2P_ENGINE=1", legacy_perf)
        self.assertIn('FLAGCX_LIB="$shared_project_build/lib"', legacy_perf)
        self.assertIn("FLAGCX_P2P_PERF_OP=read", legacy_perf)
        self.assertIn("run_mpi_with_timeout.sh", legacy_perf)
        self.assertNotIn("mpirun ", legacy_perf)
        self.assertIn('"$shared_test_build/bin/perf_p2p_engine"', legacy_perf)
        self.assertIn('LD_LIBRARY_PATH="$shared_project_build/lib:$LD_LIBRARY_PATH"', legacy_perf)
        self.assertNotIn("FLAGCX_GDR_READ_REQUIRES_FLUSH=0", legacy_perf)

    def test_rma_visibility_conformance_covers_supported_routes_and_policies(self):
        unit_runner = (
            REPO_ROOT / ".github/scripts/ci/run_unit_test.sh"
        ).read_text()
        rma_makefile = (REPO_ROOT / "test/unittest/rma/Makefile").read_text()
        rma_case = unit_runner[unit_runner.index("    rma)") :]
        rma_case = rma_case[: rma_case.index("    runner)")]

        self.assertIn("gdr_visibility.o", rma_makefile)
        self.assertIn("GDR_VISIBILITY_BUILD_DIR", rma_makefile)
        self.assertIn("BUILDDIR=$(GDR_VISIBILITY_BUILD_DIR)", rma_makefile)
        self.assertIn("$(KERNEL_INCLUDE)", rma_makefile)
        self.assertIn("RMA_VMM_ENABLE ?= 0", rma_makefile)
        self.assertIn("NET_FILTER ?= *", rma_makefile)
        for platform in ("metax)", "hygon)", "ppu)", "cuda)"):
            self.assertIn(platform, rma_case)

        du_kernel_makefile = (REPO_ROOT / "test/kernel/du/Makefile").read_text()
        visibility_platform = rma_makefile[
            rma_makefile.index("VISIBILITY_PLATFORM :=") :
            rma_makefile.index("ifneq ($(strip $(VISIBILITY_PLATFORM))")
        ]
        self.assertNotIn("$(USE_DU)", visibility_platform)
        self.assertNotIn("gdr_visibility.o", du_kernel_makefile)
        self.assertFalse((REPO_ROOT / "test/kernel/du/gdr_visibility.cu").exists())

        self.assertIn('FLAGCX_CI_MPI_LABEL="rma GDR visibility ordinary"', rma_case)
        self.assertIn("for route in va dmabuf", rma_case)
        self.assertIn("FLAGCX_DMABUF_ENABLE", rma_case)
        self.assertIn(
            'FLAGCX_CI_MPI_LABEL="rma Hygon VMM route $route"',
            rma_case,
        )
        self.assertIn('NET_FILTER="RmaTest.GetSmall"', rma_case)
        self.assertIn(
            'if [[ "$platform_name" != "hygon" && "$platform_name" != "iluvatar" ]]; then',
            rma_case,
        )
        self.assertIn(
            '-x FLAGCX_CI_GDR_VISIBILITY_EXPECT_SETUP=success_or_unsupported',
            rma_case,
        )
        self.assertIn(
            "FLAGCX_CI_GDR_VISIBILITY_EXPECT_SETUP=\"$setup_expectation\"",
            rma_case,
        )
        self.assertIn("setup_expectation=unsupported", rma_case)
        self.assertIn("setup_expectation=success_or_unsupported", rma_case)
        self.assertIn(
            'FLAGCX_CI_MPI_LABEL="rma GDR visibility forced unsupported READ"',
            rma_case,
        )
        self.assertIn("FLAGCX_GDR_READ_REQUIRES_FLUSH=1", rma_case)
        self.assertIn(
            "FLAGCX_CI_GDR_VISIBILITY_EXPECT_READ=remote_error", rma_case
        )
        self.assertNotIn("FLAGCX_P2P_PERF_OP", rma_case)

        metax_env = METAX_ENV.read_text()
        self.assertIn('export RMA_PLATFORM_ENV="-x FLAGCX_USE_TUNER=1', metax_env)
        self.assertIn('"RMA_PLATFORM_ENV=$RMA_PLATFORM_ENV"', metax_env)

        maca_kernel_makefile = (
            REPO_ROOT / "test/kernel/maca/Makefile"
        ).read_text()
        self.assertIn("filter-out -fgpu-rdc", maca_kernel_makefile)

        flagcx_source = (REPO_ROOT / "flagcx/flagcx.cc").read_text()
        self.assertIn(
            "const int useGdr = ptrType != FLAGCX_PTR_HOST;", flagcx_source
        )

        ibrc_source = (
            REPO_ROOT / "flagcx/adaptor/net/ibrc_adaptor.cc"
        ).read_text()
        shca_caps_start = ibrc_source.index("#ifdef USE_SHCA")
        shca_caps = ibrc_source[
            shca_caps_start : ibrc_source.index("#else", shca_caps_start)
        ]
        self.assertIn(
            "flagcxIbVmmMrCaps = FLAGCX_VMM_MR_CAP_DMABUF", shca_caps
        )
        self.assertNotIn("FLAGCX_VMM_MR_CAP_VA", shca_caps)

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

    def test_symmem_runs_required_local_and_network_vmm_matrix(self):
        unit_runner = (
            REPO_ROOT / ".github/scripts/ci/run_unit_test.sh"
        ).read_text()
        symmem_runner = (
            REPO_ROOT / "test/script/symmem_test.sh"
        ).read_text()

        self.assertIn(
            'adaptor|p2p|rma|runner|symmem|device_api|device_api_unified_ir)',
            unit_runner,
        )
        self.assertIn('FLAGCX_CI_MPI_LABEL="symmem IPC local"', symmem_runner)
        self.assertIn('FLAGCX_CI_MPI_LABEL="symmem VMM local"', symmem_runner)
        self.assertNotIn('FLAGCX_CI_MPI_LABEL="symmem DU multi-segment"', symmem_runner)
        self.assertIn('nvidia|metax|ppu)', symmem_runner)
        self.assertIn('nvidia|du|metax|ppu)', symmem_runner)
        self.assertIn('FLAGCX_CI_REQUIRE_MULTI_SEGMENT=1', symmem_runner)
        self.assertIn('FLAGCX_CI_REQUIRE_PROVIDER_VMM=1', symmem_runner)
        self.assertIn('FLAGCX_VMM_MULTI_SEGMENT_ENABLE=0', symmem_runner)
        self.assertIn('"${vmm_provider_env[@]}"', symmem_runner)
        self.assertIn('DuMultiSegmentDisabledPreservesSingleVmm', (
            REPO_ROOT / 'test/unittest/symmem/coll_sym_register.cpp'
        ).read_text())
        self.assertIn('FLAGCX_VMM_ENABLE=1', symmem_runner)
        self.assertIn('FLAGCX_CI_REQUIRE_VMM=1', symmem_runner)
        self.assertIn('FLAGCX_CI_MPI_LABEL="symmem IPC + NET"', unit_runner)
        self.assertIn(
            'FLAGCX_CI_MPI_LABEL="symmem remote without NET"', unit_runner
        )
        self.assertIn(
            'FLAGCX_CI_MPI_LABEL="symmem local IPC + NET fallback"',
            unit_runner,
        )
        self.assertIn(
            'FLAGCX_CI_MPI_LABEL="symmem VMM + NET auto"', unit_runner
        )
        self.assertIn(
            'FLAGCX_CI_MPI_LABEL="symmem VMM + NET route union"',
            unit_runner,
        )
        self.assertIn('FLAGCX_DMABUF_ENABLE=1', unit_runner)
        self.assertIn('FLAGCX_CI_REQUIRE_VMM_ROUTE_UNION=1', unit_runner)
        self.assertIn(
            "SymMemTest.VmmNetRouteCapabilityUnion", unit_runner
        )
        self.assertIn("local symmem_run_vmm_net_data=1", unit_runner)
        self.assertIn(
            'if [[ "$platform_name" == "ppu" || '
            '"$platform_name" == "hygon" ]]',
            unit_runner,
        )
        self.assertIn(
            "if ((symmem_run_vmm_net_data != 0)); then", unit_runner
        )
        self.assertIn(
            "if ((symmem_run_vmm_net_data == 0)); then", unit_runner
        )
        self.assertIn(
            "Skipping $platform_name VMM + NET data tests", unit_runner
        )
        self.assertIn("FLAGCX_CI_ALLOW_VMM_NET_UNSUPPORTED=1", unit_runner)
        self.assertIn('FLAGCX_CI_REQUIRE_NET_MR=1', unit_runner)
        self.assertIn('FLAGCX_CI_REQUIRE_REMOTE_NO_NET=1', unit_runner)
        self.assertIn(
            'FLAGCX_CI_REQUIRE_LOCAL_IPC_NET_FALLBACK=1', unit_runner
        )
        self.assertIn('FLAGCX_CI_EXPECT_NET_ADAPTOR="$expected_adaptor"', unit_runner)
        self.assertIn(
            "SymMemTest.RemotePeersWithoutNetworkDoNotPublishWindow",
            unit_runner,
        )
        self.assertIn(
            "SymMemTest.RankLocalIpcFailureUsesNetworkMrFallback",
            unit_runner,
        )
        self.assertIn(
            "SymMemTest.DuplicateWindowsShareMrUntilLastDeregister",
            unit_runner,
        )
        self.assertIn(
            "SymMemTest.RankLocalStatusFailureConvergesDeterministically",
            unit_runner,
        )
        self.assertIn(
            "SymMemTest.AsymmetricDeregisterUsesCollectivePublishSlot",
            unit_runner,
        )
        self.assertIn(
            "SymMemTest.MrRollbackFailureRetainsWindowLeaseUntilRetry",
            unit_runner,
        )
        self.assertIn(
            "SymMemTest.FullMeshRoundFailureConvergesAndRetries",
            unit_runner,
        )
        self.assertIn(
            "SymMemTest.PublicRegistrationUsesVmmMrRouting",
            unit_runner,
        )
        self.assertIn(
            "SymMemTest.VmmLsaMapFailureDoesNotPublishNetworkOnlyWindow",
            unit_runner,
        )
        self.assertIn(
            "SymMemTest.DirectGdrVmmPreservesMrRoute",
            unit_runner,
        )
        self.assertIn(
            "SymMemTest.CrossNodeCleanupFailureConvergesBeforeRelease",
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
        self.assertIn(
            "SymMemTest.RankLocalSignalMrFailurePreservesErrorAndRetries",
            unit_runner,
        )
        self.assertNotIn(
            "SymMemTest.FlatMapRollbackFailureRetainsOwnershipForRetry",
            symmem_runner,
        )

    def test_iluvatar_keeps_ipc_and_vmm_capability_coverage(self):
        iluvatar_env = (
            REPO_ROOT / ".github/scripts/set_env/iluvatar.sh"
        ).read_text()
        unit_runner = (
            REPO_ROOT / ".github/scripts/ci/run_unit_test.sh"
        ).read_text()
        ipc_test = (
            REPO_ROOT / "test/unittest/adaptor/coll_ipc_mem_handle.cpp"
        ).read_text()

        self.assertNotIn("FLAGCX_CI_ADAPTOR_MPI_FILTER", iluvatar_env)
        self.assertNotIn("FLAGCX_CI_RMA_IPC_FILTER", iluvatar_env)
        self.assertIn("ASSERT_FLAGCX_SUCCESS(devHandle->setDevice(rank));", ipc_test)
        self.assertIn("CrossProcessLifecycle", ipc_test)
        rma_ipc_test = (
            REPO_ROOT / "test/unittest/rma/coll_rma_get_async.cpp"
        ).read_text()
        self.assertIn("op == flagcxNotSupported", rma_ipc_test)
        self.assertIn(
            "CoreX does not support asynchronous IPC Get", rma_ipc_test
        )
        self.assertIn("FLAGCX_CI_SYMMEM_VMM_SUPPORTED=0", iluvatar_env)
        route_union = unit_runner.index(
            'FLAGCX_CI_MPI_LABEL="symmem VMM + NET route union"'
        )
        vmm_skip_return = unit_runner.index(
            'echo "Skipping $platform_name VMM flat-map data tests',
            route_union,
        )
        self.assertLess(route_union, vmm_skip_return)
        self.assertIn("FLAGCX_CI_ALLOW_VMM_NET_UNSUPPORTED=1", unit_runner)
        self.assertIn("FLAGCX_CI_RUNNER_HETERO_SUPPORTED=0", iluvatar_env)

        for platform in ("cuda", "metax", "hygon", "ppu"):
            config = (
                REPO_ROOT / f".github/configs/{platform}.yml"
            ).read_text()
            source = (
                REPO_ROOT / f".github/scripts/set_env/{platform}.sh"
            ).read_text()
            if platform != "hygon":
                self.assertIn("  - symmem", config)
            self.assertIn("FLAGCX_CI_NODE1_MPI_ARGS=(", source)
            self.assertIn("FLAGCX_CI_NODE2_MPI_ARGS=(", source)

    def test_du_symmem_mpi_build_enables_segmented_tests(self):
        result = subprocess.run(
            [
                "make",
                "-B",
                "-n",
                "-C",
                str(REPO_ROOT / "test/unittest/symmem"),
                "USE_DU=1",
                "DEVICE_HOME=/opt/dtk/cuda/cuda-12",
                "CCL_HOME=/opt/dtk/cuda/cuda-12",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        compile_command = next(
            line
            for line in result.stdout.splitlines()
            if line.startswith("g++ coll_sym_register.cpp")
        )
        link_command = next(
            line
            for line in result.stdout.splitlines()
            if " -o " in line and "symmem_mpi_tests" in line
        )
        self.assertIn("-DUSE_DU_ADAPTOR", compile_command)
        self.assertIn("-I/opt/dtk/cuda/cuda-12/include", compile_command)
        self.assertIn("-lamdhip64", link_command)
        source = (
            REPO_ROOT / "test/unittest/symmem/coll_sym_register.cpp"
        ).read_text()
        self.assertIn("#elif defined(USE_DU_ADAPTOR)\n#include <hip/hip_runtime.h>", source)
        duhip = (
            REPO_ROOT / "flagcx/adaptor/device/duhip_adaptor.cc"
        ).read_text()
        self.assertIn("duhipAdaptorSymPhysDescribe,", duhip)
        self.assertIn("duhipAdaptorSymLsaMapBeginAt, NULL", duhip)

    def test_cuda_vmm_allocation_does_not_require_imex_fabric(self):
        source = (
            REPO_ROOT / "flagcx/adaptor/device/cuda_adaptor.cc"
        ).read_text()
        start = source.index("flagcxResult_t cudaAdaptorGdrMemAlloc")
        end = source.index("flagcxResult_t cudaAdaptorGdrMemFree", start)
        allocator = source[start:end]

        self.assertIn("CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR", allocator)
        self.assertNotIn("CU_MEM_HANDLE_TYPE_FABRIC", allocator)

    def test_symmem_status_convergence_is_allocation_free(self):
        source = (REPO_ROOT / "flagcx/core/sym_heap.cc").read_text()
        start = source.index(
            "static flagcxResult_t flagcxSymGlobalConvergeScalar"
        )
        end = source.index("} // namespace", start)
        convergence = source[start:end]

        self.assertNotIn("flagcxCalloc", convergence)
        self.assertNotIn("std::vector", convergence)
        self.assertNotIn("bootstrapCollAllGather", convergence)
        self.assertIn("bootstrapRecv", convergence)
        self.assertIn("bootstrapSend", convergence)

        retry_start = source.index(
            "flagcxResult_t flagcxSymRetryPendingCleanup"
        )
        retry_end = source.index(
            "flagcxResult_t flagcxSymRetainPendingCleanup", retry_start
        )
        retry = source[retry_start:retry_end]
        self.assertNotIn("std::vector", retry)
        self.assertNotIn("bootstrapCollAllGather", retry)
        self.assertIn("flagcxSymLogicalOr", retry)

        common = (REPO_ROOT / "flagcx/flagcx.cc").read_text()
        publish_start = common.index(
            "static flagcxResult_t flagcxDevCommStatePublishWindow"
        )
        publish_end = common.index(
            "static flagcxResult_t flagcxDevCommStateInit", publish_start
        )
        staged_publish = common[publish_start:publish_end]
        self.assertEqual(staged_publish.count("flagcxSymConvergeStatus"), 1)
        self.assertIn("flagcxSymWindowValidateDataRoutes", staged_publish)
        self.assertIn("flagcxSymWindowCommit", staged_publish)
        self.assertLess(
            staged_publish.index("flagcxSymConvergeStatus"),
            staged_publish.index("flagcxSymWindowCommit"),
        )

        # Keep the source check independent of clang-format's return-type
        # wrapping for this long function name.
        route_start = source.index(
            "flagcxSymWindowValidateDataRoutesForMode("
        )
        route_end = source.index(
            "static flagcxResult_t flagcxSymCleanupStepConverge", route_start
        )
        route_validation = source[route_start:route_end]
        self.assertIn("localPeerTransportEnabled", route_validation)
        self.assertIn("hasNetworkMr", route_validation)

        register_start = source.index(
            "flagcxResult_t flagcxSymWindowRegisterInternal"
        )
        register_end = source.index("\nfail:", register_start)
        registration = source[register_start:register_end]
        self.assertIn("kSymNetworkRouteGatherTag", registration)
        self.assertIn("flagcxParamP2pDisable()", registration)
        self.assertIn("flagcxParamDeviceOneSidedForceNet()", registration)
        self.assertIn("if (needsNetworkMr)", registration)

        vmm_start = source.index("const bool localVmmAvailable")
        vmm_end = source.index("flagcxResult_t commonVmmAvailability", vmm_start)
        vmm_callbacks = source[vmm_start:vmm_end]
        for callback in (
            "symPhysAlloc",
            "symPhysFree",
            "symFlatMap",
            "symFlatMappingUnmap",
            "symFlatVaFree",
        ):
            self.assertIn(callback, vmm_callbacks)

    def test_all_device_adaptors_probe_dmabuf_then_va_for_vmm_mr(self):
        adaptor_sources = (
            "cuda_adaptor.cc",
            "maca_adaptor.cc",
            "duhip_adaptor.cc",
            "ppu_cuda_adaptor.cc",
        )
        for filename in adaptor_sources:
            source = (
                REPO_ROOT / "flagcx/adaptor/device" / filename
            ).read_text()
            self.assertIn(
                "FLAGCX_VMM_MR_CAP_DMABUF | FLAGCX_VMM_MR_CAP_VA",
                source,
                filename,
            )
            self.assertIn("SymPhysAlloc", source, filename)
            self.assertIn("MemGetHandleForAddressRange", source, filename)
            self.assertNotIn("GetAllocationVmmMrCaps", source, filename)

        # The DCU HIP runtime aborts instead of returning an
        # unsupported status for NVIDIA-only attributes 110 and 124.  Its
        # DMA-BUF on/off CI invocations probe the real operations instead.
        duhip = (
            REPO_ROOT / "flagcx/adaptor/device/duhip_adaptor.cc"
        ).read_text()
        self.assertNotIn(
            "CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED",
            duhip,
        )
        self.assertNotIn("CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED", duhip)
        gdr_alloc = duhip[
            duhip.index("flagcxResult_t duhipAdaptorGdrMemAlloc") :
            duhip.index("flagcxResult_t duhipAdaptorGdrMemFree")
        ]
        self.assertNotIn("mrCaps", gdr_alloc)
        dma_support = duhip[
            duhip.index("flagcxResult_t duhipAdaptorDmaSupport") :
            duhip.index("duhipAdaptorMemGetHandleForAddressRange")
        ]
        self.assertIn("*dmaBufferSupport = true", dma_support)

        # MetaX returns InvalidDevicePointer for valid VMM allocations that
        # cannot be exported as DMA-BUF.  The latest adaptor must normalize
        # that allocation-specific capability miss so auto mode can try VA.
        maca = (
            REPO_ROOT / "flagcx/adaptor/device/maca_adaptor.cc"
        ).read_text()
        maca_export = maca[
            maca.index("macaAdaptorMemGetHandleForAddressRange") :
            maca.index("flagcxResult_t macaAdaptorHostRegister")
        ]
        self.assertIn("mcErrorInvalidDevicePointer", maca_export)
        self.assertIn("return flagcxNotSupported", maca_export)

        common = (REPO_ROOT / "flagcx/flagcx.cc").read_text()
        dma_route = common.index("return FLAGCX_VMM_MR_ROUTE_DMABUF")
        va_route = common.index("return FLAGCX_VMM_MR_ROUTE_VA")
        unsupported_route = common.index("return FLAGCX_VMM_MR_ROUTE_NONE")
        self.assertLess(dma_route, va_route)
        self.assertLess(va_route, unsupported_route)
        self.assertIn('flagcxGetEnv("FLAGCX_DMABUF_ENABLE")', common)
        self.assertIn("dmaBufEnabled", common)
        self.assertNotIn("getAllocationVmmMrCaps", common)

        route_tests = (
            REPO_ROOT / "test/unittest/symmem/test_sym_window_struct.cpp"
        ).read_text()
        for test_name in (
            "DisabledDmaBufSkipsProbeAndRegistration",
            "DmaBufUnsupportedWithoutVaDoesNotFallback",
            "EnabledDmaBufUsesDmaBufRoute",
            "DmaBufSubrangeExportsAllocationAndUsesPageOffset",
            "DeviceCapabilitiesCanRejectUnsafeVaRoute",
            "ProviderCapabilitiesSuppressUnvalidatedVmmRoutes",
            "PartialDmaBufMrIsReleasedBeforeVaFallback",
            "PartialDmaBufMrCleanupFailureSuppressesVaFallback",
        ):
            self.assertIn(test_name, route_tests)

        mpi_route_tests = (
            REPO_ROOT / "test/unittest/symmem/coll_sym_register.cpp"
        ).read_text()
        self.assertIn("VmmNetRouteCapabilityUnion", mpi_route_tests)
        self.assertIn("routeSucceeded[0]", mpi_route_tests)
        self.assertIn("routeSucceeded[1]", mpi_route_tests)

    def test_hygon_flat_vmm_cleanup_tracks_and_unmaps_individual_slots(self):
        duhip = (
            REPO_ROOT / "flagcx/adaptor/device/duhip_adaptor.cc"
        ).read_text()
        self.assertIn(
            "hipMemGenericAllocationHandle_t handle;",
            duhip[duhip.index("struct DuhipVmmAllocation") :],
        )
        self.assertIn(
            "bool handleOwned;",
            duhip[duhip.index("struct DuhipVmmAllocation") :],
        )
        self.assertIn("struct DuhipSymPhysHandle", duhip)
        self.assertIn("bool releaseOwned;", duhip)

        gdr_alloc = duhip[
            duhip.index("flagcxResult_t duhipAdaptorGdrMemAlloc") :
            duhip.index("flagcxResult_t duhipAdaptorGdrMemFree")
        ]
        self.assertNotIn(
            "if (hipMemRelease(handle) != hipSuccess)", gdr_alloc
        )
        self.assertIn(
            "DuhipVmmAllocation{handle,allocSize,true,true,true}",
            "".join(gdr_alloc.split()),
        )

        gdr_free = duhip[
            duhip.index("flagcxResult_t duhipAdaptorGdrMemFree") :
            duhip.index("flagcxResult_t duhipAdaptorStreamCreate")
        ]
        self.assertLess(
            gdr_free.index("hipMemUnmap"),
            gdr_free.index("hipMemRelease(allocation.handle)"),
        )
        self.assertLess(
            gdr_free.index("hipMemAddressFree"),
            gdr_free.index("hipMemRelease(allocation.handle)"),
        )

        phys_alloc = duhip[
            duhip.index("flagcxResult_t duhipAdaptorSymPhysAlloc") :
            duhip.index("flagcxResult_t duhipAdaptorSymPhysFree")
        ]
        self.assertLess(
            phys_alloc.index("hipMemGetAddressRange"),
            phys_alloc.index("hipMemRetainAllocationHandle"),
        )
        self.assertIn("gDuhipVmmAllocations.find", phys_alloc)
        self.assertIn("handle->handle = allocation.handle", phys_alloc)
        self.assertIn("handle->releaseOwned = false", phys_alloc)
        self.assertIn("handle->releaseOwned = true", phys_alloc)

        phys_free = duhip[
            duhip.index("flagcxResult_t duhipAdaptorSymPhysFree") :
            duhip.index("flagcxResult_t duhipAdaptorSymFlatMappingUnmap")
        ]
        self.assertIn("duhipSymPhysHandleDestroy", phys_free)
        destroy = duhip[
            duhip.index("duhipSymPhysHandleDestroy") :
            duhip.index("flagcxResult_t duhipAdaptorSymPhysAlloc")
        ]
        self.assertLess(
            destroy.index("if (physHandle->releaseOwned)"),
            destroy.index("hipMemRelease(physHandle->handle)"),
        )

        self.assertIn("struct DuhipFlatMapping", duhip)
        self.assertIn("gDuhipFlatMappings", duhip)
        self.assertIn("mappedSlots", duhip)
        self.assertIn("importedHandleOwned", duhip)

        flat_map = duhip[
            duhip.index("flagcxResult_t duhipAdaptorSymFlatMap(") :
            duhip.index("flagcxResult_t duhipAdaptorSymFlatUnmap(")
        ]
        self.assertLess(
            flat_map.index("gDuhipFlatMappings.emplace"),
            flat_map.index("hipMemImportFromShareableHandle"),
        )
        self.assertLess(
            flat_map.index("*flatBase = (void *)base"),
            flat_map.index("hipMemImportFromShareableHandle"),
        )
        for stage in (
            "stage=address-reserve",
            "stage=import",
            "stage=map",
            "stage=set-access",
        ):
            self.assertIn(stage, flat_map)
        self.assertIn("mapping.importedHandleOwned[i] = 1", flat_map)
        self.assertNotIn("hipMemRelease(peerHandle)", flat_map)

        legacy_unmap = duhip.index(
            "flagcxResult_t duhipAdaptorSymFlatUnmap("
        )
        mapping_unmap_start = duhip.index(
            "flagcxResult_t duhipAdaptorSymFlatMappingUnmap", legacy_unmap
        )
        va_free_start = duhip.index(
            "flagcxResult_t duhipAdaptorSymFlatVaFree", mapping_unmap_start
        )
        mapping_unmap = duhip[mapping_unmap_start:va_free_start]
        self.assertIn("for (int i = 0; i < nPeers; i++)", mapping_unmap)
        self.assertIn("hipMemUnmap(slot, allocSize)", mapping_unmap)
        self.assertIn("mapping.mappedSlots[i] = 0", mapping_unmap)
        self.assertNotIn("hipMemUnmap((hipDeviceptr_t)flatBase,", mapping_unmap)
        self.assertIn("stage=release-import", mapping_unmap)
        self.assertLess(
            mapping_unmap.index("hipMemUnmap(slot, allocSize)"),
            mapping_unmap.index("hipMemRelease(mapping.importedHandles[i])"),
        )

        va_free = duhip[
            va_free_start : duhip.index(
                "flagcxResult_t duhipAdaptorSymMulticastSupported"
            )
        ]
        self.assertIn("mapping.mappedSlots[i]", va_free)
        self.assertIn("mapping.importedHandleOwned[i]", va_free)
        self.assertIn("hipMemAddressFree", va_free)
        self.assertIn("gDuhipFlatMappings.erase(it)", va_free)

    def test_multicast_retained_handle_extension_is_latest_only(self):
        header = (
            REPO_ROOT / "flagcx/adaptor/include/flagcx_device_adaptor.h"
        ).read_text()
        v1 = header[
            header.index("struct flagcxDeviceAdaptor_v1") :
            header.index("struct flagcxDeviceAdaptor_latest")
        ]
        self.assertNotIn("symMulticastImport", v1)
        self.assertIn("symMulticastImport", header)

        for filename, symbol in (
            ("cuda_adaptor.cc", "cudaAdaptorSymMulticastImport"),
            ("maca_adaptor.cc", "macaAdaptorSymMulticastImport"),
            ("duhip_adaptor.cc", "duhipAdaptorSymMulticastImport"),
            ("ppu_cuda_adaptor.cc", "ppucudaAdaptorSymMulticastImport"),
        ):
            source = (
                REPO_ROOT / "flagcx/adaptor/device" / filename
            ).read_text()
            self.assertIn(symbol, source)

        sym_heap = (REPO_ROOT / "flagcx/core/sym_heap.cc").read_text()
        self.assertIn("deviceAdaptor->symMulticastImport", sym_heap)
        self.assertIn("d->mcHandle, /*importFd=*/-1", sym_heap)

    def test_vmm_mr_provider_capability_is_latest_only(self):
        header = (
            REPO_ROOT / "flagcx/adaptor/include/flagcx_net_adaptor.h"
        ).read_text()
        v1 = header[
            header.index("struct flagcxNetAdaptor_v1") :
            header.index("struct flagcxNetAdaptor_latest")
        ]
        latest = header[header.index("struct flagcxNetAdaptor_latest") :]
        self.assertNotIn("vmmMrCaps", v1)
        self.assertNotIn("internalFlags", v1)
        self.assertIn("uint32_t vmmMrCaps", latest)
        self.assertIn("uint32_t internalFlags", latest)
        self.assertIn(
            "FLAGCX_NET_ADAPTOR_INTERNAL_LEGACY_V1", latest
        )

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
        self.assertIn("FLAGCX_CI_EXPECT_NET_ADAPTOR", runner_test)
        forced_net = unit_runner[
            unit_runner.index('FLAGCX_CI_MPI_LABEL="runner forced NET"') :
        ]
        forced_net = forced_net[:forced_net.index(";;")]
        self.assertIn("FLAGCX_CI_EXPECT_NET_ADAPTOR=IB", forced_net)
        self.assertIn("FLAGCX_IB_QPS_PER_CONNECTION=2", forced_net)
        self.assertIn("FLAGCX_IB_SPLIT_DATA_ON_QPS=0", forced_net)
        self.assertIn("FLAGCX_CI_EXPECT_COLL_MULTICHANNEL=1", forced_net)

        striping = unit_runner[
            unit_runner.index(
                'FLAGCX_CI_MPI_LABEL="runner forced NET multi-QP striping"'
            ) :
        ]
        striping = striping[:striping.index(";;")]
        self.assertIn("FLAGCX_IB_QPS_PER_CONNECTION=2", striping)
        self.assertIn("FLAGCX_IB_SPLIT_DATA_ON_QPS=1", striping)
        self.assertIn("FLAGCX_IBUC_SPLIT_DATA_ON_QPS=1", striping)
        self.assertIn("FLAGCX_CI_EXPECT_COLL_QP_STRIPING=1", striping)
        self.assertIn("FLAGCX_CI_RUNNER_BYTES=67108864", striping)
        self.assertIn("FlagCXCollTest.AlltoAll", striping)

        adaptor_case = unit_runner[unit_runner.index("    adaptor)") :]
        adaptor_case = adaptor_case[: adaptor_case.index("    core|service)")]
        self.assertIn('basename "$SET_ENV_SCRIPT" .sh', adaptor_case)
        self.assertIn('== "cuda"', adaptor_case)
        self.assertIn('FLAGCX_CI_MPI_LABEL="IBRC QP-count mismatch"', adaptor_case)
        self.assertIn('FLAGCX_CI_MPI_LABEL="IBRC split-data mismatch"', adaptor_case)
        self.assertEqual(
            adaptor_case.count("FLAGCX_CI_EXPECT_IB_GEOMETRY_MISMATCH=1"),
            1,
        )
        self.assertIn("FLAGCX_IB_QPS_PER_CONNECTION=1", adaptor_case)
        self.assertIn("FLAGCX_IB_QPS_PER_CONNECTION=2", adaptor_case)
        self.assertIn("FLAGCX_IB_SPLIT_DATA_ON_QPS=0", adaptor_case)
        self.assertIn("FLAGCX_IB_SPLIT_DATA_ON_QPS=1", adaptor_case)
        self.assertIn("IbConnectionGeometryMpiTest", adaptor_case)

        ppu_forced_net = ppu_env[
            ppu_env.index('FLAGCX_CI_MPI_LABEL="runner BAREX forced NET"') :
        ]
        ppu_forced_net = ppu_forced_net[:ppu_forced_net.index("return")]
        self.assertIn("FLAGCX_CI_EXPECT_NET_ADAPTOR=BAREX", ppu_forced_net)
        self.assertIn("FLAGCX_CI_EXPECT_COLL_MULTICHANNEL=1", ppu_forced_net)
        self.assertIn('FLAGCX_CI_MPI_LABEL="runner PPU PXN four-rank"', ppu_forced_net)
        self.assertIn("FLAGCX_PXN_DISABLE=0", ppu_forced_net)
        self.assertIn('FLAGCX_CI_PXN_RANKS_PER_NODE="$FLAGCX_CI_NODE_NP"', ppu_forced_net)
        self.assertIn("FLAGCX_CI_EXPECT_PXN=1", ppu_forced_net)

    def test_p2p_ci_runs_write_only_engine_coverage(self):
        unit_runner = (
            REPO_ROOT / ".github/scripts/ci/run_unit_test.sh"
        ).read_text()
        p2p_makefile = (
            REPO_ROOT / "test/unittest/p2p/Makefile"
        ).read_text()
        platform_envs = {
            platform: (
                REPO_ROOT / f".github/scripts/set_env/{platform}.sh"
            ).read_text()
            for platform in ("cuda", "metax", "hygon", "ppu")
        }
        p2p_runner = unit_runner[unit_runner.index("    p2p)") :]
        p2p_runner = p2p_runner[: p2p_runner.index("    rma)")]

        self.assertNotIn("read_diagnostic_filter", p2p_runner)
        self.assertIn('FLAGCX_CI_MPI_LABEL="p2p Engine WRITE MPI tests"', p2p_runner)
        self.assertIn('make -C "$suite_dir" run-mpi', p2p_runner)
        self.assertIn("test_p2p_engine_read.cpp", p2p_makefile)
        self.assertIn("test_p2p_gpu_read.cpp", p2p_makefile)
        self.assertIn("test_p2p_visibility_policy.cpp", p2p_makefile)
        self.assertIn("coll_p2p_engine_write.cpp", p2p_makefile)
        self.assertIn("ifeq ($(USE_SHARED_P2P_ENGINE),1)", p2p_makefile)
        self.assertIn("filter-out test_p2p_adaptor.cpp", p2p_makefile)
        for name in ("test_p2p_adaptor.cpp", "test_p2p_batch.cpp",
                     "test_p2p_gpu_read.cpp"):
            with self.subTest(legacy_adaptor_source=name):
                source = REPO_ROOT / "test/unittest/p2p" / name
                self.assertTrue(source.is_file())
                self.assertIn("flagcxNetIbP2p", source.read_text())
        self.assertIn("FLAGCX_IB_QPS_PER_CONNECTION=2", p2p_runner)
        self.assertIn("FLAGCX_P2P_QPS_PER_CONN=2", p2p_runner)
        for platform, platform_env in platform_envs.items():
            with self.subTest(platform=platform):
                self.assertIn(
                    "FLAGCX_CI_ENABLE_SHARED_P2P_ENGINE=1", platform_env
                )
        self.assertIn("USE_SHARED_P2P_ENGINE=1", p2p_runner)
        self.assertIn('build-p2p-shared', p2p_runner)
        self.assertIn('FLAGCX_LIB="$shared_project_build/lib"', p2p_runner)
        self.assertIn(
            'LD_LIBRARY_PATH="$shared_project_build/lib:$LD_LIBRARY_PATH"',
            p2p_runner,
        )
        self.assertIn('shared_status != 0', p2p_runner)
        self.assertIn('shared_mpi_status != 0', p2p_runner)

        test_make = (REPO_ROOT / "test/make.inc").read_text()
        read_tests = (
            REPO_ROOT / "test/unittest/p2p/test_p2p_engine_read.cpp"
        ).read_text()
        rpc_tests = (
            REPO_ROOT / "test/unittest/p2p/test_p2p_engine_rpc.cpp"
        ).read_text()
        self.assertIn("CXXFLAGS += -DUSE_SHARED_P2P_ENGINE", test_make)
        self.assertIn("#ifdef USE_SHARED_P2P_ENGINE", read_tests)
        self.assertIn("RejectsOtherEnginePrefaceAndAcceptsNextPeer",
                      rpc_tests)

        legacy_engine = (REPO_ROOT / "flagcx/core/flagcx_p2p.cc").read_text()
        self.assertIn('"P2P_QPS_PER_CONN"', legacy_engine)

        engine = (REPO_ROOT / "flagcx/core/flagcx_p2p_shared.cc").read_text()
        ibrc = (REPO_ROOT / "flagcx/adaptor/net/ibrc_adaptor.cc").read_text()
        self.assertIn("flagcxIbEngineSetConnectionConfig", engine)
        self.assertIn("flagcxIbEngineClearConnectionConfig", engine)
        self.assertIn('getenv("FLAGCX_IB_QPS_PER_CONNECTION")', engine)
        self.assertIn("flagcxIbConnectionQpsPerConn", ibrc)
        self.assertIn("flagcxIbConnectionMtuCap", ibrc)
        self.assertIn("drainAndCleanupIpcXfer", engine)
        self.assertIn("deviceAdaptor->streamSynchronize(xfer->stream)", engine)
        self.assertIn('"P2P_CONNECT_TIMEOUT"', engine)
        self.assertIn("std::chrono::steady_clock::now() >= deadline", engine)
        self.assertIn("stopWithAccept &&", engine)
        self.assertEqual(engine.count("const flagcxResult_t setDeviceResult ="), 2)
        self.assertIn("if (setDeviceResult != flagcxSuccess)", engine)
        self.assertIn("connect notification channel failed", engine)
        self.assertIn("accept notification channel failed", engine)
        self.assertIn("notification listener initialization failed", engine)
        self.assertIn("bootstrap listener initialization failed", engine)
        self.assertGreaterEqual(
            engine.count("connectNotifSocket(conn,"), 2
        )
        self.assertLess(
            engine.index("Acquire every mapping before queueing"),
            engine.index("bool usedAsync = false"),
        )

    def test_runner_converges_async_errors_and_stops_new_communicators(self):
        fixture = (
            REPO_ROOT / "test/unittest/runner/include/runner_fixtures.hpp"
        ).read_text()
        runner = (
            REPO_ROOT / "test/unittest/runner/main_mpi.cpp"
        ).read_text()
        runner_dir = REPO_ROOT / "test/unittest/runner"

        self.assertIn("synchronizeAndCheckAsyncError", fixture)
        self.assertIn("flagcxCommGetAsyncError", runner)
        self.assertIn("MPI_Allreduce", runner)
        self.assertIn("runnerTransportFailureObserved = true", runner)
        self.assertIn("if (runnerTransportFailureObserved)", runner)

        unit_runner = (
            REPO_ROOT / ".github/scripts/ci/run_unit_test.sh"
        ).read_text()
        runner_case = unit_runner[unit_runner.index("    runner)") :]
        runner_case = runner_case[: runner_case.index("    symmem)")]
        self.assertIn('platform_name" == "hygon"', runner_case)
        self.assertIn("FLAGCX_IB_TIMEOUT=14", runner_case)
        self.assertIn("FLAGCX_IB_RETRY_CNT=1", runner_case)
        self.assertIn('"${runner_net_platform_env[@]}"', runner_case)
        self.assertEqual(
            runner_case.count("./build/bin/runner_mpi_tests"), 5
        )
        self.assertIn("FLAGCX_CI_EXPECT_RUNNER_MODE=HOMO", runner_case)
        self.assertEqual(
            runner_case.count("FLAGCX_CI_EXPECT_RUNNER_MODE=HYBRID"), 4
        )
        self.assertIn('"${FLAGCX_CI_ENABLE_PXN:-1}" == "1"', runner_case)
        self.assertIn('FLAGCX_CI_EXPECT_PXN=1', runner_case)
        self.assertIn('FLAGCX_PXN_DISABLE=1', runner_case)
        self.assertIn('FLAGCX_PXN_DISABLE=0', runner_case)
        self.assertIn('FLAGCX_CI_PXN_RANKS_PER_NODE="$FLAGCX_CI_NODE_NP"', runner_case)
        pxn_case = runner_case[
            runner_case.index('if [[ "${FLAGCX_CI_ENABLE_PXN:-1}" == "1" ]]') :
        ]
        self.assertIn('pxn_launcher+=(FLAGCX_IB_HCA="$FLAGCX_CI_PXN_HCAS")', pxn_case)
        self.assertIn('-x FLAGCX_IB_HCA="$FLAGCX_CI_PXN_HCAS"', pxn_case)
        self.assertIn('local pxn_visible_devices_var=CUDA_VISIBLE_DEVICES', pxn_case)
        self.assertIn('if [[ "$platform_name" == "metax" ]]; then', pxn_case)
        self.assertIn('pxn_visible_devices_var=MACA_VISIBLE_DEVICES', pxn_case)
        self.assertIn('elif [[ "$platform_name" == "hygon" ]]; then', pxn_case)
        self.assertIn('pxn_visible_devices_var=HIP_VISIBLE_DEVICES', pxn_case)
        self.assertIn('pxn_launcher+=("$pxn_visible_devices_var=$FLAGCX_CI_PXN_VISIBLE_DEVICES")', pxn_case)
        self.assertIn('-x "$pxn_visible_devices_var=$FLAGCX_CI_PXN_VISIBLE_DEVICES"', pxn_case)
        self.assertIn('"$MPI_RUNNER" -np "$FLAGCX_CI_RUNNER_NP"', pxn_case)
        self.assertIn('FLAGCX_CI_MPI_LABEL="runner $platform_name PXN four-rank"', pxn_case)
        self.assertNotIn('FLAGCX_CI_ENABLE_PXN_INTERSERVER', runner_case)
        self.assertIn('FLAGCX_CI_EXPECT_PEER_TRANSPORT=NET', runner_case)
        self.assertIn("FLAGCX_CI_EXPECT_PEER_TRANSPORT=P2P", runner_case)
        self.assertIn("FLAGCX_CI_EXPECT_PEER_TRANSPORT=NET", runner_case)
        self.assertNotIn("--gtest_filter=FlagCXCollTest.Scatter", runner_case)
        self.assertNotIn("--gtest_filter=FlagCXCollTest.AllToAllV", runner_case)

        sendrecv = (runner_dir / "coll_sendrecv.cpp").read_text()
        self.assertIn("FLAGCX_CI_EXPECT_PEER_TRANSPORT", sendrecv)
        self.assertIn("connector->proxyConn.transport", sendrecv)
        self.assertIn("TRANSPORT_P2P", sendrecv)
        self.assertIn("TRANSPORT_NET", sendrecv)
        self.assertNotIn("connector->transportComm", sendrecv)

        proxy = (REPO_ROOT / "flagcx/core/proxy.cc").read_text()
        self.assertIn("proxyConn->transport = -1", proxy)
        self.assertIn("proxyConn->transport = transport", proxy)
        self.assertTrue((runner_dir / "coll_alltoallv.cpp").is_file())

        for path in runner_dir.glob("coll_*.cpp"):
            source = path.read_text()
            if "streamSynchronize(stream)" in source:
                self.fail(
                    f"{path.name} bypasses collective async-error convergence"
                )
            if "TEST_F(FlagCXCollTest" in source:
                self.assertIn(
                    "synchronizeAndCheckAsyncError()", source, path.name
                )

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
        self.assertIn("RmaTest.GetAsyncOffsets", rma_makefile)
        self.assertIn("RmaTest.IpcGetCounterFollowsCopy", rma_makefile)
        self.assertIn("run-mpi-ipc", rma_makefile)
        self.assertIn("run-mpi-net", rma_makefile)

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

    def test_rma_registration_failure_runs_in_two_rank_network_ci(self):
        rma_makefile = (
            REPO_ROOT / "test/unittest/rma/Makefile"
        ).read_text()
        registration_test = (
            REPO_ROOT
            / "test/unittest/rma/coll_rma_registration_failure.cpp"
        ).read_text()

        self.assertRegex(
            rma_makefile, r"MPI_SRCS\s*:=\s*\$\(wildcard coll_\*\.cpp\)"
        )
        network_target = rma_makefile[
            rma_makefile.index("run-mpi-net:") :
        ]
        network_target = network_target[: network_target.index("\n\n")]
        self.assertIn("-np 2", network_target)
        self.assertNotIn("--gtest_filter", network_target)
        self.assertIn("failRegMr", registration_test)
        self.assertIn(
            "oneSideDataMetadataExchangeCount", registration_test
        )
        self.assertIn("flagcxOneSideRegister", registration_test)

    def test_kernel_proxy_transport_regressions_are_in_ci(self):
        core_makefile = (
            REPO_ROOT / "test/unittest/core/Makefile"
        ).read_text()
        unit_runner = (
            REPO_ROOT / ".github/scripts/ci/run_unit_test.sh"
        ).read_text()
        transport_test = (
            REPO_ROOT
            / "test/unittest/core/test_kernel_proxy_transport.cpp"
        ).read_text()
        coll_proxy_test = (
            REPO_ROOT
            / "test/unittest/core/test_coll_proxy_progress.cpp"
        ).read_text()

        self.assertIn("UNIT_SRCS   := $(wildcard test_*.cpp)", core_makefile)
        core_case = unit_runner[unit_runner.index("    core|service)") :]
        core_case = core_case[: core_case.index("    rma)")]
        self.assertIn('make -C "$suite_dir" run-unit', core_case)
        for platform in ("cuda", "metax", "ppu"):
            config = (
                REPO_ROOT / f".github/configs/{platform}.yml"
            ).read_text()
            self.assertIn("  - core", config, platform)

        self.assertIn(
            "OutOfOrderRequestsAdvanceOnlyContiguousPrefix", transport_test
        )
        for test_name in (
            "GetDataCompletionWaitsForFlushAndRetriesBackpressure",
            "ImmediateGetStillWaitsForSynchronousFlush",
            "GetFlushFailureBecomesScoreboardError",
            "DataFailureSkipsRequiredGetFlush",
            "AsyncGetFlushFailureBecomesCompletionError",
            "AbortReleasesMalformedInflightRequest",
            "OutOfOrderGetFlushesAdvanceOnlyContiguousPrefix",
        ):
            self.assertIn(test_name, transport_test)
        self.assertIn(
            "StagingSlotsRemainOwnedUntilRequestCompletion", transport_test
        )
        self.assertIn("FailedDataSuppressesReleaseSubmission", transport_test)
        self.assertIn(
            "TerminalFinalizeWaitsForLateProducerReservation", transport_test
        )
        self.assertIn(
            "ClosedProducerGateRejectsNewReservations", transport_test
        )
        self.assertIn(
            "TerminalFinalizeWaitsForBackpressuredReservation",
            transport_test,
        )
        self.assertIn("ProducerGateSerializesEntryWithClose", transport_test)
        self.assertIn(
            "DequeueReturnsInProgressForUnpublishedReservation",
            transport_test,
        )
        self.assertIn("DequeueConsumesPublishedReservation", transport_test)
        self.assertIn(
            "DifferentOrderingDomainsCanRemainInflightAndRetireIndependently",
            coll_proxy_test,
        )
        self.assertIn("maxConcurrentDomains", coll_proxy_test)

        cuda_config = (
            REPO_ROOT / ".github/configs/cuda.yml"
        ).read_text()
        self.assertIn("  - device_api_host", cuda_config)
        cleanup_test = (
            REPO_ROOT
            / "test/unittest/device_api/test_dev_comm_cleanup.cpp"
        ).read_text()
        self.assertIn(
            "QuiesceReturnsKernelProxyTerminalStatus", cleanup_test
        )
        self.assertIn("PutValueStagingLayoutTest", cleanup_test)

        proxy_source = (REPO_ROOT / "flagcx/core/proxy.cc").read_text()
        terminal_publish = proxy_source[
            proxy_source.index("flagcxKernelProxyPublishTerminal(") :
        ]
        terminal_publish = terminal_publish[
            : terminal_publish.index("flagcxKernelProxyAdvanceCompleted(")
        ]
        self.assertNotIn("kernelState.fifos[", terminal_publish)
        immediate_completion = proxy_source[
            proxy_source.index("if (!postedIB) {") :
        ]
        immediate_completion = immediate_completion[
            : immediate_completion.index("hasPending = false;")
        ]
        self.assertLess(
            immediate_completion.index("flagcxKernelProxyPublishTerminal("),
            immediate_completion.index("flagcxKernelProxyAdvanceCompleted("),
        )
        kernel_retire = proxy_source.index(
            "// Retire all kernel workers before destroying any GPU-visible FIFO."
        )
        deferred_fifo_destroy = proxy_source.index(
            "fifo->flagcxFifoDestroy()", kernel_retire
        )
        self.assertLess(
            proxy_source.index("comm->workerPool->retire(", kernel_retire),
            deferred_fifo_destroy,
        )

        device_api_case = unit_runner[
            unit_runner.index("run_device_api() {") :
        ]
        device_api_case = device_api_case[
            : device_api_case.index("run_device_api_unified_ir() {")
        ]
        self.assertIn("FLAGCX_KERNEL_PROXY_PARALLELISM=4", device_api_case)
        self.assertIn("FLAGCX_IB_QPS_PER_CONNECTION=2", device_api_case)
        cuda_kernel = (
            REPO_ROOT / "test/kernel/nvidia/device_api.cu"
        ).read_text()
        self.assertIn("MultiContextPutRelease", cuda_kernel)
        cuda_ir_kernel = (
            REPO_ROOT / "test/kernel/nvidia/device_ir.cu"
        ).read_text()
        cuda_ir_test = (
            REPO_ROOT
            / "test/unittest/device_api/test_device_ir_inter.cpp"
        ).read_text()
        self.assertIn("kernelNetMultiContextPutSignalIncS", cuda_ir_kernel)
        self.assertIn("S3b MultiContextPutSignalIncS", cuda_ir_test)
        self.assertIn("flagcxDevNet net(devComm, contextId)", cuda_kernel)
        self.assertIn("flagcxTeamTagWorld{}, net, contextId", cuda_kernel)
        device_api_test = (
            REPO_ROOT
            / "test/unittest/device_api/test_device_api_inter.cpp"
        ).read_text()
        self.assertIn("usleep(100000)", device_api_test)


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
