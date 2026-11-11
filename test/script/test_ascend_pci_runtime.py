"""Exercise the public AscendCL/DCMI calls used by the CI PCI preflight."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / ".github/scripts/ci/check_ascend_pci_runtime.py"
)
SPEC = importlib.util.spec_from_file_location("check_ascend_pci_runtime", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def make_acl(forward_result=0, reverse_result=0, include_forward=True,
             include_reverse=True, duplicate=False, reverse_device=None):
    def acl_init(_):
        return 0

    def acl_finalize():
        return 0

    def device_count(out):
        out._obj.value = 2
        return 0

    def user_to_logic(device, out):
        out._obj.value = device
        return 0

    acl = SimpleNamespace(
        aclInit=acl_init,
        aclFinalize=acl_finalize,
        aclrtGetDeviceCount=device_count,
        aclrtGetLogicDevIdByUserDevId=user_to_logic,
    )
    if include_forward:
        def forward(device, out, _):
            if forward_result == 0:
                pci_device = 0 if duplicate else device
                out.value = f"0000:3d:{pci_device:02x}.0".encode()
            return forward_result

        acl.aclrtDeviceGetPCIBusId = forward
    if include_reverse:
        def reverse(bus_id, out):
            if reverse_result == 0:
                out._obj.value = (
                    reverse_device
                    if reverse_device is not None
                    else int(bus_id.decode().split(":")[2][:2], 16)
                )
            return reverse_result

        acl.aclrtDeviceGetByPCIBusId = reverse
    return acl


def make_dcmi(domain=0x12345678, init_result=0, duplicate=False):
    def dcmi_init():
        return init_result

    def logic_to_card(card, chip, logic):
        card._obj.value = 0
        chip._obj.value = logic
        return 0

    def pcie_info(_card, chip, out):
        out._obj.domain = domain
        out._obj.bus = 0x3D
        out._obj.device = 0 if duplicate else chip
        out._obj.function = 0
        return 0

    return SimpleNamespace(
        dcmi_init=dcmi_init,
        dcmi_get_card_id_device_id_from_logicid=logic_to_card,
        dcmi_get_device_pcie_info_v2=pcie_info,
    )


class AscendPciRuntimeTest(unittest.TestCase):
    def check_with(self, acl, dcmi=None):
        loaded = []

        def load(name):
            loaded.append(name)
            if name == "libascendcl.so":
                return acl
            if name == "libdcmi.so" and dcmi is not None:
                return dcmi
            raise OSError("not found")

        return MODULE.check_runtime(load), loaded

    def test_ascendcl_round_trip_does_not_load_dcmi(self):
        provider, loaded = self.check_with(make_acl())
        self.assertIn("AscendCL", provider)
        self.assertEqual(loaded, ["libascendcl.so"])

    def test_unsupported_ascendcl_uses_dcmi(self):
        provider, loaded = self.check_with(
            make_acl(forward_result=207000), make_dcmi()
        )
        self.assertIn("DCMI", provider)
        self.assertIn("libdcmi.so", loaded)

    def test_unsupported_reverse_lookup_uses_unique_forward_mapping(self):
        provider, loaded = self.check_with(make_acl(reverse_result=207000))
        self.assertIn("AscendCL", provider)
        self.assertEqual(loaded, ["libascendcl.so"])

    def test_duplicate_forward_bdf_rejected_even_if_reverse_is_unsupported(self):
        with self.assertRaisesRegex(RuntimeError, "duplicate BDF"):
            self.check_with(make_acl(reverse_result=207000, duplicate=True))

    def test_wrong_or_failed_native_reverse_does_not_pass(self):
        with self.assertRaisesRegex(RuntimeError, "reverse lookup failed"):
            self.check_with(make_acl(reverse_device=7))
        with self.assertRaisesRegex(RuntimeError, "reverse lookup failed"):
            self.check_with(make_acl(reverse_result=123))

    def test_missing_or_partial_ascendcl_pair_uses_dcmi(self):
        for forward, reverse in ((False, False), (True, False), (False, True)):
            with self.subTest(forward=forward, reverse=reverse):
                provider, _ = self.check_with(
                    make_acl(include_forward=forward, include_reverse=reverse),
                    make_dcmi(),
                )
                self.assertIn("DCMI", provider)

    def test_other_ascendcl_error_does_not_use_dcmi(self):
        with self.assertRaisesRegex(RuntimeError, "failed: 123"):
            self.check_with(make_acl(forward_result=123), make_dcmi())

    def test_missing_or_failed_dcmi_is_reported(self):
        acl = make_acl(include_forward=False, include_reverse=False)
        with self.assertRaisesRegex(RuntimeError, "cannot load DCMI"):
            self.check_with(acl)
        with self.assertRaisesRegex(RuntimeError, "dcmi_init failed"):
            self.check_with(acl, make_dcmi(init_result=17))

    def test_duplicate_dcmi_bdf_is_rejected(self):
        acl = make_acl(include_forward=False, include_reverse=False)
        with self.assertRaisesRegex(RuntimeError, "duplicate BDF"):
            self.check_with(acl, make_dcmi(duplicate=True))

    def test_negative_dcmi_domain_is_rejected(self):
        acl = make_acl(include_forward=False, include_reverse=False)
        with self.assertRaisesRegex(RuntimeError, "invalid BDF"):
            self.check_with(acl, make_dcmi(domain=-1))


if __name__ == "__main__":
    unittest.main()
