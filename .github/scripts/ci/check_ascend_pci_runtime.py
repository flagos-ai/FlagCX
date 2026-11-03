#!/usr/bin/env python3
"""Verify a visible Ascend device has a working PCI forward/reverse lookup."""

import ctypes
import sys


ACL_FEATURE_NOT_SUPPORTED = 207000


class DcmiPcieInfo(ctypes.Structure):
    _fields_ = [
        ("vendor_id", ctypes.c_uint),
        ("subvendor_id", ctypes.c_uint),
        ("device_id", ctypes.c_uint),
        ("subdevice_id", ctypes.c_uint),
        ("domain", ctypes.c_int),
        ("bus", ctypes.c_uint),
        ("device", ctypes.c_uint),
        ("function", ctypes.c_uint),
        ("reserved", ctypes.c_ubyte * 32),
    ]


def require_symbol(library, name):
    symbol = getattr(library, name, None)
    if symbol is None:
        raise RuntimeError(f"required symbol {name} is unavailable")
    return symbol


def check_dcmi(acl, load, count):
    user_to_logic = require_symbol(acl, "aclrtGetLogicDevIdByUserDevId")
    user_to_logic.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_int32)]
    user_to_logic.restype = ctypes.c_int

    errors = []
    dcmi = None
    for name in ("libdcmi.so", "libdcmi.so.1"):
        try:
            dcmi = load(name)
            break
        except OSError as error:
            errors.append(f"{name}: {error}")
    if dcmi is None:
        raise RuntimeError("cannot load DCMI: " + "; ".join(errors))

    dcmi_init = require_symbol(dcmi, "dcmi_init")
    logic_to_card = require_symbol(dcmi, "dcmi_get_card_id_device_id_from_logicid")
    pcie_info = require_symbol(dcmi, "dcmi_get_device_pcie_info_v2")
    dcmi_init.argtypes = []
    dcmi_init.restype = ctypes.c_int
    logic_to_card.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_uint,
    ]
    logic_to_card.restype = ctypes.c_int
    pcie_info.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(DcmiPcieInfo),
    ]
    pcie_info.restype = ctypes.c_int
    result = dcmi_init()
    if result != 0:
        raise RuntimeError(f"dcmi_init failed with error {result}")

    seen = set()
    for device_id in range(count):
        logic_id = ctypes.c_int32(-1)
        result = user_to_logic(device_id, ctypes.byref(logic_id))
        if result != 0 or logic_id.value < 0:
            raise RuntimeError(f"device {device_id} logic-ID query failed: {result}")
        card_id = ctypes.c_int(-1)
        chip_id = ctypes.c_int(-1)
        result = logic_to_card(
            ctypes.byref(card_id), ctypes.byref(chip_id), logic_id.value
        )
        if result != 0 or card_id.value < 0 or chip_id.value < 0:
            raise RuntimeError(f"device {device_id} DCMI card lookup failed: {result}")
        info = DcmiPcieInfo()
        result = pcie_info(card_id.value, chip_id.value, ctypes.byref(info))
        if result != 0:
            raise RuntimeError(f"device {device_id} DCMI PCI query failed: {result}")
        if (
            info.domain < 0
            or info.bus > 0xFF
            or info.device > 0x1F
            or info.function > 7
        ):
            raise RuntimeError(f"device {device_id} DCMI returned an invalid BDF")
        digits = 4 if info.domain <= 0xFFFF else 8
        bus_id = (
            f"{info.domain:0{digits}x}:{info.bus:02x}:"
            f"{info.device:02x}.{info.function:x}"
        )
        if bus_id in seen:
            raise RuntimeError(f"DCMI returned duplicate BDF {bus_id}")
        seen.add(bus_id)
    return "DCMI PCI lookup (verified on visible devices)"


def check_runtime(load=ctypes.CDLL):
    try:
        acl = load("libascendcl.so")
    except OSError as error:
        raise RuntimeError(f"cannot load libascendcl.so: {error}") from error

    acl_init = require_symbol(acl, "aclInit")
    acl_finalize = require_symbol(acl, "aclFinalize")
    device_count = require_symbol(acl, "aclrtGetDeviceCount")
    acl_init.argtypes = [ctypes.c_char_p]
    acl_init.restype = ctypes.c_int
    acl_finalize.argtypes = []
    acl_finalize.restype = ctypes.c_int
    device_count.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
    device_count.restype = ctypes.c_int
    result = acl_init(None)
    if result != 0:
        raise RuntimeError(f"aclInit failed with error {result}")

    try:
        count = ctypes.c_uint32(0)
        result = device_count(ctypes.byref(count))
        if result != 0 or count.value == 0:
            raise RuntimeError(f"aclrtGetDeviceCount failed or found no devices: {result}")

        forward = getattr(acl, "aclrtDeviceGetPCIBusId", None)
        reverse = getattr(acl, "aclrtDeviceGetByPCIBusId", None)
        if forward is not None and reverse is not None:
            forward.argtypes = [ctypes.c_int32, ctypes.c_char_p, ctypes.c_int32]
            forward.restype = ctypes.c_int
            reverse.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_int32)]
            reverse.restype = ctypes.c_int
            seen = set()
            for device_id in range(count.value):
                bus_id = ctypes.create_string_buffer(32)
                result = forward(device_id, bus_id, len(bus_id))
                if result == ACL_FEATURE_NOT_SUPPORTED:
                    break
                if result != 0 or not bus_id.value:
                    raise RuntimeError(
                        f"device {device_id} AscendCL PCI query failed: {result}"
                    )
                if bus_id.value in seen:
                    raise RuntimeError(f"AscendCL returned duplicate BDF {bus_id.value!r}")
                seen.add(bus_id.value)
                found = ctypes.c_int32(-1)
                result = reverse(bus_id.value, ctypes.byref(found))
                if result == ACL_FEATURE_NOT_SUPPORTED:
                    break
                if result != 0 or found.value != device_id:
                    raise RuntimeError(
                        f"device {device_id} AscendCL PCI reverse lookup failed: {result}"
                    )
            else:
                return "AscendCL PCI lookup (forward/reverse verified)"

        return check_dcmi(acl, load, count.value)
    finally:
        acl_finalize()


if __name__ == "__main__":
    try:
        print("Ascend CI PCI provider: " + check_runtime())
    except RuntimeError as error:
        print(f"Ascend CI PCI preflight failed: {error}", file=sys.stderr)
        sys.exit(1)
