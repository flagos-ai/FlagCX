#!/usr/bin/env python3
"""Compile/link HGGC capability probes without executing device operations."""
import argparse
import os
import shlex
from pathlib import Path
import subprocess
import tempfile

HEADERS = '#include <hggc.h>\n#include <hggc_runtime.h>\n#include <pccl.h>\n'
PROBES = {
    'MANAGED': """
      void *p;
      return int(hggcMallocManaged(&p, 1, hggcMemAttachGlobal));
    """,
    'WAIT_VALUE64': """
      return int(hgStreamWaitValue64(HGstream{}, HGdeviceptr{}, 1,
                                    HG_STREAM_WAIT_VALUE_GEQ));
    """,
    'WAIT_VALUE64_FLUSH': """
      return int(hgStreamWaitValue64(HGstream{}, HGdeviceptr{}, 1,
          HG_STREAM_WAIT_VALUE_GEQ | HG_STREAM_WAIT_VALUE_FLUSH));
    """,
    'WRITE_VALUE64': """
      return int(hgStreamWriteValue64(HGstream{}, HGdeviceptr{}, 1,
                                     HG_STREAM_WRITE_VALUE_DEFAULT));
    """,
    'DMA_BUF': """
      int v;
      hgDeviceGetAttribute(&v, HG_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, HGdevice{});
      return int(hgMemGetHandleForAddressRange(&v, HGdeviceptr{}, 1,
                                             HG_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0));
    """,
    'VMM_RDMA': """
      HGmemAllocationProp p{};
      p.allocFlags.gpuDirectRDMACapable = 1;
      int v;
      return int(hgDeviceGetAttribute(&v,
                                     HG_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED,
                                     HGdevice{}));
    """,
    'POINTER_TYPE': """
      hggcPointerAttributes a{};
      return int(a.type == hggcMemoryTypeDevice || a.type == hggcMemoryTypeManaged);
    """,
}


# Mandatory ABI declarations and symbols. In particular IPC and VMM cannot be
# silently disabled, since the existing default Device API depends on them.
REQUIRED = '''hggcDeviceSynchronize hggcGetLastError hggcGetDevice hggcSetDevice hggcGetDeviceCount
hggcMalloc hggcFree hggcFreeAsync hggcMemcpy hggcMemcpyAsync hggcMemset hggcMemsetAsync
hggcHostAlloc hggcFreeHost hggcHostRegister hggcHostUnregister hggcHostGetDevicePointer
hggcStreamCreateWithFlags hggcStreamDestroy hggcStreamSynchronize hggcStreamQuery
hggcStreamWaitEvent hggcEventCreateWithFlags hggcEventRecord hggcEventSynchronize
hggcEventDestroy hggcEventQuery hggcEventElapsedTime hggcIpcGetMemHandle
hggcIpcOpenMemHandle hggcIpcCloseMemHandle hggcLaunchHostFunc hggcPointerGetAttributes
hggcGetDeviceProperties hggcDeviceGetPCIBusId hggcDeviceGetByPCIBusId hggcDeviceCanAccessPeer
hggcMemGetInfo hgDeviceGet hgPointerSetAttribute hgMemCreate hgMemRelease
hgMemAddressReserve hgMemAddressFree hgMemMap hgMemUnmap hgMemSetAccess
hgMemImportFromShareableHandle hgMemExportToShareableHandle hgMemRetainAllocationHandle
hgMemGetAddressRange hgMemGetAllocationGranularity pcclGetVersion pcclGetUniqueId
pcclCommInitRank pcclCommFinalize pcclCommDestroy pcclCommAbort pcclCommCount
pcclCommCuDevice pcclCommUserRank pcclCommGetAsyncError pcclGetErrorString pcclGetLastError
pcclReduce pcclBroadcast pcclAllReduce pcclReduceScatter pcclAllGather
pcclSend pcclRecv pcclGroupStart pcclGroupEnd'''.split()

# These names also have typed template wrappers in HGGC headers. Match the
# adaptor's call sites so overload resolution selects the supported interface.
# Bare `auto symbol = &name` cannot resolve an overload set.
REQUIRED_CALLS = {
    'hggcMalloc': 'hggcMalloc(&ptr, 1)',
    'hggcHostAlloc': 'hggcHostAlloc(&ptr, 1, hggcHostAllocMapped)',
    'hggcHostGetDevicePointer':
        'hggcHostGetDevicePointer(&ptr, static_cast<void *>(nullptr), 0)',
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cxx', required=True)
    p.add_argument('--include', action='append', default=[])
    p.add_argument('--library', action='append', default=[])
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        source, binary = tmp / 'probe.cc', tmp / 'probe'
        command = shlex.split(args.cxx) + ['-std=c++11', str(source), '-o', str(binary)]
        command += ['-I' + path for path in args.include]
        command += ['-L' + path for path in args.library]
        command += ['-lpccl', '-lhggc_wrapper', '-lhg_wrapper']

        def compile_probe(body):
            source.write_text(HEADERS + body)
            return subprocess.run(command, capture_output=True, text=True)

        # Volatile globals keep mandatory symbol references alive at link time.
        declarations = '\n'.join(
            'auto volatile symbol_' + name + ' = &' + name + ';'
            for name in REQUIRED if name not in REQUIRED_CALLS
        )
        calls = '\n'.join('  result |= int(' + call + ');'
                          for call in REQUIRED_CALLS.values())
        # The binary is linked but never executed. Keep the call results in a
        # volatile local to retain the selected SDK entry points at link time.
        result = compile_probe(declarations + '''
int main() {
  void *ptr = nullptr;
  volatile int result = 0;
''' + calls + '''
  return result | int(hggcMallocAsync(&ptr, 1, hggcStream_t{}));
}
''')
        if result.returncode:
            raise SystemExit('HGGC/PCCL required interface compile/link check failed:\n' + result.stderr)
        values = {}
        for name, body in PROBES.items():
            result = compile_probe('int main() { ' + body + ' }\n')
            values[name] = int(result.returncode == 0)
            print(f'HGGC {name}: {values[name]}')
        content = '// Generated by probe_hggc_sdk.py; compile/link checks only.\n#pragma once\n'
        content += ''.join(f'#define FLAGCX_HGGC_HAS_{name} {value}\n' for name, value in values.items())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(content)
        os.replace(temporary, args.output)


if __name__ == '__main__':
    main()
