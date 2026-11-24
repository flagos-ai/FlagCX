"""Compile and link the selected adaptor's typed capture query; never execute it.

A failed probe means unknown/unsupported introspection, not inactive capture.
This builds only a temporary executable and does not require a device.
"""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import tempfile

# SDK headers/types are candidates: only a successful native compile+link
# enables the corresponding production function. No ABI is guessed at runtime.
PLATFORMS = [('USE_NVIDIA_ADAPTOR', 'cuda', 'cuda', 'cuda_runtime_api.h', 'cudaStream_t'),
 ('USE_AMD_ADAPTOR', 'hip', 'hip', 'hip/hip_runtime_api.h', 'hipStream_t'),
 ('USE_DU_ADAPTOR', 'duhip', 'duhip', 'hip/hip_runtime_api.h', 'hipStream_t'),
 ('USE_ILUVATAR_ADAPTOR', 'ixcuda', 'ixcuda', 'cuda_runtime_api.h', 'cudaStream_t'),
 ('USE_MUSA_ADAPTOR', 'musa', 'musa', 'musa_runtime_api.h', 'musaStream_t'),
 ('USE_METAX_ADAPTOR', 'maca', 'maca', 'mcr/mc_runtime.h', 'mcStream_t'),
 ('USE_CAMBRICON_ADAPTOR', 'mlu', 'mlu', 'cnrt.h', 'cnrtQueue_t'),
 ('USE_ASCEND_ADAPTOR', 'cann', 'cann', 'acl/acl.h', 'aclrtStream'),
 ('USE_ENFLAME_ADAPTOR', 'tops', 'tops', 'tops/tops_runtime.h', 'topsStream_t'),
 ('USE_SUNRISE_ADAPTOR', 'ptpu', 'ptpu', 'tang_runtime.h', 'tangStream_t'),
 ('USE_TSM_ADAPTOR', 'tsmicro', 'tsmicro', 'tx_runtime.h', 'txStream_t'),
 ('USE_KUNLUNXIN_ADAPTOR', 'kunlunxin', 'kunlun', 'cuda_runtime_api.h', 'cudaStream_t'),
 ('USE_PPU_ADAPTOR', 'hggc', 'hggc', 'hggc_runtime.h', 'hggcStream_t')]

def probe(root, flags, cxx, cppflags, ldflags):
    definitions = {token.split('=', 1)[0] for token in shlex.split(flags)}
    selected = next((row for row in PLATFORMS if '-D' + row[0] in definitions), None)
    if selected is None:
        return {"supported": False, "reason": "no native adaptor selected"}
    _, file, adaptor, header, stream = selected
    source = (root / f'flagcx/adaptor/device/{file}_adaptor.cc').read_text()
    name = adaptor + 'AdaptorStreamGetCaptureStatus'
    match = re.search(r'static\s+flagcxResult_t\s+' + name + r'\s*\(', source)
    if match is None:
        return {"supported": False, "reason": "query not implemented", "platform": file}
    opening = source.index('{', match.end())
    depth, end = 1, opening + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    code = f'#include <{header}>\n#include "flagcx_device_adaptor.h"\n'
    code += f'struct flagcxStream {{ {stream} base; }};\n'
    code += source[match.start():end]
    code += f'\nint main() {{ flagcxStreamCaptureStatus_t status; return {name}(NULL, &status); }}\n'
    with tempfile.TemporaryDirectory(prefix='flagcx-capture-') as directory:
        cpp = Path(directory) / 'query.cc'
        binary = Path(directory) / 'query'
        cpp.write_text(code)
        command = shlex.split(cxx) + ['-std=c++17',
                   '-I' + str(root / 'flagcx/include'), '-I' + str(root / 'flagcx/adaptor/include')]
        command += shlex.split(cppflags) + ['-DFLAGCX_NATIVE_CAPTURE_QUERY=1', str(cpp), '-o', str(binary)] + shlex.split(ldflags)
        try:
            result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, timeout=45)
            return {"supported": result.returncode == 0, "platform": file,
                    "command": [arg.replace(directory, "<probe>") for arg in command],
                    "diagnostic": result.stderr.replace(directory, "<probe>")}
        except (OSError, subprocess.TimeoutExpired) as error:
            return {"supported": False, "platform": file, "reason": str(error)}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--flags', default='')
    parser.add_argument('--cxx', default='g++')
    parser.add_argument('--cppflags', default='')
    parser.add_argument('--ldflags', default='')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    result = probe(args.root, args.flags, args.cxx, args.cppflags, args.ldflags)
    if args.report:
        try:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            content = json.dumps(result, indent=2) + '\n'
            if not args.report.exists() or args.report.read_text() != content:
                args.report.write_text(content)
        except OSError:
            pass  # Read-only make/dry-run must still be able to inspect targets.
    print(int(result['supported']))
