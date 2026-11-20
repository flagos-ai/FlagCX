"""Compile the production VMM check against small, controlled SDK headers."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
CHECK = ROOT / 'makefiles/check_iluvatar_vmm.sh'
COMPILER = shutil.which('c++')

HEADER = '''
#include <stddef.h>
typedef int CUresult;
typedef unsigned long long CUdeviceptr;
typedef unsigned long long CUmemGenericAllocationHandle;
enum CUmemAllocationHandleType { CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 1 };
enum { CU_MEM_ALLOCATION_TYPE_PINNED = 1, CU_MEM_LOCATION_TYPE_DEVICE = 1,
       CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3, CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0,
       CU_POINTER_ATTRIBUTE_SYNC_MEMOPS = 1, CUDA_SUCCESS = 0,
       CUDA_ERROR_NOT_SUPPORTED = 801 };
struct CUmemLocation { int type; int id; };
struct CUmemAllocationProp {
  int type; CUmemLocation location; unsigned requestedHandleTypes;
};
struct CUmemAccessDesc { CUmemLocation location; unsigned flags; };
CUresult cuMemGetAllocationGranularity(size_t*, const CUmemAllocationProp*, int);
CUresult cuMemCreate(CUmemGenericAllocationHandle*, size_t, const CUmemAllocationProp*, unsigned long long);
CUresult cuMemAddressReserve(CUdeviceptr*, size_t, size_t, CUdeviceptr, unsigned long long);
CUresult cuMemMap(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, unsigned long long);
CUresult cuMemSetAccess(CUdeviceptr, size_t, const CUmemAccessDesc*, size_t);
CUresult cuPointerSetAttribute(const void*, int, CUdeviceptr);
CUresult cuMemGetAddressRange(CUdeviceptr*, size_t*, CUdeviceptr);
CUresult cuMemUnmap(CUdeviceptr, size_t);
CUresult cuMemAddressFree(CUdeviceptr, size_t);
CUresult cuMemRelease(CUmemGenericAllocationHandle);
CUresult cuMemExportToShareableHandle(void*, CUmemGenericAllocationHandle, CUmemAllocationHandleType, unsigned long long);
CUresult cuMemImportFromShareableHandle(CUmemGenericAllocationHandle*, void*, CUmemAllocationHandleType);
'''


@unittest.skipUnless(COMPILER, 'A host C++ compiler is required')
class IluvatarVmmCheckTest(unittest.TestCase):
    def check_sdk(self, header, expected, stage=None, missing=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'include').mkdir()
            (root / 'include/cuda.h').write_text(header)
            scalar = subprocess.run(['bash', str(CHECK), 'check', str(root), '--', COMPILER],
                                    text=True, capture_output=True, check=True)
            self.assertEqual(scalar.stdout.strip(), str(expected))
            logs = root / 'logs'
            result = subprocess.run(['bash', '-c', 'umask 077; exec "$@"', 'bash',
                                     'bash', str(CHECK), 'diagnose', str(root), str(logs),
                                     '--', COMPILER], text=True, capture_output=True, check=True)
            self.assertEqual((logs / 'result.env').read_text().strip(),
                             f'FLAGCX_ILUVATAR_HAS_VMM={expected}')
            self.assertEqual((logs / 'iluvatar_vmm_check.cpp').read_text(),
                             (ROOT / 'makefiles/iluvatar_vmm_check.cpp').read_text())
            if stage:
                self.assertNotEqual((logs / f'{stage}.status').read_text().strip(), '0')
                self.assertIn(missing, (logs / f'{stage}.log').read_text())
                self.assertIn(missing, result.stdout)
                if stage == 'basic':
                    self.assertEqual((logs / 'posix_fd.status').read_text().strip(), '0')
            self.assertEqual(logs.stat().st_mode & 0o777, 0o755)
            for path in logs.iterdir():
                self.assertEqual(path.stat().st_mode & 0o444, 0o444)

    def test_complete_sdk_passes(self):
        self.check_sdk(HEADER, 1)

    def test_missing_type_field_reports_the_compiler_error(self):
        self.check_sdk(HEADER.replace('unsigned requestedHandleTypes;', ''), 0,
                       'types', 'requestedHandleTypes')

    def test_missing_basic_api_is_distinct_from_fd_support(self):
        self.check_sdk(HEADER.replace('CUresult cuMemUnmap(CUdeviceptr, size_t);', ''),
                       0, 'basic', 'cuMemUnmap')

    def test_missing_fd_api_reports_the_compiler_error(self):
        header = '\n'.join(line for line in HEADER.splitlines()
                           if 'CUresult cuMemExportToShareableHandle' not in line)
        self.check_sdk(header, 0, 'posix_fd', 'cuMemExportToShareableHandle')

    def test_make_uses_the_host_compiler_and_shared_check(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'include').mkdir()
            (root / 'include/cuda.h').write_text(HEADER)
            makefile = root / 'Makefile'
            makefile.write_text(
                f'include {ROOT}/makefiles/iluvatar.mk\n'
                'report:\n\t@echo support=$(ILUVATAR_VMM_HEADER_SUPPORT)\n')
            result = subprocess.run(['make', '-f', str(makefile), 'report',
                                     f'DEVICE_HOME={root}', f'HOST_COMPILER={COMPILER}',
                                     'CXX=false'], text=True, capture_output=True, check=True)
            self.assertIn('support=1', result.stdout)


if __name__ == '__main__':
    unittest.main()
