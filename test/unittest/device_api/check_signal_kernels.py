#!/usr/bin/env python3
"""Compile/run the actual legacy, scalar and unified kernel bodies against a host protocol model.

The model checks ownership, thresholds and independent rank-barrier channels;
it is not a replacement for device compilation or GPU/NIC validation.
"""
import argparse
from pathlib import Path
import re
import subprocess
import tempfile


def definitions(source, name):
    bodies = []
    for match in re.finditer(r'\b' + name + r'\s*\([^;{}]*\)\s*\{', source):
        end, depth = match.end(), 1
        while depth:
            depth += (source[end] == '{') - (source[end] == '}')
            end += 1
        bodies.append(source[match.start():end])
    if not bodies:
        raise ValueError(f'missing definition: {name}')
    return bodies


def definition(source, name):
    return definitions(source, name)[0]


def check_iluvatar_bridges(root, cxx, temporary, sanitizer):
    """Compile both overloads and execute the real bridge/launcher forwarding."""
    source = (root / 'test/kernel/iluvatar/device_ir.cu').read_text()
    bodies, calls = [], []
    names = [f'launchKernelDev{variant}{team}WorldS'
             for team in ['Intra', 'Inter']
             for variant in ['PutSignalWait', 'PutRSig', 'PutCounter']]
    for index, name in enumerate(names):
        overloads = definitions(source, name)
        if len(overloads) != 2:
            raise ValueError(f'{name}: expected AS1 launcher and host bridge')
        launcher, bridge = overloads
        kernel = name.replace('launchKernel', 'kernel', 1)
        signature = definition(source, kernel).split('{', 1)[0]
        bodies.append('void ' + signature + '{\n'
                      f'  target = {index};\n'
                      '  round = selectedVariant; length = bytes;\n'
                      '  delay = skew; output = result;\n}')
        # Replace the SDK launch with a recorder call; retain the real AS1
        # parameter types, generic overload and explicit address-space casts.
        bodies.append('void ' + re.sub(r'<<<.*?>>>', '', launcher, flags=re.S))
        bodies.append('void ' + bridge)
        variants = 3 if 'PutCounter' in name else 2
        calls.append(f'  Bridge bridge{index} = static_cast<Bridge>(&{name});\n'
                     f'  for (int v = 0; v < {variants}; ++v) {{\n'
                     '    target = round = -1;\n'
                     f'    bridge{index}(nullptr, nullptr, nullptr, &result,\n'
                     f'                   {1024 + index}, nullptr, {17 + index}, v);\n'
                     f'    assert(target == {index} && round == v);\n'
                     f'    assert(length == {1024 + index} && delay == {17 + index});\n'
                     '    assert(output == &result);\n  }')
    code = ('#include <cassert>\n#include <cstddef>\n'
            'using flagcxStream_t = void *;\n'
            'int target, round; size_t length; unsigned delay; int *output;\n' +
            '\n\n'.join(bodies) + '\nint main() {\n  int result = 0;\n'
            '  using Bridge = void (*)(const void *, const void *, const void *,\n'
            '                          int *, size_t, flagcxStream_t, unsigned, int);\n' +
            '\n'.join(calls) + '\n}\n')
    compiler = subprocess.run([cxx, '--version'], capture_output=True,
                              text=True, check=True).stdout
    if 'clang' in compiler.lower():
        code = '#define FLAGCX_IR_GLOBAL_PTR __attribute__((address_space(1)))\n' + code
    else:
        # GCC ignores address_space attributes. Preserve overload separation
        # with an explicitly constructed pointer model instead of erasing AS1.
        code = re.sub(r'const\s+void\s+FLAGCX_IR_GLOBAL_PTR\s*\*', 'As1Pointer ', code)
        code = ('struct As1Pointer { explicit As1Pointer(const void *) {} };\n' + code)
    probe = temporary / 'iluvatar_bridges.cpp'
    binary = temporary / 'iluvatar_bridges'
    probe.write_text(code)
    subprocess.run([cxx, '-std=c++14', '-O1', *sanitizer, str(probe),
                    '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True, timeout=10)
    print('actual Iluvatar bridges: PASS (six overload pairs, all variants)', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cxx', default='c++')
    parser.add_argument('--sanitize', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    names = ['ipcAlltoAll', 'flagcxInterTestResetKernel',
             'flagcxInterTestPutSignalIncKernel',
             'flagcxInterTestPutSignalAddDecoupledKernel',
             'flagcxInterTestMultiContextPutReleaseKernel',
             'flagcxInterTestCounterPipelineKernel',
             'flagcxInterTestFlushDecoupleKernel', 'flagcxInterTestPutValueKernel',
             'flagcxInterTestSignalKernel', 'flagcxInterTestWaitSignalFlushKernel',
             'flagcxInterOneSidedAlltoAllKernel']
    reference = None
    normalized_reference = None
    for platform, extension in [('nvidia', 'cu'), ('du', 'hip'),
                                 ('ppu', 'cu'), ('iluvatar', 'cu')]:
        source = (root / f'test/kernel/{platform}/device_api.{extension}').read_text()
        bodies = ['void ' + definition(source, name) for name in names]
        normalized = [re.sub(r'\s+', '', re.sub(r'//[^\n]*', '', body)) for body in bodies]
        if normalized_reference is not None and normalized != normalized_reference:
            raise ValueError(f'{platform}: legacy protocol differs from NVIDIA')
        normalized_reference = normalized
        if platform == 'nvidia':
            reference = bodies
    irnames = ['kernelNetResetS', 'kernelNetPutSignalIncS',
               'kernelNetPutSignalAddS', 'kernelNetCounterPipelineS',
               'kernelNetFlushDecoupleS', 'kernelNetPutValueS', 'kernelNetSignalS',
               'kernelNetWaitSignalFlushS', 'kernelNetOneSidedAlltoAllS']
    irnames += [f'kernelDev{variant}{team}WorldS'
                for team in ['Intra', 'Inter']
                for variant in ['SignalStandalone', 'PutSignalWait', 'PutRSig',
                                'PutCounter', 'PutValueRSig', 'SignalShadowFlush']]
    # Compile the changed launcher calls as ordinary host calls as well: an
    # omitted round argument must fail here even without a device compiler.
    irnames += [f'launchKernelDev{variant}{team}WorldS'
                for team in ['Intra', 'Inter']
                for variant in ['PutSignalWait', 'PutRSig', 'PutCounter']]
    ir_reference = None
    for platform, extension in [('nvidia', 'cu'), ('du', 'hip'),
                                 ('ppu', 'cu'), ('iluvatar', 'cu')]:
        source = (root / f'test/kernel/{platform}/device_ir.{extension}').read_text()
        bodies = ['void ' + definition(source, name) for name in irnames]
        # A flush by an early CTA cannot cover a late CTA's future requests.
        for team in ['Intra', 'Inter']:
            for variant in ['Put', 'PutValue', 'Get']:
                body = definition(source, f'kernelDev{variant}{team}WorldS')
                if 'myBlockIdx < nContexts' in body:
                    raise ValueError(f'{platform}: shared-context producer skips flush')

        # Ignore historical blank-line/comment differences between platforms.
        def normalize(body):
            text = re.sub(r'\s+', '', re.sub(r'//[^\n]*', '', body)).replace('\\', '')
            # PPU materializes macro arguments to work around its compiler.
            text = text.replace('constintsignalMode=(int)(signalVariant);', '')
            return text.replace('signalMode', '(signalVariant)')
        normalized = [normalize(body) for body in bodies]
        if ir_reference is not None and normalized != ir_reference:
            raise ValueError(f'{platform}: scalar/unified protocol differs from NVIDIA')
        ir_reference = normalized
        if platform == 'nvidia':
            ir_bodies = [re.sub(r'<<<.*?>>>', '', body, flags=re.S)
                         for body in bodies]
    reference += ir_bodies
    source = (root / 'test/kernel/klx/device_ir.xpu').read_text()
    for result, name in [('bool', 'irCoopActive'),
                         ('bool', 'irPutCoopSupported'),
                         ('IrPrimaryTeam', 'irPrimaryTeam')]:
        reference.append(result + ' ' + definition(source, name))
    reference += ['void ' + definition(source, f'kernelDev{variant}UnifiedS')
                  for variant in ['PutSignalWait', 'PutRSig', 'PutCounter']]
    # The other platform launchers must forward the round argument and retain
    # the same public test signature, including unsupported GIN stubs.
    reference.append('#include <type_traits>')
    for platform, extension in [('klx', 'xpu'), ('enflame', 'tops')]:
        source = (root / f'test/kernel/{platform}/device_ir.{extension}').read_text()
        reference.append(f'namespace {platform}Launch {{')
        if platform == 'klx':
            reference.append('using XPUStream = void *;\n'
                             'void irWait(XPUStream, const char *) {}')
        for team in ['Intra', 'Inter']:
            for variant in ['PutSignalWait', 'PutRSig', 'PutCounter']:
                name = f'launchKernelDev{variant}{team}WorldS'
                body = re.sub(r'<<<.*?>>>', '', definition(source, name), flags=re.S)
                reference.append('void ' + body)
                reference.append('static_assert(std::is_same<decltype(&' + name +
                                 '), decltype(&::' + name + ')>::value, '
                                 '"platform launcher signature mismatch");')
        reference.append('}')
    with tempfile.TemporaryDirectory(prefix='flagcx-signal-') as directory:
        temporary = Path(directory)
        (temporary / 'signal_test_kernels.inc').write_text('\n\n'.join(reference))
        binary = temporary / 'signal_kernel_sim'
        sanitizer = ['-fsanitize=address,undefined', '-fno-omit-frame-pointer'] if args.sanitize else []
        check_iluvatar_bridges(root, args.cxx, temporary, sanitizer)
        subprocess.run([args.cxx, '-std=c++14', '-pthread', '-O1', *sanitizer,
                        '-I' + str(temporary),
                        '-I' + str(root / 'test/kernel/include'),
                        str(Path(__file__).with_name('signal_kernel_sim.cpp')),
                        '-o', str(binary)], check=True)
        subprocess.run([str(binary)], check=True, timeout=60)


if __name__ == '__main__':
    main()
