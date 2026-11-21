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


def definition(source, name):
    match = re.search(r'\b' + name + r'\s*\([^;{}]*\)\s*\{', source)
    if not match:
        raise ValueError(f'missing definition: {name}')
    end, depth = match.end(), 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[match.start():end]


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
        subprocess.run([args.cxx, '-std=c++14', '-pthread', '-O1', *sanitizer,
                        '-I' + str(temporary),
                        '-I' + str(root / 'test/kernel/include'),
                        str(Path(__file__).with_name('signal_kernel_sim.cpp')),
                        '-o', str(binary)], check=True)
        subprocess.run([str(binary)], check=True, timeout=60)


if __name__ == '__main__':
    main()
