#!/usr/bin/env python3
"""Generate device-only subsets and test driver loading, never launch kernels."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import zipfile


def masked(source):
    # Keep offsets/newlines stable while hiding braces in comments/strings.
    token = r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\''
    return re.sub(token, lambda m: re.sub(r'[^\n]', ' ', m[0]), source)


def body_end(code, start):
    opening = code.index('{', start)
    depth = 1
    for index in range(opening + 1, len(code)):
        depth += (code[index] == '{') - (code[index] == '}')
        if depth == 0:
            return index + 1
    raise ValueError('Unbalanced source braces')


def extract(source):
    code = masked(source)
    pattern = r'(?m)^(?:template\s*<[^>]+>\s*)?(?:__global__|FLAGCX_GLOBAL_DECORATOR)\b'
    kernels = {}
    starts = []
    template_launcher = ''
    for match in re.finditer(pattern, code):
        end = body_end(code, match.start())
        header = code[match.start():code.index('{', match.start())]
        name = re.search(r'\b(flagcx\w+Kernel)\s*\(', header)
        if not name or name[1] in kernels:
            raise ValueError('Unrecognized or duplicate global kernel definition')
        kernels[name[1]] = source[match.start():end]
        starts.append(match.start())
        if header.lstrip().startswith('template'):
            # Device-only compilation does not reliably instantiate kernels
            # through explicit instantiation of an unused static host launcher.
            if name[1] != 'flagcxIntraAllReduceKernel':
                raise ValueError('Unexpected template kernel; update extractor')
            boundary = re.search(r'(?m)^flagcxResult_t\s+launchKernelIntraAllReduce\s*\(', code[end:])
            if not boundary:
                raise ValueError('Cannot locate original template launcher')
            original_launcher = source[end:end + boundary.start()]
            if not all('launchFlagcxIntraAllReduce<' + dtype + '>' in original_launcher
                       for dtype in ('float', 'double')):
                raise ValueError('Original template instantiation types changed')
            template_launcher = '\n'.join(
                'template __global__ void flagcxIntraAllReduceKernel<' + dtype +
                '>(flagcxDevComm, flagcxDevMem, size_t, size_t);'
                for dtype in ('float', 'double'))
    if not kernels or len(kernels) != len(re.findall(r'\b(?:__global__|FLAGCX_GLOBAL_DECORATOR)\b', code)):
        raise ValueError('Kernel inventory mismatch; refusing an incomplete extraction')
    # A fixed source may use concrete entry points and need no instantiation.
    # Retain shared device functions defined between kernels, not just the
    # helpers in the initial header prefix. In this source ipcAlltoAll is used
    # by five kernels; dropping it creates artificial compilation failures.
    helpers = []
    for match in re.finditer(r'(?m)^FLAGCX_DEVICE(?:_INLINE)?_DECORATOR\b', code):
        if match.start() >= starts[0]:
            helpers.append(source[match.start():body_end(code, match.start())])
    prefix = source[:starts[0]] + '\n\n' + '\n\n'.join(helpers)
    if any('ipcAlltoAll(' in body for body in kernels.values()) and not re.search(
            r'\bipcAlltoAll\s*\([^;]*?\)\s*\{', masked(prefix)):
        raise ValueError('Required ipcAlltoAll device helper was not preserved')
    return prefix, kernels, template_launcher


def invoke(command, logfile, timeout):
    with logfile.open('w', encoding='utf-8') as log:
        log.write(json.dumps(command) + '\n')
        log.flush()
        process = None
        try:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=(os.name == 'posix'))
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Kill the compiler/linker too, not just their parent make process.
            if os.name == 'posix':
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
            process.wait()
            log.write('\nTIMEOUT\n')
            return 124


def kernel_symbols(nm_text, selected):
    symbols, found = [], set()
    for line in nm_text.splitlines():
        fields = line.split()
        # Explicit template instantiations are defined weak functions (W).
        # nm is invoked with --defined-only, excluding undefined weak imports.
        if len(fields) != 3 or fields[1] not in ('T', 't', 'W', 'w') or 'ConstCalcProg' in fields[2]:
            continue
        for name in selected:
            if name in fields[2]:
                symbols.append(fields[2])
                found.add(name)
                break
    return symbols, set(selected) - found


def case_key(case):
    # Focus variants can select the same kernel name but different code/types.
    return ('variant', case['variant']) if 'variant' in case else tuple(case['kernels'])


def recheck(args, root):
    previous = Path(args.recheck).resolve()
    manifest = json.loads((previous / 'manifest.json').read_text(encoding='utf-8'))
    output = previous / ('recheck_' + time.strftime('%Y%m%d_%H%M%S'))
    output.mkdir(exist_ok=False)
    diagnostic = root / 'build/bin/device_api_module_diag'
    if not diagnostic.is_file():
        raise ValueError('Build module-diag first')
    rows, images, cached = [], {}, {}
    print('RECHECK_OUTPUT=' + str(output), flush=True)
    for case in manifest['cases']:
        key = case_key(case)
        image = previous / (case['label'] + '.mubin')
        if image.is_file():
            images.setdefault(key, image)
    cases = [{'label': 'original', 'kernels': ['flagcxIntraTestCommQueriesKernel']}]
    cases += manifest['cases']
    for case in cases:
        label, selected = case['label'], case['kernels']
        key = case_key(case)
        image = root / 'build/obj/device_api.mubin' if label == 'original' else images.get(key)
        detail = ''
        if label != 'original' and key in cached:
            status, code, original = cached[key]
            detail = 'reused=' + original
        elif image is None or not image.is_file():
            status, code = 'IMAGE_MISSING', 1
        else:
            nm_log = output / (label + '.nm.log')
            code = invoke(['nm', '--defined-only', str(image)], nm_log, args.load_timeout)
            if code:
                status = 'SYMBOL_SCAN_FAIL'
            else:
                symbols, missing = kernel_symbols(nm_log.read_text(
                    encoding='utf-8', errors='replace'), selected)
                if missing:
                    status, code, detail = 'MISSING_KERNEL_SYMBOL', 1, ','.join(sorted(missing))
                else:
                    code = invoke([str(diagnostic), str(args.gpu), str(image)] + symbols,
                                  output / (label + '.load.log'), args.load_timeout)
                    status = 'LOAD_LOOKUP_OK' if code == 0 else 'LOAD_LOOKUP_FAIL'
            if label != 'original':
                cached[key] = status, code, label
        rows.append({'case': label, 'status': status, 'exit': code,
                     'kernels': ','.join(selected), 'detail': detail})
        with (output / 'results.tsv').open('w', newline='', encoding='utf-8') as table:
            writer = csv.DictWriter(table, fieldnames=list(rows[-1]), delimiter='\t')
            writer.writeheader()
            writer.writerows(rows)
        print(label + ': ' + status + (' ' + detail if detail else ''), flush=True)
    counts = {}
    for row in rows:
        counts[row['status']] = counts.get(row['status'], 0) + 1
    (output / 'summary.json').write_text(json.dumps(counts, indent=2), encoding='utf-8')
    (output / 'manifest.json').write_text(json.dumps(
        {'batch_version': 3, 'gpu': args.gpu, 'previous': str(previous),
         'original_manifest': manifest}, indent=2), encoding='utf-8')
    with zipfile.ZipFile(output / 'diagnostics.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(output.iterdir()):
            if file.suffix in ('.log', '.tsv', '.json'):
                archive.write(file, file.name)
    print('SUMMARY=' + json.dumps(counts), flush=True)
    print('Download: ' + str(output / 'diagnostics.zip'), flush=True)
    print('Existing images only; no compilation and no Kernel execution.', flush=True)
    return 0 if all(row['exit'] == 0 for row in rows) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device-home', default='/usr/local/musa')
    parser.add_argument('--arch', default='mp_31')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--build-timeout', type=int, default=180)
    parser.add_argument('--load-timeout', type=int, default=30)
    parser.add_argument('--make', default='make')
    parser.add_argument('--output')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--recheck', help='Recheck existing batch images without compiling')
    parser.add_argument('--focus-allreduce', action='store_true',
                        help='Seven focused dtype/template/full-image comparisons')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if args.recheck:
        return recheck(args, root)
    source_path = root / 'device_api.cu'
    source = source_path.read_text(encoding='utf-8')
    prefix, kernels, template_launcher = extract(source)
    anchor = 'flagcxIntraTestCommQueriesKernel'
    if anchor not in kernels:
        raise ValueError('Formal CommQueries kernel is missing')
    output = Path(args.output).resolve() if args.output else root / 'build' / ('image_batch_' + time.strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True, exist_ok=False)
    (output / 'device_api.original.cu').write_text(source, encoding='utf-8')
    names = list(kernels)
    cases = [('generated_all', names)]
    cases += [('single_%02d' % i, [name]) for i, name in enumerate(names)]
    width = (len(names) + 3) // 4
    cases += [('group_%02d' % i, list(dict.fromkeys([anchor] + names[start:start + width])))
              for i, start in enumerate(range(0, len(names), width))]
    accumulated = [anchor]
    for i, name in enumerate(n for n in names if n != anchor):
        accumulated = accumulated + [name]
        cases.append(('prefix_%02d' % i, accumulated))
    allreduce = 'flagcxIntraAllReduceKernel'
    if args.focus_allreduce:
        if allreduce not in kernels:
            raise ValueError('Focused template comparison requires the pre-fix source; use the normal batch on concrete kernels')
        cases = [('without_allreduce', [n for n in names if n != allreduce]),
                 ('allreduce_float', [allreduce]),
                 ('allreduce_double', [allreduce]),
                 ('allreduce_concrete_float', [allreduce]),
                 ('allreduce_concrete_double', [allreduce]),
                 ('full_float', names), ('full_concrete_float', names)]
    manifest = {'batch_version': 3,
                'source_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
                'gpu': args.gpu, 'arch': args.arch, 'device_home': args.device_home,
                'focus_allreduce': args.focus_allreduce,
                'kernels': names, 'cases': []}
    for label, selected in cases:
        # Original source order, signatures, mangling and kernel bodies survive.
        parts = [prefix]
        for name in names:
            if name in selected:
                body = kernels[name]
                if args.focus_allreduce and name == allreduce and 'concrete' in label:
                    dtype = 'double' if label.endswith('double') else 'float'
                    body, removed = re.subn(r'^template\s*<typename T>\s*', '', body, count=1)
                    if removed != 1:
                        raise ValueError('Cannot concretize original AllReduce template')
                    body = re.sub(r'\bT\b', dtype, body)
                parts.append(body)
                if name == 'flagcxIntraAllReduceKernel':
                    if not args.focus_allreduce:
                        parts.append(template_launcher)
                    elif 'concrete' not in label:
                        dtype = 'double' if label.endswith('double') else 'float'
                        parts.append('\n'.join(line for line in template_launcher.splitlines()
                                               if '<' + dtype + '>' in line))
        generated = '\n\n'.join(parts) + '\n'
        (output / (label + '.cu')).write_text(generated, encoding='utf-8')
        case = {'label': label, 'kernels': [n for n in names if n in selected]}
        if args.focus_allreduce:
            case['variant'] = label
        manifest['cases'].append(case)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('OUTPUT=' + str(output), flush=True)
    print('%d kernel definitions, %d scheduled cases; load/lookup only.' % (len(names), len(cases)), flush=True)
    if args.prepare_only:
        return 0
    diagnostic = root / 'build/bin/device_api_module_diag'
    if not diagnostic.is_file():
        raise ValueError('Build module-diag first')
    rows = []
    cache = {}

    def record(label, selected, status, code, detail=''):
        if not detail and status == 'BUILD_FAIL':
            build_log = output / (label + '.build.log')
            if build_log.is_file():
                errors = [line.strip() for line in build_log.read_text(
                    encoding='utf-8', errors='replace').splitlines()
                    if re.search(r'(?:fatal )?error:|undefined reference', line)]
                detail = ' | '.join(errors[:2]) or 'see build log'
        row = {'case': label, 'status': status, 'exit': code,
               'kernels': ','.join(selected), 'detail': detail}
        rows.append(row)
        with (output / 'results.tsv').open('w', newline='', encoding='utf-8') as table:
            writer = csv.DictWriter(table, fieldnames=list(row), delimiter='\t')
            writer.writeheader()
            writer.writerows(rows)
        print('%s: %s %s' % (label, status, detail), flush=True)

    def load(label, image, selected):
        nm_log = output / (label + '.nm.log')
        code = invoke(['nm', '--defined-only', str(image)], nm_log, args.load_timeout)
        if code:
            return 'SYMBOL_SCAN_FAIL', code
        symbols, missing = kernel_symbols(nm_log.read_text(
            encoding='utf-8', errors='replace'), selected)
        if missing:
            (output / (label + '.missing-symbols.log')).write_text(
                '\n'.join(sorted(missing)) + '\n', encoding='utf-8')
            return 'MISSING_KERNEL_SYMBOL', 1
        code = invoke([str(diagnostic), str(args.gpu), str(image)] + symbols,
                      output / (label + '.load.log'), args.load_timeout)
        return ('LOAD_LOOKUP_OK' if code == 0 else 'LOAD_LOOKUP_FAIL'), code

    # Independent baseline: unmodified existing formal image, not regenerated.
    image = root / 'build/obj/device_api.mubin'
    if image.is_file():
        status, code = load('original', image, [anchor])
        record('original', [anchor], status, code)
    else:
        record('original', [anchor], 'BASELINE_MISSING', 1)
    for label, selected in cases:
        key = ('variant', label) if args.focus_allreduce else tuple(n for n in names if n in selected)
        if key in cache:
            status, code, reused = cache[key]
            record(label, selected, status, code, 'reused=' + reused)
            continue
        command = [args.make, '-s', '-C', str(root), 'batch-one',
                   'USE_MUSA=1', 'COMPILE_KERNEL=1', 'DEVICE_HOME=' + args.device_home,
                   'MUSA_ARCH=' + args.arch, 'BATCH_SOURCE=' + str(output / (label + '.cu')),
                   'BATCH_PREFIX=' + str(output / label)]
        print('Building ' + label + ' (' + str(len(selected)) + ' kernels)', flush=True)
        code = invoke(command, output / (label + '.build.log'), args.build_timeout)
        if code:
            status = 'BUILD_TIMEOUT' if code == 124 else 'BUILD_FAIL'
        else:
            status, code = load(label, output / (label + '.mubin'), selected)
        cache[key] = (status, code, label)
        record(label, selected, status, code)
    counts = {}
    for row in rows:
        counts[row['status']] = counts.get(row['status'], 0) + 1
    (output / 'summary.json').write_text(json.dumps(counts, indent=2), encoding='utf-8')
    archive = output / 'diagnostics.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as zipped:
        for file in sorted(output.iterdir()):
            if file.is_file() and file.suffix in ('.log', '.json', '.tsv', '.cu'):
                zipped.write(file, file.name)
    print('SUMMARY=' + json.dumps(counts), flush=True)
    print('Results: ' + str(output / 'results.tsv'), flush=True)
    print('Download logs and sources: ' + str(archive), flush=True)
    print('No Kernel execution or functional acceptance claimed.', flush=True)
    return 0 if all(row['exit'] == 0 for row in rows) else 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError) as error:
        print('ERROR: ' + str(error), file=sys.stderr)
        sys.exit(2)
