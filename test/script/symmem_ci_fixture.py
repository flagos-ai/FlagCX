"""Create named rank reports for launchers that only simulate MPI execution."""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]


def install_rank_report_fixture(root):
    ci = root / '.github/scripts/ci'
    ci.mkdir(parents=True, exist_ok=True)
    for name in ('check_required_gtests.py', 'run_gtest_rank_xml.sh'):
        shutil.copyfile(ROOT / '.github/scripts/ci' / name, ci / name)
    generator = ci / 'mock_rank_reports.py'
    generator.write_text('''import sys
from pathlib import Path
import xml.etree.ElementTree as ET
args = sys.argv[1:]
wrappers = [i for i, a in enumerate(args) if a.endswith('/run_gtest_rank_xml.sh')]
if wrappers:
    index = wrappers[0]
    output = Path(args[index + 1])
    output.mkdir(parents=True, exist_ok=True)
    selected = next(a.split('=', 1)[1] for a in args if a.startswith('--gtest_filter='))
    nranks = int(args[args.index('-np') + 1])
    for rank in range(nranks):
        report = ET.Element('testsuites')
        suites = {}
        for case in selected.split(':'):
            suite, name = case.rsplit('.', 1)
            if suite not in suites:
                suites[suite] = ET.SubElement(report, 'testsuite', name=suite)
            ET.SubElement(suites[suite], 'testcase', name=name, status='run', result='completed')
        (output / ('rank-%d.xml' % rank)).write_text(ET.tostring(report, encoding='unicode'))
''')
    return generator
