#!/usr/bin/env python3
"""
Convert project changelog to Debian changelog and RPM %changelog format.

Data sources (priority order):
  1. docs/CHANGELOG.md - parsed entries with full descriptions
  2. git tags - date + commit summaries between tags
  3. Fallback - "New upstream release vX.Y.Z"
"""

import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def normalize_version(version):
    """Normalize version string: '0.7' -> '0.7.0', '0.10.0' -> '0.10.0'."""
    parts = version.split('.')
    while len(parts) < 3:
        parts.append('0')
    return '.'.join(parts)


def run_git(*args):
    """Run a git command and return stdout, or None on failure."""
    try:
        result = subprocess.run(
            ['git'] + list(args),
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def parse_changelog_md(changelog_path):
    """Parse docs/CHANGELOG.md and extract version entries."""
    if not changelog_path.exists():
        return {}

    with open(changelog_path, 'r') as f:
        content = f.read()

    version_pattern = r'- \*\*\[(\d{4})/(\d{2})\]\*\* Released \[v([^\]]+)\]'
    versions = {}

    lines = content.split('\n')
    i = 0

    while i < len(lines):
        line = lines[i]
        match = re.search(version_pattern, line)

        if match:
            year, month, version = match.group(1), match.group(2), match.group(3)
            date = f"{year}-{month}-01"
            entries = []
            i += 1

            while i < len(lines) and lines[i].strip() == '':
                i += 1

            while i < len(lines):
                line = lines[i]
                if re.search(version_pattern, line):
                    break

                if line.strip().startswith('- '):
                    entry = line.strip()[2:].strip()
                    i += 1
                    while i < len(lines):
                        next_line = lines[i]
                        if next_line.startswith('    ') and not next_line.strip().startswith('-'):
                            entry += ' ' + next_line.strip()
                            i += 1
                        else:
                            break
                    # Remove markdown formatting
                    entry = re.sub(r'\*([^*]+)\*', r'\1', entry)
                    entry = re.sub(r'_([^_]+)_', r'\1', entry)
                    entry = re.sub(r'`([^`]+)`', r'\1', entry)
                    entries.append(entry)
                elif line.strip() == '':
                    i += 1
                    peek = i
                    while peek < len(lines) and lines[peek].strip() == '':
                        peek += 1
                    if peek < len(lines) and re.search(version_pattern, lines[peek]):
                        break
                else:
                    i += 1

            norm_ver = normalize_version(version)
            versions[norm_ver] = {
                'version': norm_ver,
                'date': date,
                'entries': entries,
                'source': 'changelog.md'
            }
        else:
            i += 1

    return versions


def get_git_tags():
    """Get all version tags sorted by version (newest first)."""
    output = run_git('tag', '-l', 'v*', '--sort=-v:refname',
                     '--format=%(refname:short) %(creatordate:short)')
    if not output:
        return []

    tags = []
    for line in output.strip().split('\n'):
        parts = line.split(' ', 1)
        if len(parts) == 2:
            tag, date = parts
            version = tag.lstrip('v')
            tags.append({'tag': tag, 'version': version, 'date': date})
    return tags


def get_commits_between_tags(tag_old, tag_new):
    """Get commit summaries between two tags."""
    range_spec = f"{tag_old}..{tag_new}" if tag_old else tag_new
    output = run_git('log', range_spec, '--oneline', '--no-merges',
                     '--format=%s')
    if not output:
        return []

    commits = []
    for line in output.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        # Skip Dependabot bumps and trivial commits
        if line.startswith('Bump ') or line.startswith('Merge '):
            continue
        # Clean up [TAG] prefixes for readability
        clean = re.sub(r'^\[(CRL|PAL|UIL|CICD|CI|Others)\]\s*', '', line)
        # Remove PR number suffix
        clean = re.sub(r'\s*\(#\d+\)$', '', clean)
        if clean:
            commits.append(clean)
    return commits


def build_version_list(changelog_md_path):
    """Build complete version list from CHANGELOG.md + git tags."""
    # Source 1: CHANGELOG.md
    md_versions = parse_changelog_md(changelog_md_path)
    print(f"  CHANGELOG.md: {len(md_versions)} version(s)")

    # Source 2: git tags
    tags = get_git_tags()
    print(f"  git tags: {len(tags)} tag(s)")

    # Merge: use CHANGELOG.md entries when available, git tag fallback otherwise
    all_versions = []
    seen = set()

    for i, tag_info in enumerate(tags):
        version = normalize_version(tag_info['version'])
        if version in seen:
            continue
        seen.add(version)

        if version in md_versions:
            all_versions.append(md_versions[version])
        else:
            # Fallback: generate from git
            prev_tag = tags[i + 1]['tag'] if i + 1 < len(tags) else None
            commits = get_commits_between_tags(prev_tag, tag_info['tag'])

            entries = commits if commits else [f"New upstream release v{version}"]

            all_versions.append({
                'version': version,
                'date': tag_info['date'],
                'entries': entries,
                'source': 'git-tag'
            })

    # Also include CHANGELOG.md versions that have no git tag
    for version, data in md_versions.items():
        if version not in seen:
            all_versions.append(data)
            seen.add(version)

    # Sort by version descending
    def version_key(v):
        parts = v['version'].split('.')
        return tuple(int(p) for p in parts if p.isdigit())

    all_versions.sort(key=version_key, reverse=True)
    return all_versions


def generate_debian_changelog(versions, output_path):
    """Generate Debian changelog format."""
    lines = []

    for v in versions:
        date_obj = datetime.strptime(v['date'], '%Y-%m-%d')
        deb_date = date_obj.strftime('%a, %d %b %Y 10:00:00 +0800')

        lines.append(f"flagcx ({v['version']}-1) unstable; urgency=medium")
        lines.append("")

        for entry in v['entries']:
            lines.append(f"  * {entry}")

        lines.append("")
        lines.append(f" -- FlagOS Contributors <contact@flagos.io>  {deb_date}")
        lines.append("")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"  Generated Debian changelog: {output_path}")


def generate_rpm_changelog(versions):
    """Generate RPM %changelog format."""
    lines = []

    for v in versions:
        date_obj = datetime.strptime(v['date'], '%Y-%m-%d')
        rpm_date = date_obj.strftime('%a %b %d %Y')

        lines.append(f"* {rpm_date} FlagOS Contributors <contact@flagos.io> - {v['version']}-1")

        for entry in v['entries']:
            lines.append(f"- {entry}")

        lines.append("")

    return '\n'.join(lines)


def update_rpm_spec(spec_path, changelog_content):
    """Update %changelog section in RPM spec file."""
    if not spec_path.exists():
        print(f"  Warning: RPM spec not found: {spec_path}")
        return

    with open(spec_path, 'r') as f:
        spec_content = f.read()

    changelog_pattern = r'%changelog.*$'
    new_spec = re.sub(changelog_pattern, f'%changelog\n{changelog_content}',
                      spec_content, flags=re.DOTALL)

    with open(spec_path, 'w') as f:
        f.write(new_spec)

    print(f"  Updated RPM spec changelog: {spec_path}")


def main():
    project_root = Path(__file__).parent.parent
    changelog_md = project_root / 'docs' / 'CHANGELOG.md'
    debian_changelog = project_root / 'packaging' / 'debian' / 'changelog'
    rpm_spec = project_root / 'packaging' / 'rpm' / 'specs' / 'flagcx.spec'

    print("Collecting version data...")
    versions = build_version_list(changelog_md)

    if not versions:
        print("Error: No versions found from any source")
        sys.exit(1)

    print(f"\nTotal: {len(versions)} version(s)")
    for v in versions:
        src = v.get('source', '?')
        print(f"  v{v['version']} ({v['date']}) [{src}] - {len(v['entries'])} entries")

    print("\nGenerating changelogs...")
    generate_debian_changelog(versions, debian_changelog)

    rpm_changelog = generate_rpm_changelog(versions)
    update_rpm_spec(rpm_spec, rpm_changelog)

    print("\nDone!")


if __name__ == '__main__':
    main()
