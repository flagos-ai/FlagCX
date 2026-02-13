#!/usr/bin/env python3
"""
Convert docs/CHANGELOG.md to Debian changelog and RPM %changelog format
"""

import re
import sys
from datetime import datetime
from pathlib import Path

def parse_changelog(changelog_path):
    """Parse docs/CHANGELOG.md and extract version entries"""
    with open(changelog_path, 'r') as f:
        content = f.read()

    # Pattern: - **[YYYY/MM]** Released [vX.Y.Z](link):
    version_pattern = r'- \*\*\[(\d{4})/(\d{2})\]\*\* Released \[v([^\]]+)\]'
    versions = []

    lines = content.split('\n')
    i = 0

    while i < len(lines):
        line = lines[i]
        match = re.search(version_pattern, line)

        if match:
            year = match.group(1)
            month = match.group(2)
            version = match.group(3)

            # Construct date (use first day of month)
            date = f"{year}-{month}-01"

            # Collect change entries (lines starting with "  - ")
            entries = []
            i += 1

            # Skip empty lines after version header
            while i < len(lines) and lines[i].strip() == '':
                i += 1

            # Collect bullet points
            while i < len(lines):
                line = lines[i]

                # Stop if we hit another version entry
                if re.search(version_pattern, line):
                    break

                # Collect bullet points (lines starting with "  - ")
                if line.strip().startswith('- '):
                    # Clean up the entry
                    entry = line.strip()[2:].strip()

                    # Continue collecting if next lines are continuations (indented but not bullet points)
                    i += 1
                    while i < len(lines):
                        next_line = lines[i]
                        # If it's a continuation (starts with spaces but not a bullet)
                        if next_line.startswith('    ') and not next_line.strip().startswith('-'):
                            entry += ' ' + next_line.strip()
                            i += 1
                        else:
                            break

                    # Remove markdown formatting
                    entry = re.sub(r'\*([^*]+)\*', r'\1', entry)  # Remove *italic*
                    entry = re.sub(r'_([^_]+)_', r'\1', entry)    # Remove _italic_
                    entry = re.sub(r'`([^`]+)`', r'\1', entry)    # Remove `code`
                    entries.append(entry)
                elif line.strip() == '':
                    # Empty line might indicate end of this version's entries
                    i += 1
                    # Check if next non-empty line is a new version
                    peek = i
                    while peek < len(lines) and lines[peek].strip() == '':
                        peek += 1
                    if peek < len(lines) and re.search(version_pattern, lines[peek]):
                        break
                else:
                    i += 1

            versions.append({
                'version': version,
                'date': date,
                'entries': entries
            })
        else:
            i += 1

    return versions

def generate_debian_changelog(versions, output_path):
    """Generate Debian changelog format"""
    lines = []

    for v in versions:
        # Parse date
        date_obj = datetime.strptime(v['date'], '%Y-%m-%d')
        # Debian date format: Mon, 10 Feb 2025 10:00:00 +0800
        deb_date = date_obj.strftime('%a, %d %b %Y 10:00:00 +0800')

        # Header
        lines.append(f"flagcx ({v['version']}-1) unstable; urgency=medium")
        lines.append("")

        # Entries
        for entry in v['entries']:
            lines.append(f"  * {entry}")

        lines.append("")
        # Footer
        lines.append(f" -- FlagOS Contributors <contact@flagos.io>  {deb_date}")
        lines.append("")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"✓ Generated Debian changelog: {output_path}")

def generate_rpm_changelog(versions):
    """Generate RPM %changelog format"""
    lines = []

    for v in versions:
        # Parse date
        date_obj = datetime.strptime(v['date'], '%Y-%m-%d')
        # RPM date format: Mon Feb 10 2025
        rpm_date = date_obj.strftime('%a %b %d %Y')

        # Header
        lines.append(f"* {rpm_date} FlagOS Contributors <contact@flagos.io> - {v['version']}-1")

        # Entries
        for entry in v['entries']:
            lines.append(f"- {entry}")

        lines.append("")

    return '\n'.join(lines)

def update_rpm_spec(spec_path, changelog_content):
    """Update %changelog section in RPM spec file"""
    with open(spec_path, 'r') as f:
        spec_content = f.read()

    # Replace %changelog section
    changelog_pattern = r'%changelog.*$'
    new_spec = re.sub(changelog_pattern, f'%changelog\n{changelog_content}',
                      spec_content, flags=re.DOTALL)

    with open(spec_path, 'w') as f:
        f.write(new_spec)

    print(f"✓ Updated RPM spec changelog: {spec_path}")

def main():
    # Paths
    project_root = Path(__file__).parent.parent
    changelog_md = project_root / 'docs' / 'CHANGELOG.md'
    debian_changelog = project_root / 'packaging' / 'debian' / 'changelog'
    rpm_spec = project_root / 'packaging' / 'rpm' / 'specs' / 'flagcx.spec'

    if not changelog_md.exists():
        print(f"Error: {changelog_md} not found")
        sys.exit(1)

    # Parse CHANGELOG.md
    print(f"Parsing {changelog_md}...")
    versions = parse_changelog(changelog_md)
    print(f"Found {len(versions)} version(s)")

    if not versions:
        print("Error: No versions found in changelog")
        sys.exit(1)

    # Generate Debian changelog
    generate_debian_changelog(versions, debian_changelog)

    # Generate and update RPM changelog
    rpm_changelog = generate_rpm_changelog(versions)
    update_rpm_spec(rpm_spec, rpm_changelog)

    print("\n✓ Changelog sync complete!")
    print(f"\nTo update changelogs:")
    print(f"  1. Edit {changelog_md}")
    print(f"  2. Run: {Path(__file__).name}")

if __name__ == '__main__':
    main()
