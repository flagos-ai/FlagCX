# Changelog Management

This document describes how to manage changelogs for FlagCX packages.

## Overview

`packaging/sync-changelog.py` generates Debian and RPM changelogs from two data sources:

1. **`docs/CHANGELOG.md`** (primary) - Human-written release summaries
2. **Git tags** (fallback) - Auto-generated from commit history between tags

When a version exists in `docs/CHANGELOG.md`, its curated entries are used. For versions that only have a git tag (e.g. upstream hasn't updated the CHANGELOG yet), the script extracts commit summaries between tags.

## Usage

```bash
python3 packaging/sync-changelog.py
```

Output:
- `packaging/debian/changelog` (Debian format)
- `packaging/rpm/specs/flagcx.spec` (%changelog section)

## Data Source Priority

| Source | When used | Entry style |
|--------|-----------|-------------|
| `docs/CHANGELOG.md` | Version has entry in file | Curated, 3-5 bullet points |
| Git tags + commits | Version has tag but no CHANGELOG entry | Per-commit summaries (filtered) |
| Fallback | Tag exists but no commits found | "New upstream release vX.Y.Z" |

The git tag fallback filters out:
- Merge commits
- Dependabot version bumps (`Bump ...`)
- `[CRL]`, `[PAL]`, etc. prefixes are stripped for readability
- PR numbers (`(#123)`) are removed

## Version Normalization

Versions are normalized to three-part format: `0.7` becomes `0.7.0`. This prevents duplicates when `docs/CHANGELOG.md` uses `v0.7` but git tags use `v0.7.0`.

## Build Integration

The sync script runs automatically during package builds:

- **Local builds**: Both `build-flagcx.sh` and `build-flagcx-rpm.sh` call it
- **CI/CD**: Runs as part of the Docker build process

## Best Practices

1. Prefer editing `docs/CHANGELOG.md` for polished release notes
2. Git tag fallback is automatic - no action needed for new tags
3. Run `sync-changelog.py` before committing packaging changes to verify output
4. Never hand-edit generated files (`debian/changelog`, spec `%changelog`)
