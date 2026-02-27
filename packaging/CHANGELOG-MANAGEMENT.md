# Changelog Management

This document describes how to manage changelogs for FlagCX packages.

## Overview

FlagCX uses `docs/CHANGELOG.md` as the single source of truth for release history. This changelog is automatically converted to both Debian and RPM formats during the build process.

## Files

- `docs/CHANGELOG.md` - Single source of truth for project release history
- `packaging/sync-changelog.py` - Converts docs/CHANGELOG.md to Debian and RPM formats

## Workflow

### Updating Changelogs

1. Edit `docs/CHANGELOG.md` following the existing format
2. Run the sync script to update package-specific changelogs:
   ```bash
   python3 packaging/sync-changelog.py
   ```
3. The script will update:
   - `packaging/debian/changelog` (Debian format)
   - `packaging/rpm/specs/flagcx.spec` (%changelog section)

## Build Integration

The changelog sync is automatically run during package builds:

- **CI/CD**: Both `.github/workflows/build-deb.yml` and `build-rpm.yml` run the sync script
- **Local builds**: Both `build-flagcx.sh` and `build-flagcx-rpm.sh` run the sync script

## Changelog Format

### docs/CHANGELOG.md (Source Format)

```markdown
## Release History

- **[2025/11]** Released [v0.7](https://github.com/flagos-ai/FlagCX/releases/tag/v0.7.0):

  - Added support to TsingMicro, including device adaptor `tsmicroAdaptor` and CCL adaptor `tcclAdaptor`.
  - Implemented an experimental kernel-free non-reduce collective communication
    (*SendRecv*, *AlltoAll*, *AlltoAllv*, *Broadcast*, *Gather*, *Scatter*, *AllGather*)
    using device-buffer IPC/RDMA.
```

### Debian Format (auto-generated)

```
flagcx (0.7-1) unstable; urgency=medium

  * Added support to TsingMicro, including device adaptor tsmicroAdaptor and CCL adaptor tcclAdaptor.
  * Implemented an experimental kernel-free non-reduce collective communication (SendRecv, AlltoAll, AlltoAllv, Broadcast, Gather, Scatter, AllGather) using device-buffer IPC/RDMA.

 -- FlagOS Contributors <contact@flagos.io>  Sat, 01 Nov 2025 10:00:00 +0800
```

### RPM Format (auto-generated)

```
* Sat Nov 01 2025 FlagOS Contributors <contact@flagos.io> - 0.7-1
- Added support to TsingMicro, including device adaptor tsmicroAdaptor and CCL adaptor tcclAdaptor.
- Implemented an experimental kernel-free non-reduce collective communication (SendRecv, AlltoAll, AlltoAllv, Broadcast, Gather, Scatter, AllGather) using device-buffer IPC/RDMA.
```

## Best Practices

1. Always edit `docs/CHANGELOG.md` directly, never edit the generated files
2. Run `sync-changelog.py` after updating `docs/CHANGELOG.md`
3. Follow the existing format for consistency
4. Use clear, descriptive changelog entries
5. Multi-line entries are supported - just indent continuation lines with 4 spaces
