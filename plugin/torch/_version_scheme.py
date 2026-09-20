"""Version derived from git tags, plus a suffix the packager supplies.

FlagCX owns the public part of its version because it is the only side that
knows its own tags. A packager that builds this tree for a specific environment
owns one thing FlagCX cannot know: which vendor SDK the build linked against.
`makefiles/ascend.mk` links the SDK's own libhccl.so, so the same commit built
against CANN 8.5.0 and CANN 9.0.0 produces two different binaries, and without
an input for it both would claim the same version and collide in an index.

The suffix is appended verbatim; FlagCX never interprets it.
"""

import os

from setuptools_scm.version import guess_next_dev_version

SUFFIX_ENV = "FLAGCX_VERSION_SUFFIX"


def version_scheme(version):
    """Compose `<public>[+<suffix>.<date>.<node>[.dirty]]`.

    The local part is built here rather than by a local_scheme: setuptools_scm
    deduplicates the segments of a local part, so a suffix with a repeated
    number (`cann9.0.0`) loses one of them on the way through.
    """
    main = guess_next_dev_version(version)
    parts = []
    suffix = os.environ.get(SUFFIX_ENV, "").strip().lstrip("+")
    if suffix:
        parts.append(suffix)
    if version.distance:
        node_date = version.node_date
        parts.append(
            node_date.strftime("%Y%m%d")
            if hasattr(node_date, "strftime")
            else str(node_date).replace("-", "")
        )
        parts.append(version.node)
    if version.dirty:
        parts.append("dirty")
    return main + ("+" + ".".join(parts) if parts else "")
