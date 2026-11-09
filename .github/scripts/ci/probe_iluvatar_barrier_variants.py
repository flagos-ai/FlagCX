#!/usr/bin/env python3
"""Generate the #696 baseline and the variant that restores constructors #4/#5."""

import argparse
import pathlib


INTRA = "// ---- Barrier<DefaultBackend<P>, flagcxTeamTagIntra, Coop> ----"
INTER = "// ---- Barrier<DefaultBackend<P>, flagcxTeamTagInter, Coop> ----"
WORLD = "// ---- Barrier<DefaultBackend<P>, flagcxTeamTagWorld, Coop> ----"
ACTIVE = "  // Active ctor\n"
INTRA_ARRIVE = "  // arrive: thread-striped store epoch+1"
CONTEXT = "  FLAGCX_DEVICE_INLINE_DECORATOR DefaultNetContext context() const {"


def section(source, start, end):
    if source.count(start) != 1 or source.count(end) != 1:
        raise ValueError(f"Expected one source anchor: {start!r} / {end!r}")
    return source[source.index(start) : source.index(end, source.index(start))]


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Expected one constructor block: {old[:90]!r}")
    return source.replace(old, new, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=pathlib.Path, required=True)
    parser.add_argument("--new", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    old = args.old.read_text()
    new = args.new.read_text()

    restored = replace_once(
        new,
        section(section(new, INTRA, INTER), ACTIVE, INTRA_ARRIVE),
        section(section(old, INTRA, INTER), ACTIVE, INTRA_ARRIVE),
    )
    restored = replace_once(
        restored,
        section(section(new, INTER, WORLD), ACTIVE, CONTEXT),
        section(section(old, INTER, WORLD), ACTIVE, CONTEXT),
    )
    if restored == new:
        raise ValueError("Restoring #4/#5 did not change the header")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "after_696.h").write_text(new)
    (args.output_dir / "undo_04_05.h").write_text(restored)
    (args.output_dir / "manifest.tsv").write_text(
        "case\trestored\tenabled\n"
        "after_696\t\t1,2,3,4,5,6,7,8,9\n"
        "undo_04_05\t4,5\t1,2,3,6,7,8,9\n"
    )


if __name__ == "__main__":
    main()
