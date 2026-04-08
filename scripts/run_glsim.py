#!/usr/bin/env python3
"""Official GLSim entrypoint.

The current round-two POPE/DASH-B pipeline does not use this script.
Use `scripts/run_glsim_adapted.py` only when you explicitly want the
queried-object pre-generation adaptation, not the official GLSim method.
"""

from __future__ import annotations


def main() -> int:
    raise SystemExit(
        "Official GLSim is a post-generation MSCOCO + CHAIR pipeline and is not wired into "
        "the current round-two POPE/DASH-B workflow. Use scripts/run_glsim_adapted.py only "
        "for the explicitly labeled adapted path."
    )


if __name__ == "__main__":
    raise SystemExit(main())
