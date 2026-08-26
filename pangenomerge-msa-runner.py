#!/usr/bin/env python

"""Convenience wrapper for running pangenomerge-msa directly from source tree."""

import sys

from pangenomerge.alignment_functions.post_pgmerge_alignment import main

if __name__ == '__main__':
    # main() reports failure by returning 1; without sys.exit() that code is
    # discarded and a failed run looks successful to the caller.
    sys.exit(main())
