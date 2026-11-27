#!/usr/bin/env python3
"""Read one actual retired-instruction counter per scan, failing closed."""
import pathlib
import sys
from opentsp_tools.host_counters import count


def main():
    if len(sys.argv) < 3:
        sys.exit('usage: retired_instructions.py Darwin|Linux counter-file ...')
    try:
        print(sum(count(sys.argv[1], pathlib.Path(p).read_text()) for p in sys.argv[2:]))
    except (OSError, ValueError) as error:
        sys.exit('retired instructions: ' + str(error))


if __name__ == '__main__':
    main()
