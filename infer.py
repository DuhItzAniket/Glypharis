"""Run GLYPHARIS on images: ``python infer.py --image page.jpg`` (see `glypharis.cli`)."""

import sys

from glypharis.cli import main

if __name__ == "__main__":
    sys.exit(main())
