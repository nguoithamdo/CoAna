# Copyright (c) 2026 Nguyen Truong Co
# SPDX-License-Identifier: LGPL-2.1-or-later
"""``python -m oligomer_analysis`` -> the ``coana`` command-line interface."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
