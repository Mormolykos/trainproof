"""Entry point for `python -m trainproof`.

The console script is the documented way in, but `python -m` is what people
reach for first when a package is already installed -- and it used to fail with
"'trainproof' is a package and cannot be directly executed", which reads like a
broken install rather than a missing shim.

Both paths call the same main(), so exit codes stay identical either way. That
matters: exit codes are a documented contract (CONTRACTS.md), and a second entry
point that scored runs differently would quietly break it.
"""

from .cli import main

if __name__ == "__main__":
    main()
