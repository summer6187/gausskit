from ase.io import gaussian as g

def read_gaussian_opt_scan(fd, index: int = -1):
    """
    Wrapper around ASE's `read_gaussian_out` that tags geometries
    whose section is followed by 'Optimization complete' with
    `atoms.calc.results['Optimized'] = True`.

    This implementation avoids duplicating ASE's internal parsing logic
    by temporarily patching `_compare_merge_configs` during parsing.

    Parameters
    ----------
    fd : file-like
        Open Gaussian output file handle (e.g., from `open("scan.log")`).
    index : int, optional
        Index of the configuration to return. Defaults to -1 (the last one).

    Returns
    -------
    ase.Atoms
        Parsed Atoms object with an attached `SinglePointCalculator`.
        If the corresponding optimization step was followed by the line
        'Optimization complete', then `atoms.calc.results['Optimized']`
        will be set to `True`.
    """

    # Track whether an "Optimization complete" line was seen
    state = {"pending_opt": False}

    # ---- 1. Wrap the file descriptor to intercept each line ----
    class _FDWrapper:
        """File-like wrapper that detects 'Optimization complete' lines."""
        def __init__(self, raw):
            self._raw = raw

        def _check(self, line: str) -> str:
            if line.lstrip().startswith("Optimization complete"):
                # Mark that the next parsed geometry should be tagged as optimized
                state["pending_opt"] = True
            return line

        def readline(self, *args, **kwargs):
            """Intercept `readline()` calls and inspect each line."""
            line = self._raw.readline(*args, **kwargs)
            if not line:
                return line
            return self._check(line)

        def __iter__(self):
            """Support iteration with `for line in fd` syntax."""
            return self

        def __next__(self):
            """Intercept iteration line by line."""
            line = next(self._raw)  # may raise StopIteration
            return self._check(line)

        def __getattr__(self, name):
            """Delegate all other attributes to the underlying file object."""
            return getattr(self._raw, name)

    wrapped_fd = _FDWrapper(fd)

    # ---- 2. Temporarily patch `_compare_merge_configs` ----
    # This hook is called each time a new geometry (Atoms) is finalized.
    orig_compare = g._compare_merge_configs

    def patched_compare(configs, atoms):
        # If an 'Optimization complete' was seen since the last geometry,
        # tag this Atoms object as optimized
        if state["pending_opt"] and getattr(atoms, "calc", None) is not None:
            atoms.calc.results["Optimized"] = True
            # Reset the flag so only the immediate geometry is tagged
            state["pending_opt"] = False
        return orig_compare(configs, atoms)

    # Temporarily replace the original function
    g._compare_merge_configs = patched_compare
    try:
        # Call the standard ASE parser (which internally calls `_compare_merge_configs`)
        atoms = g.read_gaussian_out(wrapped_fd, index=index)
    finally:
        # Always restore the original function to avoid side effects
        g._compare_merge_configs = orig_compare

    return atoms


def read_opt_scan_configs(fname):
    print(f"Load {fname}, ", end="")
    with open(fname) as f:
        _configs = read_gaussian_opt_scan(f, index=slice(None, None, None))

    configs = []
    for a in _configs:
        if a.calc.results.get('Optimized'):
            configs.append(a)

    print(f"Found {len(configs)} geometries.")
    return configs
