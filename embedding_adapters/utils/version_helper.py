from importlib.metadata import version as pkg_version


def _get_current_package_version_str(package_name: str = "embedding-adapters") -> str:
    """Return the current installed version string of the embedding-adapters package."""
    return pkg_version(package_name)


def _version_to_tuple(ver: str, max_len: int = 3) -> tuple[int, ...]:
    """
    Convert a version string like '0.3.1' to a tuple of ints (0, 3, 1).

    - Ignores any non-numeric suffix like '0.3.1rc1' -> '0.3.1'.
    - Pads / truncates to max_len components.
    """
    ver = str(ver).strip()
    # Strip off any pre-release / build metadata crud after first non [0-9.]
    clean = []
    for ch in ver:
        if ch.isdigit() or ch == ".":
            clean.append(ch)
        else:
            break
    clean = "".join(clean) or "0"

    parts = clean.split(".")
    nums = []
    for p in parts:
        if p == "":
            nums.append(0)
        else:
            try:
                nums.append(int(p))
            except ValueError:
                nums.append(0)

    # Pad or truncate
    if len(nums) < max_len:
        nums += [0] * (max_len - len(nums))
    else:
        nums = nums[:max_len]

    return tuple(nums)


def _compare_versions(a: str, b: str) -> int:
    """
    Compare two version strings a and b.

    Returns:
      -1 if a < b
       0 if a == b
       1 if a > b
    """
    ta = _version_to_tuple(a)
    tb = _version_to_tuple(b)
    if ta < tb:
        return -1
    if ta > tb:
        return 1
    return 0


def _version_satisfies_spec(spec_str: str, current_version: str) -> bool:
    """
    Check whether current_version satisfies a simple spec string.

    Supported forms:
      - "" or missing -> no constraint (always True)
      - "0.0.1"       -> exact match required (== 0.0.1)
      - ">=0.2.0"     -> single comparator
      - ">=0.2.0,<0.4.0" -> comma-separated AND of comparators

    Supported operators: ==, !=, >=, >, <=, <
    """
    spec_str = str(spec_str or "").strip()
    if not spec_str:
        # No constraint
        return True

    # If no operator present at all, treat it as exact-version
    if not any(op in spec_str for op in ("<", ">", "=", "!")):
        spec_str = f"=={spec_str}"

    current = str(current_version).strip()
    # Split on commas to get multiple constraints
    parts = [s.strip() for s in spec_str.split(",") if s.strip()]

    for part in parts:
        # Extract operator and version
        op = None
        version_part = None
        for candidate_op in ("<=", ">=", "==", "!=", "<", ">"):
            if part.startswith(candidate_op):
                op = candidate_op
                version_part = part[len(candidate_op):].strip()
                break

        if op is None or not version_part:
            # Bad spec: fail safe (or you could log and skip)
            return False

        cmp_result = _compare_versions(current, version_part)

        if op == "==":
            if cmp_result != 0:
                return False
        elif op == "!=":
            if cmp_result == 0:
                return False
        elif op == ">":
            if cmp_result <= 0:
                return False
        elif op == ">=":
            if cmp_result < 0:
                return False
        elif op == "<":
            if cmp_result >= 0:
                return False
        elif op == "<=":
            if cmp_result > 0:
                return False
        else:
            # unknown op
            return False

    # All constraints satisfied
    return True


def _entry_supports_version(entry_obj: dict, current_version: str) -> bool:
    """
    Return True if this registry entry should be considered for the
    given package version, based on its 'version' field.

    Rules:
      - Missing/empty 'version' -> no constraint (allowed).
      - Otherwise, 'version' is treated as a spec string understood by
        _version_satisfies_spec (single version or ranges).
    """
    spec_str = str(entry_obj.get("version", "")).strip()
    return _version_satisfies_spec(spec_str, current_version)