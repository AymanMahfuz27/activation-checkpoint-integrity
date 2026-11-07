"""Standard-library checks for a hashlocked, hardlinked installation.

These checks inspect versions and filesystem identity. They never modify cache
contents, package permissions or installed package files.
"""

import os
from pathlib import Path
import re
import stat
import time


def require(value, message):
    if not value:
        raise RuntimeError(message)


def normalized_name(name):
    require(
        isinstance(name, str)
        and bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name)),
        "Invalid distribution name",
    )
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_lock(path):
    """Read exact requirement pins without resolving or importing dependencies."""
    pins, counts, current = {}, {}, None
    for line in Path(path).read_text().splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        pin = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;]+)(?:\s+\\)?", text)
        if pin:
            name = normalized_name(pin[1])
            require(name not in pins, "Duplicate normalized lock package")
            pins[name], counts[name], current = pin[2], 0, name
            continue
        hashed = re.fullmatch(r"--hash=sha256:[0-9a-f]{64}(?:\s+\\)?", text)
        require(
            current is not None and hashed is not None,
            "Unsupported requirement lock syntax",
        )
        counts[current] += 1
    require(
        bool(pins) and all(counts[name] > 0 for name in pins),
        "Every package must have allowed SHA256 hashes",
    )
    return pins


def verify_versions(expected, distributions):
    actual = {}
    for item in distributions:
        name = normalized_name(item["name"])
        require(name not in actual, "Duplicate installed normalized package")
        require(isinstance(item["version"], str), "Installed version is not text")
        actual[name] = item["version"]
    require(
        actual == expected,
        "Installed package census or exact versions differ from lock",
    )
    return actual


def regular_stat(path):
    path = Path(path)
    require(not path.is_symlink(), "Representative package/cache member is a symlink")
    value = path.stat()
    require(
        stat.S_ISREG(value.st_mode),
        "Representative package/cache member is not regular",
    )
    return value


def stat_record(path, value):
    return {
        "path": str(path),
        "st_dev": value.st_dev,
        "st_ino": value.st_ino,
        "st_nlink": value.st_nlink,
        "bytes": value.st_size,
    }


def hardlink_proof(
    target,
    cache,
    deadline_epoch,
    *,
    minimum_bytes=64_000_000,
    max_scanned_files=500_000,
):
    """Find the actual cached inode rather than assuming a versioned cache layout."""
    target, cache = Path(target), Path(cache)
    native = regular_stat(target)
    require(
        native.st_size >= minimum_bytes and native.st_nlink >= 2,
        "Representative large Torch library is not demonstrably hardlinked",
    )
    require(
        target.stat().st_dev == cache.stat().st_dev,
        "Package and cache are on different filesystems",
    )
    scanned, matches = 0, []
    for directory, subdirectories, files in os.walk(cache, followlinks=False):
        require(
            time.time() < deadline_epoch, "Hardlink proof reached processing cutoff"
        )
        subdirectories[:] = [
            name for name in subdirectories if not (Path(directory) / name).is_symlink()
        ]
        for filename in files:
            scanned += 1
            require(
                scanned <= max_scanned_files,
                "Bounded cache metadata scan exceeded file census",
            )
            if filename != target.name:
                continue
            candidate = Path(directory) / filename
            if candidate.is_symlink():
                continue
            value = regular_stat(candidate)
            if (value.st_dev, value.st_ino) == (native.st_dev, native.st_ino):
                require(
                    value.st_size == native.st_size and value.st_nlink >= 2,
                    "Same cached inode has contradictory size or link count",
                )
                matches.append(stat_record(candidate, value))
    require(matches, "No cached regular file shares the installed Torch library inode")
    return {
        "status": "ACTUAL_CACHE_AND_INSTALLED_INODE_IDENTICAL",
        "installed": stat_record(target, native),
        "cache_matches": matches,
        "cache_root": str(cache),
        "metadata_files_scanned": scanned,
        "cache_layout_assumption": None,
        "shared_files_modified": False,
    }
