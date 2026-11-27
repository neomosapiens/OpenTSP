"""The ship-program cache keeps its newest completed entries.

A completed entry is a directory that contains `manifest.json` or the legacy `done` marker and whose name has no `.tmp`.
The entry a run has just opened is passed as `retain`: it is touched, so it counts as
newest, and it is never removed by that eviction pass, even after clock rollback. A `.tmp` directory is left alone, because it may be a
build that is still running. Digest locks protect builders and artifact readers; busy
entries can temporarily leave more than `keep` directories. Pass `incomplete=True` only when nothing is building.
"""
import fcntl
import os
import shutil

KEEP = 3


def evict(cache, keep=KEEP, retain=None, incomplete=False):
    """Remove completed entries of `cache` beyond the `keep` newest. Returns the names removed."""
    if keep < 1:
        raise ValueError(f"keep must be at least 1, not {keep}")
    if not os.path.isdir(cache):
        return []
    retain_path = os.path.realpath(retain) if retain is not None else None
    if retain is not None:
        try:
            os.utime(retain, None)
        except FileNotFoundError:
            pass  # another pass may evict it after its reader has finished loading
    removed = []
    if incomplete:
        for name in list(os.listdir(cache)):
            path = os.path.join(cache, name)
            if os.path.isdir(path) and ".tmp" in name and os.path.realpath(path) != retain_path:
                shutil.rmtree(path)
                removed.append(name)
    done = []
    for name in os.listdir(cache):
        path = os.path.join(cache, name)
        if not os.path.isdir(path) or ".tmp" in name or not any(os.path.isfile(os.path.join(path, marker)) for marker in ("manifest.json", "done")):
            continue
        try:
            stat = os.stat(path)
        except FileNotFoundError:
            continue  # another eviction pass completed after the directory scan
        done.append((os.path.realpath(path) == retain_path, stat.st_mtime, name, path, (stat.st_dev, stat.st_ino)))
    done.sort()
    keep_names = {name for _, _, name, _, _ in done[-keep:]}
    for _, _, name, path, identity in done:
        if name in keep_names:
            continue
        # The same persistent digest lock protects ensure_cache and its consumer.
        # Never wait while evicting: a reader/builder has priority and no eviction
        # pass acquires a second lock while holding another one.
        with open(path + ".lock", "a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            try:
                stat = os.stat(path)
            except FileNotFoundError:
                continue
            if (stat.st_dev, stat.st_ino) != identity:
                continue  # another builder published a new directory after the scan
            shutil.rmtree(path)
            removed.append(name)
    return removed


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="keep the newest completed ship-program cache entries")
    ap.add_argument("--cache", default=os.path.expanduser("~/.cache/opentsp/ship"))
    ap.add_argument("--keep", type=int, default=KEEP)
    ap.add_argument("--incomplete", action="store_true", help="also remove .tmp directories; use only when no build is running")
    a = ap.parse_args(argv)
    gone = evict(a.cache, keep=a.keep, incomplete=a.incomplete)
    print(f"removed {len(gone)}: {', '.join(gone) if gone else '(none)'}")


if __name__ == "__main__":
    main()
