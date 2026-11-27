"""The ship-program cache drops completed entries beyond the three newest, and keeps the one a run has open."""
import os
import tempfile
import unittest

from opentsp_vllm.models.gpt2 import cache as ship_cache


class CacheEviction(unittest.TestCase):
    def _entries(self, root):
        # Persistent digest locks are protocol state, not cached program entries.
        self.assertTrue(all(os.path.isdir(os.path.join(root, n)) or n.endswith(".lock") for n in os.listdir(root)))
        return {n for n in os.listdir(root) if os.path.isdir(os.path.join(root, n))}

    def _done(self, root, name, when):
        d = os.path.join(root, name)
        os.makedirs(d)
        with open(os.path.join(d, "done"), "w") as f:
            f.write("ok")
        os.utime(d, (when, when))
        return d

    def test_three_completed_entries_stay_and_the_one_in_use_stays_even_when_it_is_the_oldest(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = [self._done(tmp, f"{i:020x}", 1000 + i) for i in range(5)]
            ship_cache.evict(tmp, keep=3, retain=old[0])
            self.assertEqual(self._entries(tmp),
                             {os.path.basename(old[0]), os.path.basename(old[3]), os.path.basename(old[4])})

    def test_retain_has_priority_over_future_timestamps_and_still_uses_one_slot(self):
        import time
        for keep in (1, 3):
            with self.subTest(keep=keep), tempfile.TemporaryDirectory() as tmp:
                active = self._done(tmp, "active", 1000)
                for i in range(3):
                    self._done(tmp, f"future{i}", time.time() + 86400 + i)
                removed = ship_cache.evict(tmp, keep=keep, retain=active)
                self.assertTrue(os.path.isdir(active), removed)
                self.assertNotIn("active", removed)
                self.assertEqual(sum(os.path.isdir(os.path.join(tmp, n)) for n in os.listdir(tmp)), keep)

    def test_an_entry_with_an_active_digest_lock_is_not_evicted(self):
        import fcntl
        with tempfile.TemporaryDirectory() as tmp:
            active = self._done(tmp, "active", 1000)
            others = [self._done(tmp, f"newer{i}", 1001 + i) for i in range(3)]
            with open(active + ".lock", "a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                removed = ship_cache.evict(tmp, keep=3, retain=others[-1])
                self.assertTrue(os.path.isdir(active), removed)
            ship_cache.evict(tmp, keep=3, retain=others[-1])
            self.assertFalse(os.path.isdir(active), "a later pass can evict an idle old entry")

    def test_a_retained_entry_already_removed_by_another_pass_is_harmless(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._done(tmp, "live", 1000)
            self.assertEqual(ship_cache.evict(tmp, keep=1, retain=os.path.join(tmp, "already-gone")), [])
            self.assertEqual(self._entries(tmp), {"live"})

    def test_unified_manifest_marks_a_completed_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(4):
                d = self._done(tmp, f"{i:020x}", 1000 + i)
                os.rename(os.path.join(d, "done"), os.path.join(d, "manifest.json"))
            ship_cache.evict(tmp, keep=3)
            self.assertEqual(self._entries(tmp), {f"{i:020x}" for i in range(1, 4)})

    def test_a_cache_smaller_than_the_limit_is_left_whole(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._done(tmp, f"{1:020x}", 10)
            self._done(tmp, f"{2:020x}", 20)
            ship_cache.evict(tmp, keep=3)
            self.assertEqual(len(self._entries(tmp)), 2)

    def test_an_incomplete_directory_stays_unless_asked(self):
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(4):
                self._done(tmp, f"{i:020x}", 1000 + i)
            tmpdir = os.path.join(tmp, "abcd.tmp7")
            os.makedirs(tmpdir)
            ship_cache.evict(tmp, keep=3)
            self.assertTrue(os.path.isdir(tmpdir))
            ship_cache.evict(tmp, keep=3, incomplete=True)
            self.assertFalse(os.path.isdir(tmpdir))
            self.assertEqual(len(self._entries(tmp)), 3)


if __name__ == "__main__":
    unittest.main()
