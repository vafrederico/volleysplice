# Temporary host-memory pressure monitor

`scripts/watch-generalization-host-memory.py` observes Windows physical-memory
availability and Linux page-cache statistics during the NAS-backed experiments.
It changes no training recipe, feature values, labels, service configuration or
persistent operating-system setting.

The monitor samples every 30 seconds as the regular WSL user. A cache-release
attempt requires all three strict conditions:

- Windows free physical memory below 2 GiB.
- Linux `Cached` above 4 GiB.
- Linux `MemAvailable` above 6 GiB.

Attempts must be at least 180 seconds apart; startup also has a conservative
180-second cooldown. Immediately before an attempt the helper refreshes all
pressure and stop checks. Its only mutating command invokes the current WSL
distribution through Windows `wsl.exe --user root`, running
`sync; echo 3 > /proc/sys/vm/drop_caches`. This releases reclaimable page cache,
dentries and inodes. It deletes no source/data/model files and restarts no service.
There is no automated elevation prompt.

The default `--once` mode only inspects. A long-running helper requires explicit
`--watch --permit-cache-flush`. It exits on its NAS `STOP` file, a process stop
signal, study `progress.json` status `complete`, or 12 hours of runtime. A failed
or timed-out flush stops the helper rather than retrying. Three consecutive
inspection errors also stop it. A NAS advisory lock permits only one watch process;
the lock file remains after exit and is not deleted.

All plan and event artifacts are under the study's `resource-watch-v1` directory.
The immutable plan binds the monitor source, five policy tests, existing resource
amendment and unchanged study protocol. Plan SHA-256:
`64bd3eaa8f99a613fb640cb18a1e45a5ac7e4ecc798e0e71214efb4b27d54815`.
Monitor source SHA-256:
`9516bc7bd442015843946b7487776b39189bd70af8fa2d26b997193394c05a09`.

Five pure policy tests passed. An inspection-only run on 2026-09-23 at 11:17 UTC
verified regular WSL UID 1000, Windows memory queries and a read-only nested
`id -u` returning root UID 0. It performed zero cache flushes. Following review,
root started the approved watch at 11:19 UTC, PID 91815; its first observation did
not trigger a flush. The owned helper must be stopped when the study finishes.

Windows C: free space is logged as context, not a trigger or a causal conclusion.
Windows paging, WSL swap and other host activity can all affect C:; these samples
do not uniquely attribute every change to the pagefile or NAS caching. Memory
return to Windows may lag cache release. Concurrent timing measurements remain
unsuitable as exclusive throughput or phone-performance benchmarks.
