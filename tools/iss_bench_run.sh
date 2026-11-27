#!/usr/bin/env bash
# Time `tsp-iss run` on the benchmark inputs of `tools/iss_bench_prep.py`: best of N runs of each bucket, printing the
# simulated cycles, the ISS's own wall seconds (`--stats` `wall_seconds`: machine, image load and the cycles) and the
# rate. Usage: tools/iss_bench_run.sh [-n REPEATS] [-d DIR] [-b BIN] [-l MAX_LOAD] [-f] [BUCKET ...]
#
# The run reports load and refuses the throughput measurement when it exceeds the requested limit. The first line gives
# the date, the one-minute load average, the power mode (macOS `pmset`), the host and the sha256 of the binary; every result
# line ends with the load average read after its runs; and when the load average before the runs is above MAX_LOAD
# (default 4, or $ISS_BENCH_MAX_LOAD) the script exits 3 without a figure, unless `-f` (`--force`) is given, which prints
# the figures marked `FORCED (load X above Y)`. Compare binaries with the interleaved A/B of `tools/iss_ab_overhead.py`.
set -euo pipefail
repeats=3
dir=/tmp/iss-bench
bin=target/release/tsp-iss
max_load=${ISS_BENCH_MAX_LOAD:-4}
force=
while getopts "n:d:b:l:f" o; do
    case "$o" in
        n) repeats=$OPTARG ;;
        d) dir=$OPTARG ;;
        b) bin=$OPTARG ;;
        l) max_load=$OPTARG ;;
        f) force=1 ;;
        *) exit 2 ;;
    esac
done
shift $((OPTIND - 1))
buckets=("$@")
[ ${#buckets[@]} -gt 0 ] || buckets=(P1_1_64 P8_4_128)

load() { uptime | sed -E 's/.*load averages?: *([0-9.]+).*/\1/'; }
power() {
    if command -v pmset > /dev/null 2>&1; then
        local lpm src
        lpm=$(pmset -g 2> /dev/null | awk '/lowpowermode/ {print $2}')
        src=$(pmset -g batt 2> /dev/null | sed -n "1s/.*'\(.*\)'.*/\1/p")
        echo "power ${src:-unknown}, low-power-mode ${lpm:-0}"
    else
        echo "power $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2> /dev/null || echo unknown)"
    fi
}
sha=$(shasum -a 256 "$bin" | cut -c1-16)
before=$(load)
echo "iss_bench_run: $(date '+%Y-%m-%d %H:%M:%S %Z'), load $before, $(power), $(uname -sm), binary $bin sha256 $sha"
mark=
if awk -v l="$before" -v m="$max_load" 'BEGIN { exit !(l > m) }'; then
    if [ -z "$force" ]; then
        echo "iss_bench_run: the load average $before is above $max_load: the figures would not be comparable; run on a quiet host, raise -l, or force with -f" >&2
        exit 3
    fi
    mark=" FORCED (load $before above $max_load)"
fi
stats=$(mktemp)
trap 'rm -f "$stats"' EXIT
for b in "${buckets[@]}"; do
    : > "$stats"
    best=""
    for _ in $(seq "$repeats"); do
        : > "$stats"
        "$bin" run --profile ref --weights "$dir/$b.safetensors" --stats="$stats" "$dir/$b.elf" >/dev/null
        best=$(python3 -c "
import json,sys
d=json.load(open('$stats'))
w=d['wall_seconds']
best='$best'
print(json.dumps(d if not best else min([d, json.loads(best)], key=lambda x: x['wall_seconds'])))")
    done
    python3 -c "
import json
d=json.loads('''$best''')
c, w, t = d['last_cycle'], d['wall_seconds'], d['ticks']
print(f\"$b: {c:,} cycles, {t:,} ticks, {w:.4f} s -> {c/w:,.0f} cycles/s, {t/w:,.0f} ticks/s, {1e6*w/t:.2f} us/tick, load $(load), sha $sha$mark\")"
done
