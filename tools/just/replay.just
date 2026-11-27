# Compare parameterized schedule replay with the normal driver.
# Everything is behind the ISS feature `replay`; nothing here is a default.
#
#   just replay-plan-prep   make the inputs (ISS_BENCH_BUCKET, default P8_4_128): the object and weights of the all-padding plan (ISS_BENCH_DIR, default /tmp/iss-bench),
#                          then the site table, the text arena, the logical text and the site values of the 64-step decode sequence
#                          (REPLAY_PLAN_DIR, default /tmp/replay-plan). Needs the program cache entry of the P8_4_128 bucket
#                          (`python3 tools/iss_bench_prep.py` builds it when missing) and takes about 10 seconds.
#   just replay-plan        synthetic checks, exactness for live plans, and fault and refusal controls, followed by the optional timing comparison. The long runs all use the release build.

replay-plan-prep:
    #!/usr/bin/env bash
    set -euo pipefail
    source "${OPENTSP_TOOLCHAIN:-$HOME/.local/opt/opentsp-toolchain}/env.sh"
    bench="${ISS_BENCH_DIR:-/tmp/iss-bench}"
    plans="${REPLAY_PLAN_DIR:-/tmp/replay-plan}"
    bucket="${ISS_BENCH_BUCKET:-P8_4_128}"
    python3 tools/iss_bench_prep.py --out "$bench" "$bucket"
    python3 - "$plans" "${REPLAY_PLAN_STEPS:-64}" "$bucket" <<'PY'
    #!/usr/bin/env python3
    """Export the inputs for parameterized replay: the site table with the place of every site unit, the whole logical text of the paged
    queues, and the site values of a decode sequence (N steps) of the P8_4_128 bucket.  Writes into DIR:
      sites.txt   one line per site: name q unit nunits lo width vmin vmax stride span placeholder fixed loc0 [loc1]
                  loc = I:queue:unit (initial image) or W:slice:addr:unit_in_word (a word of the text arena)
      text.bin    per queue with sites: u32 queue, u32 image_len, u32 total, then `total` u32 units (image, then blocks, padding included)
      steps.bin   int32 LE: [nsteps][nsites] site values
      padded.bin  int32 LE [nsites]: the all-padding plan
    """
    import os, sys, struct
    import numpy as np

    OUT = sys.argv[1]
    NSTEPS = int(sys.argv[2])
    BUCKET = sys.argv[3]
    from opentsp_vllm.models.tinystories import build_programs as BP
    from opentsp_vllm.models.tinystories import program_cache as PC
    from opentsp_vllm import step_plan as SP  # noqa: E402

    os.makedirs(OUT, exist_ok=True)
    model = PC.Model(8)
    bucket = PC.parse_bucket(BUCKET)
    art = BP.Artifacts(PC.bucket_dir(None, model, bucket), model.profile)
    t = art.paged.table
    names = [str(n) for n in t.names]
    geometry = art.record["geometry"]
    vl = len(next(iter(art.text.values()))[1][0])
    upw = vl // 4
    bu = art.paged.block_units
    # block (queue, index) -> list of (slice, addr, words)
    blk = {(b["queue"], b["index"]): b["tensors"] for b in art.blocks}


    def unit_loc(q, block, u):
        if block is None:
            return f"I:{q}:{u}"
        w, k = divmod(u, upw)
        for (n, sl, addr, words) in blk[(q, block)]:
            if w < words:
                return f"W:{sl}:{addr + w}:{k}"
            w -= words
        raise SystemExit("unit outside the block")


    with open(os.path.join(OUT, "sites.txt"), "w") as f:
        for i, n in enumerate(names):
            locs = [unit_loc(int(t.queue[i]), b, u) for b, u in art.paged.where(i)]
            f.write(" ".join([n, str(int(t.queue[i])), str(int(t.unit[i])), str(int(t.nunits[i])), str(int(t.lo[i])), str(int(t.width[i])),
                              str(int(t.vmin[i])), str(int(t.vmax[i])), str(int(t.stride[i])), str(int(t.span[i])),
                              str(int(t.placeholder[i])), str(int(t.fixed[i]))] + locs) + "\n")

    # where the SRAM holds the blocks: queue logical_start slice addr words
    with open(os.path.join(OUT, "arena.txt"), "w") as f:
        for b in art.blocks:
            q, j = b["queue"], b["index"]
            il = art.paged.image_len[q]
            pos = il + j * bu
            for (n, sl, addr, words) in b["tensors"]:
                f.write(f"{q} {pos} {sl} {addr} {words}\n")
                pos += words * upw

    # the whole logical text of every queue that has sites: image then blocks (padding included)
    blocks = art.block_bytes()
    queues = sorted({int(x) for x in t.queue} | {int(q) for q in art.images} | {int(b['queue']) for b in art.blocks})
    with open(os.path.join(OUT, "text.bin"), "wb") as f:
        for q in queues:
            img = list(art.images[q])
            units = list(img)
            j = 0
            while (q, j) in blocks:
                units += np.frombuffer(bytes(blocks[(q, j)]), "<u4").tolist()
                j += 1
            il = art.paged.image_len.get(q, len(img))
            assert q not in art.paged.image_len or il == len(img), (q, il, len(img))
            f.write(struct.pack("<III", q, il, len(units)))
            f.write(np.array(units, dtype="<u4").tobytes())


    def plan_of(reqs):
        return SP.plan_step([SP.PlanRequest(*r) for r in reqs], vocab=50257)


    def token_at(row):
        # Hold the embedding Permute field fixed. Replay admits changes to MEM addresses only.
        return SP.PACK * (1 + row % ((50257 - 1) // SP.PACK))


    def sequence(nsteps):
        # the lanes of a decode sequence of the bucket (T token lanes, L logit rows, C context slots): L requests decode one token a step
        # (a logit row each, at positions that move one a step and differ by four between requests) and, when T > L, one prefill request
        # takes a chunk of T - L tokens a step (no logit row); the context reaches the bucket's last block
        T, L, C = bucket
        top = C - nsteps  # the last step's first request is at the last context slot
        out = []
        for s in range(nsteps):
            reqs = []
            for i in range(L):
                pos = top + s - 4 * i
                table = [8 * i + 1 + j for j in range(8)]
                ids = [token_at(7 * i + k) for k in range(pos)] + [token_at(1000 + 37 * i + s)]
                reqs.append((f"d{i}", ids, table, pos, 1))
            if T > L:
                n = T - L
                c = n * (s % 16)
                ids = [token_at(k * 13) for k in range(80)]
                reqs.append(("p", ids, [40 + j for j in range(8)], c, n))
            out.append(plan_of(reqs))
        return out


    plans = sequence(NSTEPS)
    for p in plans:
        assert p.bucket == bucket, (p.bucket, bucket)
    steps = np.zeros((NSTEPS, len(names)), dtype="<i4")
    for s, p in enumerate(plans):
        v = PC.plan_site_values(geometry, names, p)
        steps[s] = [v[n] for n in names]
    steps.tofile(os.path.join(OUT, "steps.bin"))
    pv = PC.plan_site_values(geometry, names, SP.padding_plan(bucket, vocab=50257))
    np.array([pv[n] for n in names], dtype="<i4").tofile(os.path.join(OUT, "padded.bin"))
    print("sites", len(names), "steps", NSTEPS, "queues with sites", len(queues), "block_units", bu, "words/unit", upw)
    changed = (steps != steps[0]).any(axis=0).sum()
    print("sites whose value changes over the sequence:", int(changed))
    PY

replay-plan: replay-plan-prep
    #!/usr/bin/env bash
    set -euo pipefail
    source "${OPENTSP_TOOLCHAIN:-$HOME/.local/opt/opentsp-toolchain}/env.sh"
    export ISS_BENCH_DIR="${ISS_BENCH_DIR:-/tmp/iss-bench}" REPLAY_PLAN_DIR="${REPLAY_PLAN_DIR:-/tmp/replay-plan}"
    cargo test --release -p tsp-iss --features replay --test replay_param
    cargo test --release -p tsp-iss --features replay --test replay_param -- --ignored --nocapture --test-threads=1 shape_and_origins faults_and_foreign every_step accepted64_sequence
    cargo test --release -p tsp-iss --features replay --test replay_param -- --ignored --nocapture --test-threads=1 bench
