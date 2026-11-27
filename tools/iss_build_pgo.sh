#!/usr/bin/env bash
# Build a release ISS trained on a real command line, e.g. tools/iss_build_pgo.sh run --profile ref --weights W.safetensors P.elf
#   tools/iss_build_pgo.sh --corpus [DIR]   train on the coverage programs of benches/programs (`bench --program`: b1, b3, the dense
#                                           f16 and i8 matrix programs, the valid VXM, transcendental and barrier programs) and, when
#                                           DIR (default /tmp/iss-bench, from tools/iss_bench_prep.py) holds them, the P1_1_64 and
#                                           P8_4_128 buckets (`just iss-pgo`)
#   tools/iss_build_pgo.sh --binding [DIR]  build the in-process binding that serving runs (the manifest `opentsp-vllm/native/Cargo.toml`,
#                                           its own workspace) the same way: trained by `opentsp-iss/perf/binding_p8.py`
#                                           on the P8_4_128 bucket in DIR (default /tmp/iss-bench, from tools/iss_bench_prep.py), written to
#                                           target/iss-pgo/_tsp_iss.abi3.so (`just iss-pgo-binding`)
#   tools/iss_build_pgo.sh --path           print the ISS binary a serving recipe should run: target/iss-pgo/tsp-iss when it is newer than
#                                           every ISS and toolchain source file, else target/release/tsp-iss (a PGO binary of older
#                                           sources is stale; one built with the host's CPU features only runs on a host that has them)
set -euo pipefail
if [[ $# == 0 ]]; then
    echo "usage: tools/iss_build_pgo.sh <tsp-iss subcommand and training arguments> | --corpus [DIR] | --binding [DIR] | --path" >&2
    exit 2
fi
root=$(cd "$(dirname "$0")/.." && pwd)
if [[ $1 == --path ]]; then
    pgo="$root/target/iss-pgo/tsp-iss"
    if [[ -x $pgo && -z $(find "$root/opentsp-iss/crates" "$root/opentsp-sw-toolchain/crates" -path '*/src/*' -name '*.rs' -newer "$pgo" -print -quit) ]]; then
        echo "$pgo"
    else
        echo "$root/target/release/tsp-iss"
    fi
    exit 0
fi
corpus=
binding=
if [[ $1 == --corpus ]]; then
    corpus=${2:-/tmp/iss-bench}
elif [[ $1 == --binding ]]; then
    binding=${2:-/tmp/iss-bench}
    if [[ ! -f $binding/P8_4_128.elf || ! -f $binding/P8_4_128.safetensors ]]; then
        echo "iss_build_pgo: $binding/P8_4_128.elf not found: write the training input with tools/iss_bench_prep.py --out $binding P8_4_128" >&2
        exit 2
    fi
fi
# The training runs: the arguments given, or the corpus.
train() {
    local exe=$1
    shift
    if [[ -n $binding ]]; then
        local py=${OPENTSP_VLLM_VENV:-$HOME/.local/opt/opentsp-vllm-venv}/bin/python
        [[ -x $py ]] || py=$(command -v python3)
        "$py" "$root/opentsp-iss/perf/binding_p8.py" --lib "$exe" --dir "$binding" --runs 3 > /dev/null
        return
    fi
    if [[ -z $corpus ]]; then
        "$exe" "$@"
        return
    fi
    local p b
    for p in b1 b3 dense_f16 dense_i8 vxm_valid transc syncnotify; do
        "$exe" bench --program "$root/opentsp-iss/crates/tsp-iss/benches/programs/$p.txt" --repeat 1 > /dev/null
    done
    for b in P1_1_64 P8_4_128; do
        if [[ -f $corpus/$b.elf && -f $corpus/$b.safetensors ]]; then
            "$exe" run --profile ref --weights "$corpus/$b.safetensors" "$corpus/$b.elf" > /dev/null
        else
            echo "iss_build_pgo: $corpus/$b.elf not found: the $b bucket is not in the training corpus" >&2
        fi
    done
}
host=$(rustc -vV | sed -n 's/^host: //p')
profdata="$(rustc --print sysroot)/lib/rustlib/$host/bin/llvm-profdata"
if [[ ! -x "$profdata" ]]; then
    echo "llvm-profdata is required for this rustc. Install with: rustup component add llvm-tools-preview" >&2
    exit 1
fi
# Train in a fresh directory. Each invocation gets its own binaries and counters,
# so interrupted, concurrent or older runs cannot supply a profile to this build.
out="$root/target/iss-pgo"
mkdir -p "$out"
work=$(mktemp -d "$out/build.XXXXXXXX")
trap 'rm -rf "$work"' EXIT
mkdir "$work/raw"
# Cargo's encoded flags preserve whitespace in paths and caller-provided flags.
flags=${CARGO_ENCODED_RUSTFLAGS:-}
if [[ -z ${CARGO_ENCODED_RUSTFLAGS+x} && -n ${RUSTFLAGS:-} ]]; then
    read -r -a inherited <<< "$RUSTFLAGS"
    printf -v flags '%s\037' "${inherited[@]}"
    flags=${flags%$'\037'}
fi
build() {
    if [[ -n $binding ]]; then
        # the binding has its own workspace and builds without a Python interpreter (PyO3 abi3, extension-module mode)
        local features=()
        [[ ${OPENTSP_BINDING_REPLAY:-1} == 0 ]] || features=(--features replay)
        PYO3_NO_PYTHON=1 PYO3_BUILD_EXTENSION_MODULE=1 CARGO_ENCODED_RUSTFLAGS="${flags:+$flags$'\037'}$1" \
            cargo build --manifest-path "$root/opentsp-vllm/native/Cargo.toml" --locked --release \
            --target "$host" --target-dir "$work/target" ${features[@]+"${features[@]}"}
        return
    fi
    CARGO_ENCODED_RUSTFLAGS="${flags:+$flags$'\037'}$1" cargo build --manifest-path "$root/Cargo.toml" \
        --locked --release --package tsp-iss --target "$host" --target-dir "$work/target"
}
exe="$work/target/$host/release/tsp-iss"
if [[ -n $binding ]]; then
    case "$(uname -s)" in Darwin) exe="$work/target/$host/release/lib_tsp_iss.dylib" ;; *) exe="$work/target/$host/release/lib_tsp_iss.so" ;; esac
fi
build "-Cprofile-generate=$work/raw"
LLVM_PROFILE_FILE="$work/raw/%m-%p.profraw" train "$exe" "$@"
"$profdata" merge -o "$work/training.profdata" "$work"/raw/*.profraw
build "-Cprofile-use=$work/training.profdata"
# Only a fully built executable replaces the previous result, on the same filesystem.
if [[ -n $binding ]]; then
    if [[ $(uname -s) == Darwin ]]; then install_name_tool -id @loader_path/_tsp_iss.abi3.so "$exe"; fi
    mv -f "$exe" "$out/_tsp_iss.abi3.so"
    echo "Profile-guided binding: $out/_tsp_iss.abi3.so"
    exit 0
fi
mv -f "$exe" "$out/tsp-iss"
echo "Profile-guided ISS: $out/tsp-iss"
