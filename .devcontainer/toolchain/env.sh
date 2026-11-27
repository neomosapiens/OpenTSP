# Source from bash or zsh.
OTS_TC="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export PATH="$OTS_TC/zwrap:$OTS_TC/venv/bin:$OTS_TC/sail/bin:$HOME/.cargo/bin:$PATH"
export GMP_PREFIX=/usr GMP_LIBS=-lgmp Z_LIBS=-lz
export PYTHON="$OTS_TC/venv/bin/python"
