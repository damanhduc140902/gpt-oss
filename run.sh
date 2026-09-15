#!/usr/bin/env bash
# Build, run and evaluate gpt-oss on AMD GPUs.
# A thin wrapper over the Makefile; every target it calls also works with plain `make`.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  B=$'\033[1m'; DIM=$'\033[2m'; R=$'\033[0m'
  RED=$'\033[31m'; GRN=$'\033[32m'; YLW=$'\033[33m'; CYN=$'\033[36m'
else
  B=''; DIM=''; R=''; RED=''; GRN=''; YLW=''; CYN=''
fi

die() { printf '%s\n' "${RED}error:${R} $*" >&2; exit 1; }
step() { printf '%s\n' "${CYN}==>${R} ${B}$*${R}"; }

usage() {
  cat <<EOF
${B}gpt-oss on AMD GPUs${R} ${DIM}- pure C++ and HIP inference${R}

${B}USAGE${R}
  ${GRN}./run.sh${R} ${YLW}build${R} [default|fast|omp|debug]   compile the inference binary
  ${GRN}./run.sh${R} ${YLW}run${R}   <checkpoint> [options]     run the model
  ${GRN}./run.sh${R} ${YLW}eval${R}  [20b|120b]                 score outputs with METEOR and BERTScore
  ${GRN}./run.sh${R} ${YLW}clean${R}                            remove build artifacts

${B}BUILD MODES${R}
  ${YLW}fast${R}      -O3, the default and the one the benchmarks use
  ${YLW}omp${R}       -O3 with OpenMP and -march=native
  ${YLW}debug${R}     -O0 with symbols, for gdb and valgrind
  ${YLW}default${R}   -O0, no flags, most portable

${B}RUN OPTIONS${R} ${DIM}(passed straight through to ./run)${R}
  ${YLW}-m${R} <mode>    generate | chat | getp            ${DIM}default: generate${R}
  ${YLW}-i${R} <string>  prompt, or input file in getp mode
  ${YLW}-o${R} <file>    output file, getp mode only
  ${YLW}-n${R} <int>     steps to run, 0 means full context ${DIM}default: 1024${R}
  ${YLW}-t${R} <float>   temperature                        ${DIM}default: 0.0${R}
  ${YLW}-p${R} <float>   top-p (nucleus) sampling           ${DIM}default: 0.9${R}
  ${YLW}-s${R} <int>     random seed                        ${DIM}default: time(NULL)${R}
  ${YLW}-y${R} <string>  system prompt, chat mode only
  ${YLW}-z${R} <file>    custom tokenizer                   ${DIM}default: tokenizer.bin${R}

${B}EXAMPLES${R}
  ${DIM}# build, then chat${R}
  ./run.sh build
  ./run.sh run model.bin -m chat -y "You are a concise assistant."

  ${DIM}# one prompt${R}
  ./run.sh run model.bin -m generate -i "Write a haiku about parallelism." -t 0.8

  ${DIM}# batch serving, the mode the throughput numbers come from${R}
  ./run.sh run model.bin -m getp -i tests/data/input.txt -o tests/data/output.txt

  ${DIM}# restrict to two GPUs (all visible ones are used by default, up to 8)${R}
  HIP_VISIBLE_DEVICES=0,1 ./run.sh run model.bin -m getp -i tests/data/input.txt -o out.txt

${B}DOCS${R}
  docs/MODEL.md  docs/KERNELS.md  docs/PARALLELISM.md  tests/README.md
EOF
}

cmd_build() {
  local mode="${1:-fast}" target
  case "$mode" in
    fast)    target=runfast  ;;
    omp)     target=runomp   ;;
    debug)   target=rundebug ;;
    default) target=run      ;;
    *) die "unknown build mode '$mode' (expected: default, fast, omp, debug)" ;;
  esac
  command -v hipcc >/dev/null 2>&1 || printf '%s\n' "${YLW}warning:${R} hipcc not found, falling back to g++ (CPU only)"
  step "building ($mode)"
  make "$target"
  printf '%s\n' "${GRN}done${R} -> ./run"
}

cmd_run() {
  [ $# -ge 1 ] || die "run needs a checkpoint: ./run.sh run model.bin [options]"
  [ -x ./run ] || die "./run not built yet - run './run.sh build' first"
  [ -f "$1" ] || die "checkpoint '$1' not found"
  [ -f tokenizer.bin ] || die "tokenizer.bin missing - run './run.sh build' to regenerate it"
  exec ./run "$@"
}

cmd_eval() {
  local model="${1:-20b}"
  case "$model" in 20b|120b) ;; *) die "unknown model '$model' (expected: 20b or 120b)" ;; esac
  step "evaluating $model"
  cd tests && exec python3 eval.py -m "$model"
}

case "${1:-help}" in
  build)               shift; cmd_build "$@" ;;
  run)                 shift; cmd_run   "$@" ;;
  eval)                shift; cmd_eval  "$@" ;;
  clean)               step "cleaning"; make clean ;;
  -h|--help|help)      usage ;;
  *)                   printf '%s\n\n' "${RED}error:${R} unknown command '${1}'" >&2; usage >&2; exit 1 ;;
esac
