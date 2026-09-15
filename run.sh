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

die()  { printf '%s\n' "${RED}error:${R} $*" >&2; exit 1; }
step() { printf '%s\n' "${CYN}==>${R} ${B}$*${R}"; }

usage() {
  cat <<EOF
${B}gpt-oss on AMD GPUs${R} ${DIM}- pure C++ and HIP inference, no external libraries${R}

${B}USAGE${R}
  ${GRN}./run.sh${R} ${YLW}build${R}   [fast|omp|debug|default]  compile the inference binary
  ${GRN}./run.sh${R} ${YLW}run${R}     <checkpoint> [options]    run the model
  ${GRN}./run.sh${R} ${YLW}mkinput${R} [20b|120b] [gpus] <out>   build a correctly sized getp input file
  ${GRN}./run.sh${R} ${YLW}decode${R}  [line] [-i <file>]        turn getp token ids back into text
  ${GRN}./run.sh${R} ${YLW}tok${R}     "<text>"                  tokenize a string with the C tokenizer
  ${GRN}./run.sh${R} ${YLW}eval${R}    [20b|120b]                score outputs with METEOR and BERTScore
  ${GRN}./run.sh${R} ${YLW}clean${R}                             remove build artifacts

${B}BUILD MODES${R}
  ${YLW}fast${R}      -O3, the default and the one the benchmarks use
  ${YLW}omp${R}       -O3 with OpenMP and -march=native
  ${YLW}debug${R}     -O0 with symbols, for gdb and valgrind
  ${YLW}default${R}   -O0, no flags, most portable

${B}RUN OPTIONS${R} ${DIM}(passed straight through to ./run)${R}
  ${YLW}-m${R} <mode>    generate | chat | getp             ${DIM}default: generate${R}
  ${YLW}-i${R} <string>  prompt, or input file in getp mode
  ${YLW}-o${R} <file>    output file, getp mode only
  ${YLW}-n${R} <int>     steps to run, 0 means full context ${DIM}default: 1024${R}
  ${YLW}-t${R} <float>   temperature, 0.0 is greedy         ${DIM}default: 0.0${R}
  ${YLW}-p${R} <float>   top-p (nucleus) sampling           ${DIM}default: 0.9${R}
  ${YLW}-s${R} <int>     random seed                        ${DIM}default: time(NULL)${R}
  ${YLW}-y${R} <string>  system prompt, chat mode only
  ${YLW}-z${R} <file>    custom tokenizer                   ${DIM}default: tokenizer.bin${R}

${B}EXAMPLES${R}
  ${DIM}# smallest possible check: does it answer at all${R}
  ./run.sh run gpt-oss-20b.bin -m generate -i "1+1="

  ${DIM}# a longer completion, sampled rather than greedy${R}
  ./run.sh run gpt-oss-20b.bin -m generate -i "Once upon a time" -n 256 -t 0.8 -p 0.95

  ${DIM}# interactive chat with a system prompt${R}
  ./run.sh run gpt-oss-20b.bin -m chat -y "You are a concise assistant."

  ${DIM}# batch serving - this is the mode the throughput numbers come from.${R}
  ${DIM}# getp accepts exactly n_devices x 1536 requests (20b) or n_devices x 768 (120b),${R}
  ${DIM}# so build the input file first - none of the shipped ones is the right length.${R}
  ./run.sh mkinput 20b 8 input_20b.txt  ${DIM}# 12288 requests${R}
  ./run.sh run gpt-oss-20b.bin -m getp -i input_20b.txt -o out.txt
  ./run.sh decode -i out.txt            ${DIM}# read those completions back as text${R}

  ${DIM}# restrict to two GPUs (every visible GPU is used by default, up to 8)${R}
  ./run.sh mkinput 20b 2 input_2gpu.txt ${DIM}# 3072 requests${R}
  HIP_VISIBLE_DEVICES=0,1 ./run.sh run gpt-oss-20b.bin -m getp -i input_2gpu.txt -o out.txt

  ${DIM}# tokenizer sanity check${R}
  ./run.sh tok "Hello world"           ${DIM}# -> 13225 2375${R}

${B}DOCS${R}
  docs/MODEL.md  docs/KERNELS.md  docs/PARALLELISM.md  docs/SERVING.md  tests/README.md
EOF
}

cmd_build() {
  local mode="${1:-fast}" target
  case "$mode" in
    fast)    target=runfast  ;;
    omp)     target=runomp   ;;
    debug)   target=rundebug ;;
    default) target=run      ;;
    *) die "unknown build mode '$mode' (expected: fast, omp, debug, default)" ;;
  esac
  command -v hipcc >/dev/null 2>&1 || printf '%s\n' "${YLW}warning:${R} hipcc not found, falling back to g++ (CPU only)"
  step "building ($mode)"
  make "$target"
  printf '%s\n' "${GRN}done${R} -> ./run"
}

cmd_run() {
  [ $# -ge 1 ] || die "run needs a checkpoint: ./run.sh run gpt-oss-20b.bin [options]"
  [ -x ./run ] || die "./run not built yet - run './run.sh build' first"
  [ -f "$1" ] || die "checkpoint '$1' not found"
  [ -f tokenizer.bin ] || die "tokenizer.bin missing - run './run.sh build' to regenerate it"
  exec ./run "$@"
}

cmd_mkinput() {
  local model="${1:-20b}" gpus="${2:-8}" out="${3:-}"
  [ -n "$out" ] || die "mkinput needs an output file: ./run.sh mkinput 20b 8 input_20b.txt"
  case "$model" in 20b|120b) ;; *) die "unknown model '$model' (expected: 20b or 120b)" ;; esac
  exec python3 tools/make_getp_input.py -m "$model" -g "$gpus" -o "$out"
}

cmd_decode() {
  [ -x ./decode ] || { step "building decode"; make decode; }
  local line="${1:--1}"
  [ $# -gt 0 ] && shift || true
  exec ./decode "$line" "$@"
}

cmd_tok() {
  [ $# -ge 1 ] || die "tok needs a string: ./run.sh tok \"Hello world\""
  [ -x ./test_tokenizer ] || { step "building test_tokenizer"; make tokenizer-test; }
  exec ./test_tokenizer -t tokenizer.bin -i "$1"
}

cmd_eval() {
  local model="${1:-20b}"
  case "$model" in 20b|120b) ;; *) die "unknown model '$model' (expected: 20b or 120b)" ;; esac
  step "evaluating $model"
  cd tests && exec python3 eval.py -m "$model"
}

case "${1:-help}" in
  build)          shift; cmd_build   "$@" ;;
  run)            shift; cmd_run     "$@" ;;
  mkinput)        shift; cmd_mkinput "$@" ;;
  decode)         shift; cmd_decode  "$@" ;;
  tok)            shift; cmd_tok    "$@" ;;
  eval)           shift; cmd_eval   "$@" ;;
  clean)          step "cleaning"; make clean ;;
  -h|--help|help) usage ;;
  *)              printf '%s\n\n' "${RED}error:${R} unknown command '${1}'" >&2; usage >&2; exit 1 ;;
esac
