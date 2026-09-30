"""Generate an input file with a number of requests `getp` mode accepts.

Batch mode does not accept an arbitrary number of prompts. `distribute_requests`
asserts, in src/getp/run.cpp:

    assert(n_parallel_models * requests_per_model == requests->num_reqs);
    assert(requests_per_model == EXPERT_PARALLELISM * BATCH_SIZE);

With EXPERT_PARALLELISM = 2 for the 20B model and EXPERT_PARALLELISM = n_devices
for the 120B model, both reduce to the same rule:

    num_reqs == n_devices * BATCH_SIZE

BATCH_SIZE, the rows each GPU serves, is taken from the input file: warm_up
reads the request count and sets BATCH_SIZE = num_reqs / n_devices, up to the
most rows that fit in a GPU's memory - 1984 for the 20B and 1024 for the 120B
(GETP_BATCH_CAP_20B / GETP_BATCH_CAP_120B in src/getp/run.cpp). More rows per
step means more tokens per step for nearly the same weight traffic, so this
script writes the largest file by default: on eight GPUs 8 * 1984 = 15872
requests for 20B and 8 * 1024 = 8192 for 120B - the two counts in the results
table. The input files shipped in this repository declare 32, 256, 448, 3584
and 4096 requests; on eight GPUs they run as batches of 4 to 512 rows, far from
full speed.

This script takes a pool of prompts and repeats it until the file holds exactly
the required number.

Examples:
    # 15872 prompts for the 20B model on 8 GPUs, from the shipped pool
    python3 tools/make_getp_input.py -m 20b -g 8 -o input_20b.txt

    # 8192 prompts for the 120B model on 8 GPUs
    python3 tools/make_getp_input.py -m 120b -g 8 -o input_120b.txt

    # the earlier fixed batches (1536 / 768 rows per GPU): 12288 and 6144 prompts
    python3 tools/make_getp_input.py -m 20b -g 8 -r 1536 -o input_20b_1536.txt

    # your own prompts, one per line
    python3 tools/make_getp_input.py -m 20b -g 2 -s my_prompts.txt -o input.txt
"""

import argparse
import sys

# The most rows per GPU src/getp/run.cpp accepts (GETP_BATCH_CAP_*): what fits in memory.
BATCH_SIZE = {"20b": 1984, "120b": 1024}

# tokenizer->max_token_length; a request longer than max_token_len * (steps + 1)
# bytes would overrun its slot in read_inputfile's memcpy.
MAX_TOKEN_LENGTH = 128

DEFAULT_SOURCE = "tests/data/input.txt"


def read_prompts(path):
  """Read a prompt pool, accepting either getp format or one prompt per line."""
  with open(path, "r", encoding="utf-8") as f:
    lines = [line.rstrip("\n").rstrip("\r") for line in f]

  # getp files start with a bare request count; a plain list does not.
  if lines and lines[0].strip().isdigit():
    lines = lines[1:]

  prompts = [p for p in lines if p.strip()]
  if not prompts:
    raise SystemExit("no prompts found in %s" % path)
  return prompts


def main():
  ap = argparse.ArgumentParser(
      description="Generate a getp input file of the exact required length.")
  ap.add_argument("-m",
                  "--model",
                  choices=sorted(BATCH_SIZE),
                  default="20b",
                  help="which model the file is for (default: 20b)")
  ap.add_argument("-g",
                  "--gpus",
                  type=int,
                  default=8,
                  help="number of GPUs the run will see (default: 8)")
  ap.add_argument("-s",
                  "--source",
                  default=DEFAULT_SOURCE,
                  help="prompt pool to repeat (default: %s)" % DEFAULT_SOURCE)
  ap.add_argument("-r",
                  "--rows",
                  type=int,
                  default=0,
                  help="rows per GPU (default: the most that fit, 1984 for 20b "
                  "and 1024 for 120b)")
  ap.add_argument("-o", "--output", required=True, help="file to write")
  ap.add_argument("-n",
                  "--steps",
                  type=int,
                  default=1024,
                  help="steps the run will use, for the length check "
                  "(default: 1024)")
  args = ap.parse_args()

  if args.gpus < 1:
    raise SystemExit("--gpus must be at least 1")

  batch = args.rows if args.rows > 0 else BATCH_SIZE[args.model]
  if batch > BATCH_SIZE[args.model]:
    raise SystemExit("--rows %d is more than the %d rows per GPU that fit for %s" %
                     (batch, BATCH_SIZE[args.model], args.model))
  required = args.gpus * batch

  prompts = read_prompts(args.source)

  # read_inputfile memcpy's each line into a slot of this size.
  limit = MAX_TOKEN_LENGTH * (args.steps + 1)
  too_long = [i for i, p in enumerate(prompts) if len(p.encode("utf-8")) >= limit]
  if too_long:
    raise SystemExit("prompt %d is %d bytes, over the %d byte slot; shorten it "
                     "or raise --steps" %
                     (too_long[0], len(prompts[too_long[0]]), limit))

  with open(args.output, "w", encoding="utf-8", newline="\n") as f:
    f.write("%d\n" % required)
    for i in range(required):
      f.write(prompts[i % len(prompts)] + "\n")

  print("wrote %s" % args.output)
  print("  model      %s" % args.model)
  print("  gpus       %d" % args.gpus)
  print("  rows/GPU   %d  (BATCH_SIZE; the engine reads it from the request count)" % batch)
  print("  requests   %d  = %d x %d" % (required, args.gpus, batch))
  print("  pool       %d prompt(s) from %s, repeated %.1fx" %
        (len(prompts), args.source, required / len(prompts)))

  if len(prompts) < required:
    print("\nnote: the pool is smaller than the file, so prompts repeat. That is "
          "fine for a\n      throughput measurement but makes the output "
          "redundant for quality scoring.",
          file=sys.stderr)


if __name__ == "__main__":
  main()
