"""Generate an input file with the exact number of requests `getp` mode accepts.

Batch mode does not accept an arbitrary number of prompts. `distribute_requests`
asserts, in src/getp/run.cpp:

    assert(n_parallel_models * requests_per_model == requests->num_reqs);
    assert(requests_per_model == EXPERT_PARALLELISM * BATCH_SIZE);

With EXPERT_PARALLELISM = 2 and BATCH_SIZE = 1536 for the 20B model, and
EXPERT_PARALLELISM = n_devices with BATCH_SIZE = 768 for the 120B model, both
reduce to the same rule:

    num_reqs == n_devices * BATCH_SIZE

so on eight GPUs that is 8 * 1536 = 12288 requests for 20B and 8 * 768 = 6144
for 120B -- the two counts in the results table. None of the input files shipped
in this repository satisfies it; they declare 32, 256, 448, 3584 and 4096
requests, so every one of them aborts at the assert.

This script takes a pool of prompts and repeats it until the file holds exactly
the required number.

Examples:
    # 12288 prompts for the 20B model on 8 GPUs, from the shipped pool
    python3 tools/make_getp_input.py -m 20b -g 8 -o input_20b.txt

    # 6144 prompts for the 120B model on 8 GPUs
    python3 tools/make_getp_input.py -m 120b -g 8 -o input_120b.txt

    # your own prompts, one per line
    python3 tools/make_getp_input.py -m 20b -g 2 -s my_prompts.txt -o input.txt
"""

import argparse
import sys

# src/getp/run.cpp sets these from Config.n_experts at startup.
BATCH_SIZE = {"20b": 1536, "120b": 768}

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

  batch = BATCH_SIZE[args.model]
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
  print("  batch size %d  (EXPERT_PARALLELISM x BATCH_SIZE per model)" % batch)
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
