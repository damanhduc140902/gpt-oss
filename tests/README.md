# Evaluation

Evaluate generated completions against references using **METEOR** and **BERTScore (F1)**.

## Prerequisites

- Python 3.10+
- `tiktoken` and `tqdm` come from [`requirements.txt`](../requirements.txt); the two scoring
  libraries do not, so install them separately:

  ```bash
  pip install nltk bert-score
  ```

- **GPU strongly recommended**. On AMD GPUs, ensure PyTorch is installed **with ROCm** — check with
  `python -c "import torch; print(torch.version.hip)"`, which must not print `None`. On CPU, BERTScore
  runs at roughly 1.7 sentences per second, so 4096 samples take about 40 minutes.
- `MODELS_ROOT` (optional) says where the BERTScore model is cached. It defaults to `tests/models/`,
  and the model is downloaded to `$MODELS_ROOT/bs_model/roberta-large-mnli` on first use.

> NLTK resources (`punkt`, `wordnet`, `omw-1.4`) are downloaded automatically on first run. If your machine is offline, pre-download them:
>
> ```bash
> python -c "import nltk; [nltk.download(x) for x in ['punkt','wordnet','omw-1.4']]"
> ```

## Quick start

From the repository root:

```bash
./run.sh eval 20b     # or: ./run.sh eval 120b
```

The rest of this page runs `eval.py` directly, from inside `tests/`.

### SLURM

Evaluate the **20B** model outputs on 2 GPUs:

```bash
srun --gres=gpu:2 python eval.py -m 20b
```

Evaluate the **120B** model:

```bash
srun --gres=gpu:2 python eval.py -m 120b
```

## CLI options

`eval.py` supports:

- `-m, --model_type {20b,120b}`
  Selects default input paths for submission and references.
- `-s, --submission PATH`
  Override path to submission completions (space-separated token IDs per line). It takes precedence
  over the default `--model_type` selects, so you can score a file from anywhere:

  ```bash
  python eval.py -m 20b -s /path/to/your_output_4096.txt
  ```

- `-r, --references PATH`
  Override path to reference completions (space-separated token IDs per line).
- `-e, --encoding NAME` (default: `o200k_harmony`)
  Tiktoken encoding used to decode token IDs.

Run `python eval.py -h` for the full help text. See the docstring in `eval.py` for details on
defaults and behavior.

> `threshold.json` must be present in the `tests/` folder: `eval.py` opens it before it even parses
> arguments, so a missing file aborts the run with `FileNotFoundError`. It holds the target
> thresholds for METEOR and BERTScore F1, and the script **raises an `AssertionError`** if either
> metric is below its threshold.

## Sample output (truncated)

```bash
20b
parsing args...
loading tiktoken encoding: o200k_harmony
reading submission: submission/output_20b_token_ids.txt
decoded lines: 4096 from submission/output_20b_token_ids.txt
reading references: references/output_20b_token_ids.txt
decoded lines: 4096 from references/output_20b_token_ids.txt

computing METEOR...
ensuring nltk resources...
aligning by index, items: 4096
computing METEOR in parallel with 96 workers...
100%|██████████| 4096/4096 [00:46<00:00, 88.0it/s]
meteor  0.532643

computing BERTScore (roberta-large-mnli)...
BERTScore device: GPU (2 device(s))
BERTScore: 100%|██████████| 4096/4096 [02:22<00:00, 28.7it/s]
bertscore_f1    0.977838

==========
done.
items           4096
meteor          0.532643
bertscore_f1    0.977838
```

Those two numbers come from the completions committed in `submission/`: `gpt-oss-20b` in `getp` mode
on 8x AMD MI250, 12288 requests, scored over the first 4096. They clear both thresholds in
`threshold.json` comfortably. They are not the scores in the [main README](../README.md) table, which
come from the current engine -- see the note under the 120B block.

```bash
120b
==========
done.
items           4096
meteor          0.567394
bertscore_f1    0.98164
```

Both models' completions in `submission/` come from runs on 8x AMD MI250, scored over the first 4096
of 12288 (20B) and 6144 (120B) requests -- on this engine's first complete version, which ran at
33,979 and 13,149 tok/s, not the current one. The current one scores METEOR 0.516 / BERTScore 0.977
on 20B and 0.552 / 0.980 on 120B at its largest batches, and 0.520 / 0.978 and 0.547 / 0.980 on
`input.txt` as it is. Reproducing the table's last two columns therefore means re-running `getp` on
this commit and re-scoring, not scoring the files committed here.

**Expected runtime:** \~4 minutes for 4096 samples on 2 GPUs (your hardware and load may vary).

## Scoring your own run

`getp` takes the rows per GPU from the request count, `num_reqs / n_devices`, up to 1984 for 20B and
1024 for 120B (see [Run a batch](../README.md#5-run-a-batch) in the main README). The reference files
here hold 4096 completions, one per prompt of `input.txt`, and `input.txt` is itself a valid input: on
eight GPUs it runs at 512 rows per GPU, and its output lines up one-for-one with the references:

```bash
../run ../gpt-oss-20b.bin -m getp -i input.txt -o /tmp/out.txt -z ../tokenizer.bin
python eval.py -m 20b -s /tmp/out.txt
```

512 rows is a quality check, not the batch the throughput figures run at. To score the largest batch,
repeat `input.txt` up to the largest count and score the first 4096 outputs; on eight GPUs that is
15872 requests for 20B:

```bash
python3 ../tools/make_getp_input.py -m 20b -g 8 -s input.txt -o /tmp/input_15872.txt
../run ../gpt-oss-20b.bin -m getp -i /tmp/input_15872.txt -o /tmp/out.txt -z ../tokenizer.bin
head -4096 /tmp/out.txt > /tmp/submission.txt
python eval.py -m 20b -s /tmp/submission.txt
```

Completions are written in request order, so output line _i_ is the answer to prompt _i_.

The engine is reproducible: two runs of the same binary on the same input file produce identical
output, and that is how changes are validated here -- the 20B against output hash 81e82a2c077f and
the 120B against 5ab5041b86f2 on the standard inputs (22ce0d3418b7 and 8254b50f71f9 on the largest),
both with zero differing lines. (The hashes have changed by design along the way, each time a change
that moves the numbers was accepted on its METEOR and BERTScore instead; the main README marks those
rows.) Hashing the output, not METEOR or BERTScore, is therefore the gate for any change that is
supposed to move no numbers; the scores are the coarse backstop, too noisy to accept or reject a
sub-percent change.

It was not always so. Before the expert exchange became a pull, a receiver could read a peer's buffer
while it was still being written, and two 8-GPU runs of the same binary agreed on only 50 % to
95 % of output lines, run pair by run pair. The 20B kept a rarer exception for longer: about one
full-length run in twelve differed in a few hundred lines of one GPU or one pair, more often under
host load. A copy engine wrote the partner's expert output to HBM behind the L2, and the add kernel
read stale lines; the pairs now exchange through kernels that read the peer's memory directly, and
six runs under the same load are identical (see the main README).

What the engine is still not is position-invariant: a completion depends on where its prompt sits in
the batch. The same prompt at a different batch index takes a different numerical path through the
expert grouping, so repeated copies of one prompt do not produce identical completions even at
temperature 0.

## Tips & troubleshooting

- **GPU vs CPU:** If logs show CPU, performance will be much slower. Verify:

  ```bash
  python -c "import torch; print(torch.version.hip, torch.cuda.device_count())"
  ```

  On ROCm, `torch.version.hip` must not be `None`.
