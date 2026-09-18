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

Run `python eval.py -h` for the full help text. See the docstring in `eval.py` for details on defaults and behavior.

> `threshold.json` must be present in the `tests/` folder: `eval.py` opens it before it even parses arguments, so a missing file aborts the run with `FileNotFoundError`. It holds the target thresholds for METEOR and BERTScore F1, and the script **raises an `AssertionError`** if either metric is below its threshold.

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
`threshold.json` comfortably. They sit just under the METEOR 0.534 and BERTScore 0.978 in the
[main README](../README.md) table, and that gap is a different engine rather than run-to-run noise --
see the note under the 120B block.

```bash
120b
==========
done.
items           4096
meteor          0.567394
bertscore_f1    0.98164
```

Both models' completions in `submission/` come from runs on 8x AMD MI250, scored over the first 4096
of 12288 (20B) and 6144 (120B) requests -- on the pre-rewrite build that ran at 33,979 and 13,149
tok/s, not the current one, which scores METEOR 0.534 / BERTScore 0.978 on 20B and 0.566 / 0.981 on
120B. Reproducing the table's last two columns therefore means re-running `getp` on this commit and
re-scoring, not scoring the files committed here.

**Expected runtime:** \~4 minutes for 4096 samples on 2 GPUs (your hardware and load may vary).

## Scoring your own run

`getp` accepts exactly `n_devices x BATCH_SIZE` requests -- 1536 per GPU for 20B, 768 for 120B -- while
the reference files here hold 4096 completions. The two line up if you build the input by repeating
`input.txt` and then score only the first 4096 outputs. On eight GPUs that is 12288 = 3 x 4096:

```bash
python3 ../tools/make_getp_input.py -m 20b -g 8 -s input.txt -o /tmp/input_12288.txt
../run ../gpt-oss-20b.bin -m getp -i /tmp/input_12288.txt -o /tmp/out.txt -z ../tokenizer.bin
head -4096 /tmp/out.txt > /tmp/submission.txt
python eval.py -m 20b -s /tmp/submission.txt
```

Completions are written in request order, so output line _i_ is the answer to prompt _i_.

Note that the engine is not deterministic. The same prompt placed at a different batch index takes a
different numerical path through the expert grouping, so repeated copies of one prompt do not produce
identical completions even at temperature 0 -- and neither do two runs of the same binary on the same
input file: at 8 GPUs and 1024 steps they agree on only about 64% of tokens, while METEOR and BERTScore
stay put. (At 16 steps on 4 GPUs it is exactly deterministic.) Token agreement therefore cannot validate
a change at this scale; the quality scores are the gate.

## Tips & troubleshooting

- **GPU vs CPU:** If logs show CPU, performance will be much slower. Verify:

  ```bash
  python -c "import torch; print(torch.version.hip, torch.cuda.device_count())"
  ```

  On ROCm, `torch.version.hip` must not be `None`.
