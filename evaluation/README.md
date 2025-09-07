- To generate references

```bash
# python generate.py /path/to/gpt-oss/safetensors -p /path/to/prompt/file -l max_new_tokens
python generate.py /dev/shm/gpt-oss-20b/ -p ../data/input.txt -l 2048
```

- To evaluate

```bash
python eval.py -p ../data/input.txt -s ../data/output.txt -r references/refs_fp32.txt
```
