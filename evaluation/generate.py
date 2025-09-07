import argparse
import pathlib

import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import TextStreamer


def parseArgs():
    parser = argparse.ArgumentParser(description="Text generation example")
    parser.add_argument(
        "checkpoint",
        metavar="FILE",
        type=pathlib.Path,
        help="Path to the SafeTensors checkpoint",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        metavar="PROMPT",
        type=pathlib.Path,
        default="../data/input.txt",
        help="Path to the LLM prompt file",
    )
    parser.add_argument(
        "-l",
        "--limit",
        metavar="LIMIT",
        type=int,
        default=2048,
        help="Limit on the number of tokens (0 to disable)",
    )
    parser.add_argument(
        "-dt",
        "--dtype",
        metavar="DTYPE",
        type=str,
        default="fp32",
        help="Model data type (fp32 or bf16)",
    )
    args = parser.parse_args()
    return args


def generate(prompt_path, model, tokenizer, limit):
    with open(prompt_path, "r") as f_prompts, open("refs.txt", "w") as f_refs:
        # Skip the first line, it is the prompt count
        next(f_prompts)
        for i, prompt in enumerate(f_prompts):
            enc = tokenizer(prompt,
                            return_tensors="pt",
                            add_special_tokens=False)
            input_ids = enc.input_ids
            attention_mask = enc.attention_mask if "attention_mask" in enc else torch.ones_like(
                input_ids)
            streamer = TextStreamer(tokenizer,
                                    skip_prompt=True,
                                    skip_special_tokens=True)
            print(f"\nPROCESSING PROMPT {i}\n")
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=limit,
                    do_sample=False,
                    num_beams=1,
                    use_cache=True,
                    streamer=streamer,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
            f_refs.write(" ".join(map(str,
                                      output_ids.squeeze(0).tolist())) + "\n")


if __name__ == "__main__":
    args = parseArgs()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    print(model.dtype)
    if args.dtype == "fp32":
        model.to(torch.float32)
    elif args.dtype == "bf16":
        model.to(torch.bfloat16)
    else:
        print(f"[ERROR] {args.dtype} is not supported, try fp32 or bf16")
    print(model.dtype)
    generate(args.prompt, model, tokenizer, args.limit)
