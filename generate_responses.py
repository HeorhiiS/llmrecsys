import sys
import fire
import torch
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
from datasets import load_dataset

import pandas as pd
import numpy as np
import tqdm


def generate(
        model_type: str = "LLAMA-7B",
):

    # Load the model
    device_map = "auto" if torch.cuda.is_available() else "cpu"

    base_model = "../hfcheckpoints/7B/"
    lora_weights = "finetuned_models/7B/"
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
        )

    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map=device_map, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map=device_map,
        )

    def find_closest_string(original_string, string_list):
        max_common_chars = 0
        closest_string = None

        for string in string_list:
            common_chars = 0
            temp_string = string

            for char in original_string:
                if char in temp_string:
                    common_chars += 1
                    temp_string = temp_string.replace(char, '', 1)

            if common_chars > max_common_chars:
                max_common_chars = common_chars
                closest_string = string

        return closest_string

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    model.eval()
    eval_data = load_dataset("json", data_files="../litllamadata/finetune_dataset/llama_eval_red.json")

    precision_scores = []
    progress_bar = tqdm.tqdm(total=len(eval_data['train']), ncols=100, colour='green', ascii="░▒█")

    for prompt in eval_data['train']:
        instruction = prompt['instruction']
        input = prompt['input']
        og_output = prompt['output']

        prompt = f'### Instruction: {instruction}\n ### Input: ' + f"{input}\n ### Output:"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to('cuda')

        movie_df = pd.read_json('../movie_map.json')
        all_titles = movie_df['title']
        all_titles = np.array(all_titles)

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                repetition_penalty=2.0,
                max_new_tokens=128,
                temperature=1,
                top_p=1,
                top_k=50,
                num_beams=20,
                do_sample=True,
                eos_token_id=model.config.eos_token_id,

            )
            output = tokenizer.decode(generation_output[0])
            parsed = output.split("Output:")[1].strip().strip()[:-4].split(", ")

            parsed_og_output = og_output.strip().strip().split(", ")

            fixed_output = []
            for parsed_title in parsed:
                fixed_title = find_closest_string(parsed_title, all_titles)
                fixed_output.append(fixed_title)

            for row in movie_df.iterrows():
                title = row[1]['title']
                mapping = row[1]['movie_id']

                if title in fixed_output:
                    fixed_output[fixed_output.index(title)] = mapping
                if title in parsed_og_output:
                    parsed_og_output[parsed_og_output.index(title)] = mapping

            set_preds = set(fixed_output)
            set_test = set(parsed_og_output)

            common_elements = set_preds.intersection(set_test)
            precision = len(common_elements) / len(set_preds)

            precision_scores.append(precision)
            progress_bar.update(1)

    # mean average precision
    print(f'Mean Average Precision for {model_type}: {np.mean(precision_scores)}')


if __name__ == '__main__':
    fire.Fire(generate)