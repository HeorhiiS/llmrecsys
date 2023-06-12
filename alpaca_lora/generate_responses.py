import sys
import fire
import torch
import Levenshtein as leven
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
        batch_size: int = 10,
):

    # Load the model
    device_map = "auto" if torch.cuda.is_available() else "cpu"

    print(f"Is CUDA available? => {torch.cuda.is_available()}")

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

    def find_most_similar_string(query, string_list):
        most_similar_string = None
        highest_similarity = 0

        for string in string_list:
            similarity = leven.distance(query, string)
            if most_similar_string is None or similarity < highest_similarity:
                most_similar_string = string
                highest_similarity = similarity

        return most_similar_string

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    eval_data = load_dataset("json", data_files="../litllamadata/finetune_dataset/llama_eval_red.json")
    precision_scores = []
    batched_prompt = []
    batched_og_output = []

    batch_count = 0
    global_counter = 0
    num_batches = (len(eval_data['train']) // batch_size) * batch_size
    remainder = len(eval_data['train']) % batch_size
    movie_df = pd.read_json('../movie_map.json')
    all_titles = movie_df['title']
    all_titles = np.array(all_titles)

    progress_bar = tqdm.tqdm(total=len(eval_data['train']), ncols=100, colour='magenta', ascii="░▒█")

    for prompt in eval_data['train']:

        condition1 = (batch_count !=0) and batch_count % batch_size == 0
        condition2 = num_batches == global_counter

        if not condition2:

            instruction = prompt['instruction']
            input = prompt['input']
            og_output = prompt['output']

            prompt = f'### Instruction: {instruction}\n ### Input: ' + f"{input}\n ### Output:"
            batched_prompt.append(prompt)
            batched_og_output.append(og_output)
            batch_count += 1

            if condition1:
                input_ids = tokenizer(batched_prompt, padding=True, return_tensors="pt").input_ids
                input_ids = input_ids.to('cuda')
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

                    output = tokenizer.batch_decode(generation_output)


                    for i in range(len(output)):
                        parsed = output[i].split("Output:")[1].split("</s>")[0].strip().split(", ")[:4]
                        parsed_og_output = batched_og_output[i].strip().split(", ")

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
                batched_prompt = []
                batch_count = 0

        else:
            instruction = prompt['instruction']
            input = prompt['input']
            og_output = prompt['output']

            prompt = f'### Instruction: {instruction}\n ### Input: ' + f"{input}\n ### Output:"
            batched_prompt.append(prompt)
            batched_og_output.append(og_output)

            input_ids = tokenizer(batched_prompt, padding=True, return_tensors="pt").input_ids
            input_ids = input_ids.to('cuda')

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
                output = tokenizer.batch_decode(generation_output[0])

                for i in range(len(output)):

                    parsed = output[i].split("Output:")[1].split("</s>")[0].strip().split(", ")[:4]

                    parsed_og_output = batched_og_output[i].strip().split(", ")

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
            batched_prompt = []

        global_counter += 1

    # mean average precision
    print(f'Mean Average Precision for {model_type}: {np.mean(precision_scores)}')


if __name__ == '__main__':
    fire.Fire(generate)