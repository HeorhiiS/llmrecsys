from transformers import GPTJForCausalLM, AutoTokenizer
import torch

device = "cuda"

model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16,
                                        cache_dir="gptjmodel/").to(device)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", cache_dir="gptjmodel/")


context = """Recommend 3 movies similar to ('The Dark Knight', 'Iron Man') =>"""

context_fewshot ="""Recommend 3 movies similar to ('Star Wars: Attack of The Clones', 'Star Trek: Beyond') => 'Space Odyssey', 'Star Trek: The Wrath of Khan', 'Dune'
    Recommend 3 movies similar to ('Braveheart', 'Gladiator') => 'Sense and Sensibility', 'The King's Speech', 'The Last Samurai'
    Recommend 3 movies similar to ('Princess Mononoke', 'Spirited Away') =>""",

input_ids = tokenizer(context_fewshot, return_tensors="pt").input_ids
input_ids = input_ids.to(device)
gen_tokens = model.generate(input_ids, do_sample=True, temperature=0.9, max_length=250)
gen_text = tokenizer.batch_decode(gen_tokens)[0]
print(gen_text)