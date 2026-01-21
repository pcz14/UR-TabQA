from transformers import AutoTokenizer

local_path = ''
tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)

messages = [{'content': 'You are a helpful assistant', 'role': 'system'}, 
            {'content': 'question".', 'role': 'user'}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
print(prompt)
