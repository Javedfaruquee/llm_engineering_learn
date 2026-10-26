import tiktoken

encoding = tiktoken.encoding_for_model("gpt-4o-mini")

tokens = encoding.encode("Hi my name is 'Shaikh Mohammed Javed Iqbal Faruquee' and I like Kebabs")

print(f"Number of tokens: {len(tokens)}")
print(f"Tokens: {tokens}")

for token_id in tokens:
    token_text = encoding.decode([token_id])
    print(f"{token_id} = {token_text}")