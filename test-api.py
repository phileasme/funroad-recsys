import anthropic

client = anthropic.Anthropic(api_key="<>")

try:
    message = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=10,
        messages=[
            {"role": "user", "content": "Hello, Claude!"}
        ]
    )
    print("Success:", message.content)
except Exception as e:
    print("Error:", e)