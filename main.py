from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/blenderbot-400M-distill"
#model_name = "EleutherAI/gpt-j-6B"
# Load model (download on first run and reference local installation for consequent runs)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

'''conversation_history = []

while True:
    # Create conversation history string
    history_string = "\n".join(conversation_history)

    # Get the input data from the user
    input_text = input("> ")

    # Tokenize the input text and history
    inputs = tokenizer.encode_plus(history_string, input_text, return_tensors="pt")

    # Generate the response from the model
    outputs = model.generate(**inputs)

    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    print(response)

    # Add interaction to conversation history
    conversation_history.append(input_text)
    conversation_history.append(response)'''
# Keep conversation context manageable
conversation_history = ""

while True:
    user_input = input("You: ")

    # Combine last exchange into input (basic format)
    full_input = conversation_history + user_input

    # Tokenize
    inputs = tokenizer(full_input, return_tensors="pt", truncation=True)

    # Generate with safeguards
    output = model.generate(
        **inputs,
        max_length=100,
        repetition_penalty=1.3,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.eos_token_id,
    )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print("Bot:", response)

    # Append this interaction to history
    conversation_history += f" {user_input} {response}"
