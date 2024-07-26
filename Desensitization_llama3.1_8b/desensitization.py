from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Define model ID and tokenizer
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Load tokenizer and model with specified torch dtype
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32,  # Using float32 for compatibility with different devices
    device_map="auto",
)

# Define input messages
messages = [
    {"role": "system", "content": (
        "You are a natural language processor. "
        "Please replace the AWS EC2 instance id with X, "
        "and output the rest of the information as it is. "
        "For example, change [ec2-285dct67i5 is in our cloud] to [ec2-XXX is in our cloud]"
    )},
    {"role": "user", "content": (
        "The down AWS EC2 instance id is ec2-01845dct67i, please page the on-call engineer."
    )},
]

# Tokenize input messages
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
)

# Debugging: Print the structure and content of inputs
print("Inputs structure:", inputs)
print("Shape of inputs:", inputs.shape)

# Create attention mask
attention_mask = torch.ones_like(inputs)

# Ensure inputs and attention mask are on the correct device
inputs = inputs.to(model.device)
attention_mask = attention_mask.to(model.device)

# Debugging: Print device and tensors
print("Model device:", model.device)
print("Inputs device:", inputs.device)
print("Attention mask device:", attention_mask.device)

# Define terminators for the generated output
eos_token_id = tokenizer.eos_token_id
if eos_token_id is None:
    eos_token_id = tokenizer.convert_tokens_to_ids("")

# Generate response from the model
outputs = model.generate(
    inputs,
    attention_mask=attention_mask,
    max_new_tokens=256,
    eos_token_id=eos_token_id,
    do_sample=True,
    temperature=0.5,
    top_p=0.9,
)

# Extract and decode the generated response
response_ids = outputs[0][len(inputs[0]):]
response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

# Print the response
print("Generated Response:")
print(response_text)