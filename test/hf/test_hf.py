import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

# checkpoint = "facebook/opt-125m"
checkpoint = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def test_login():
    from huggingface_hub import login

    from src.utils.env_config import settings

    token = settings.hf_token.get_secret_value()
    login(token=token)


def test_chat_template():
    messages = [
        {
            "role": "assistant",
            "content": "You are a friendly chatbot who always responds in the style of a pirate",
        },
        {
            "role": "user",
            "content": "How many helicopters can a human eat in one sitting?",
        },
    ]

    conversation_string = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        return_tensors="pt",
    )
    print(tokenizer.default_chat_template)
    print(conversation_string)


def test_model():
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        # bnb_4bit_use_double_quant=True,
        # bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        device_map="auto",
        quantization_config=quantization_config,
    )
    print(model.get_memory_footprint() / 1024**3)  # in GB
    generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer)
    print(generator("Hello, how are you?", max_length=100)[0]["generated_text"])  # type: ignore


def test_chat_pipeline():
    checkpoint = "meta-llama/Meta-Llama-3-8B-Instruct"
    checkpoint = "meta-llama/Llama-2-7b-chat-hf"
    generator = pipeline("text-generation", model=checkpoint, tokenizer=checkpoint)
    messages = [
        {
            "role": "system",
            "content": "You are a friendly chatbot who always responds in the style of a pirate",
        },
        {
            "role": "user",
            "content": "How many helicopters can a human eat in one sitting?",
        },
    ]
    print(generator(messages, max_new_tokens=128)[0]["generated_text"])  # type: ignore
