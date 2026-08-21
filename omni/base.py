from transformers import AutoModelForCausalLM, AutoTokenizer


def load_base(model_name_or_path):
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model, tokenizer
