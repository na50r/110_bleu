from transformers import AutoModelForSeq2SeqLM, MarianTokenizer
# Based on: https://gist.github.com/ashleyha/04cb3879b1d869adfce9e0cd0794f094

def translate(src_lang, tgt_lang, src_text, model_name=None):
    # If model not specified, use Helsinki-NLP OPUS model for the language pair
    if model_name is None:
        model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    # Load model and tokenizer
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    input_ids = tokenizer.encode(src_text, return_tensors="pt")
    outputs = model.generate(input_ids)
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translation
    
# Example usage:
if __name__ == "__main__":
    # Example 1: Helsinki model (language pair specific)
    result1 = translate("de", "en", "Hallo, wie geht es dir?")
    print(f"German to English: {result1}")
