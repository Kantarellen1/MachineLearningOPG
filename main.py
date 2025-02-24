from transformers import MBartForConditionalGeneration, MBart50TokenizerFast


model_name = "facebook/mbart-large-50"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

def summarize_danish(text, max_length=100, min_length=30):
    
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    
    
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        min_length=min_length,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )

    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Example Danish text
danish_text = """Den danske regering har annonceret nye tiltag for at forbedre miljøet og reducere CO2-udledningen. 
Blandt de vigtigste initiativer er en investering i grøn energi og øgede skatter på fossile brændstoffer."""

print(summarize_danish(danish_text))
