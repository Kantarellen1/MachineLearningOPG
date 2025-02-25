import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="corpus.txt", 
    model_prefix="danish_tokenizer", 
    vocab_size=200, 
    character_coverage=1.0, 
    model_type="bpe"
)