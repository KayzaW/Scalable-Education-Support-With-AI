from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering, AutoModelForCausalLM

# Download and save BERT-based model for grading
bert_model = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-MRPC")
bert_tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-MRPC")
bert_model.save_pretrained("./models/bert-base-uncased-MRPC")
bert_tokenizer.save_pretrained("./models/bert-base-uncased-MRPC")

# Download and save BART-based model for summarization
bart_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
bart_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
bart_model.save_pretrained("./models/bart-large-cnn")
bart_tokenizer.save_pretrained("./models/bart-large-cnn")

# Download and save RoBERTa-based model for question answering
roberta_model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
roberta_tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
roberta_model.save_pretrained("./models/roberta-base-squad2")
roberta_tokenizer.save_pretrained("./models/roberta-base-squad2")

# Download and save GPT-2 model for text generation
gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2")
gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
gpt2_model.save_pretrained("./models/gpt2")
gpt2_tokenizer.save_pretrained("./models/gpt2")

