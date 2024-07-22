import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoModelForQuestionAnswering


models_dir = './models'


print("Downloading roberta-large-mnli...")
roberta_model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")
roberta_tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
roberta_model.save_pretrained(os.path.join(models_dir, "roberta-large-mnli"))
roberta_tokenizer.save_pretrained(os.path.join(models_dir, "roberta-large-mnli"))


print("Downloading facebook/bart-large-cnn...")
bart_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
bart_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
bart_model.save_pretrained(os.path.join(models_dir, "bart-large-cnn"))
bart_tokenizer.save_pretrained(os.path.join(models_dir, "bart-large-cnn"))


print("Downloading EleutherAI/gpt-neo-2.7B...")
gpt_neo_model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")
gpt_neo_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
gpt_neo_model.save_pretrained(os.path.join(models_dir, "gpt-neo-2.7B"))
gpt_neo_tokenizer.save_pretrained(os.path.join(models_dir, "gpt-neo-2.7B"))


print("Downloading bert-large-uncased-whole-word-masking-finetuned-squad...")
bert_qa_model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
bert_qa_tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
bert_qa_model.save_pretrained(os.path.join(models_dir, "bert-large-uncased-whole-word-masking-finetuned-squad"))
bert_qa_tokenizer.save_pretrained(os.path.join(models_dir, "bert-large-uncased-whole-word-masking-finetuned-squad"))

print("All models have been downloaded and old models removed.")
