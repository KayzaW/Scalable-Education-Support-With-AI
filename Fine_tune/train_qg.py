import os
import json
import time
import torch
import GPUtil
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset, DatasetDict

model_name = "EleutherAI/gpt-neo-2.7B"
output_dir = "./fine_tuned_qg_model"
data_file = "./question_generation_dataset.json"


def print_gpu_usage():
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print(f"GPU {gpu.id}: {gpu.name}, Memory Used: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB")

def load_custom_dataset(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)["data"]

    contexts = []
    questions = []

    for entry in data:
        for paragraph in entry["paragraphs"]:
            context = paragraph["context"]
            for question in paragraph["questions"]:
                contexts.append(context)
                questions.append(question)

    return Dataset.from_dict({"context": contexts, "question": questions})


custom_dataset = load_custom_dataset(data_file)

train_test_split = custom_dataset.train_test_split(test_size=0.1)
dataset_dict = DatasetDict({
    'train': train_test_split['train'],
    'validation': train_test_split['test']
})

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.resize_token_embeddings(len(tokenizer))

def preprocess_function(examples):
    inputs = examples["context"]
    targets = examples["question"]

    model_inputs = tokenizer(inputs, max_length=384, truncation=True, padding="max_length")
    labels = tokenizer(text_target=targets, max_length=64, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_dataset = dataset_dict.map(preprocess_function, batched=True)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=2,
    weight_decay=0.01,
    save_steps=500,
    save_total_limit=2,
    fp16=True,
    gradient_accumulation_steps=2,
    logging_steps=100,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Measure retraining time
print("Starting training with GPU usage monitoring...")
print_gpu_usage()

start_time = time.time()
trainer.train()
end_time = time.time()

print_gpu_usage()

print(f"Retraining Time: {end_time - start_time:.4f} seconds")

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
