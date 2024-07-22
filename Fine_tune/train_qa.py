import os
import json
import torch
import time
import GPUtil
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict

model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
output_dir = "./fine_tuned_qa_model"
data_file = "question_answering_dataset.json"

# Function to monitor and print GPU usage
def print_gpu_usage():
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print(f"GPU {gpu.id}: {gpu.name}, Memory Used: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB")

def load_custom_dataset(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)["data"]

    contexts = []
    questions = []
    answers = []

    for entry in data:
        for paragraph in entry["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                question = qa["question"]
                answer_list = qa["answers"]
                if len(answer_list) == 0:
                    continue
                answer_texts = [answer['text'] for answer in answer_list]
                answer_starts = [answer['answer_start'] for answer in answer_list]

                contexts.append(context)
                questions.append(question)
                answers.append({'text': answer_texts, 'answer_start': answer_starts})

    return Dataset.from_dict({'context': contexts, 'question': questions, 'answers': answers})

# Load dataset
dataset = load_custom_dataset(data_file)
train_test_split = dataset.train_test_split(test_size=0.1)
dataset = DatasetDict({
    'train': train_test_split['train'],
    'validation': train_test_split['test']
})

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def prepare_train_features(examples):
    tokenized_examples = tokenizer(
        examples['question'],
        examples['context'],
        truncation="only_second",
        max_length=384,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized_examples.sequence_ids(i)
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        answer_starts = answers["answer_start"]
        answer_texts = answers["text"]

        if len(answer_starts) == 0:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            start_char = answer_starts[0]
            end_char = start_char + len(answer_texts[0])

            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                start_positions.append(cls_index)
                end_positions.append(cls_index)
            else:
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                start_positions.append(token_start_index - 1)

                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                end_positions.append(token_end_index + 1)

    tokenized_examples["start_positions"] = start_positions
    tokenized_examples["end_positions"] = end_positions

    return tokenized_examples

# Preprocess dataset
tokenized_train_dataset = dataset['train'].map(prepare_train_features, batched=True, remove_columns=dataset["train"].column_names)
tokenized_validation_dataset = dataset['validation'].map(prepare_train_features, batched=True, remove_columns=dataset["validation"].column_names)

# Define training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,
    save_total_limit=2,
    save_steps=500,
    logging_steps=100,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_validation_dataset,
    tokenizer=tokenizer,
)

# Measure retraining time
print("Starting training with GPU usage monitoring...")
print_gpu_usage()

start_time = time.time()
trainer.train()
end_time = time.time()

print_gpu_usage()

print(f"Retraining Time: {end_time - start_time:.4f} seconds")

trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
