from transformers import pipeline, RagTokenizer, RagRetriever, RagTokenForGeneration, TrainingArguments, Trainer
import pandas as pd
from datasets import Dataset
import os

# Get the absolute path to the models directory
base_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(base_dir, '../models')
uploads_dir = os.path.join(base_dir, '../uploads')

# Initialize Hugging Face pipelines
classifier = pipeline('text-classification', model=os.path.join(models_dir, 'bert-base-uncased-MRPC'))
text_generator = pipeline('text-generation', model=os.path.join(models_dir, 'gpt2'))
feedback_generator = pipeline('summarization', model=os.path.join(models_dir, 'bart-large-cnn'))
weakness_identifier = pipeline('question-answering', model=os.path.join(models_dir, 'roberta-base-squad2'))

def load_additional_context():
    context = ""
    for filename in os.listdir(uploads_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(uploads_dir, filename), 'r', encoding='utf-8') as file:
                context += file.read() + "\n"
    return context

additional_context = load_additional_context()

def grade_assignment(text):
    results = classifier(text)
    return results

def generate_exercise(prompt):
    combined_prompt = additional_context + "\n" + prompt
    results = text_generator(combined_prompt, max_length=100, num_return_sequences=1)
    return results[0]['generated_text']

def generate_feedback(text):
    combined_text = additional_context + "\n" + text
    results = feedback_generator(combined_text, max_length=100)
    return results[0]['summary_text']

def identify_weaknesses(text, question):
    combined_text = additional_context + "\n" + text
    results = weakness_identifier(question=question, context=combined_text)
    return results['answer']

def train_rag_model(dataset_path):
    # Load the dataset
    dataset = pd.read_csv(dataset_path)
    questions = dataset['question'].tolist()
    contexts = dataset['context'].tolist()
    answers = dataset['answer'].tolist()

    # Initialize the tokenizer and retriever
    tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
    retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", use_dummy_dataset=True)
    model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)

    # Tokenize the inputs
    def tokenize_function(examples):
        inputs = tokenizer(examples['question'], return_tensors="pt", padding=True, truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples['answer'], return_tensors="pt", padding=True, truncation=True)
        inputs["labels"] = labels["input_ids"]
        return inputs

    tokenized_datasets = Dataset.from_dict({
        'question': questions,
        'context': contexts,
        'answer': answers
    }).map(tokenize_function, batched=True)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=2,
        num_train_epochs=3,
        save_steps=10_000,
        save_total_limit=2,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
    )

    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model("./trained_model")
