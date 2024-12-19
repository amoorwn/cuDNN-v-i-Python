from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
from sklearn.model_selection import train_test_split
import time

# Generate synthetic text data
texts = ["This is a positive example." for _ in range(500)] + \
        ["This is a negative example." for _ in range(500)]
labels = [1] * 500 + [0] * 500

# Tokenize data
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
labels = torch.tensor(labels)

train_inputs, val_inputs, train_labels, val_labels = train_test_split(inputs, labels, test_size=0.2)

# Create a BERT model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    num_train_epochs=2,
    logging_dir="./logs",
    logging_steps=10,
)

# Train with and without cuDNN
for use_cudnn in [False, True]:
    torch.backends.cudnn.enabled = use_cudnn

    start_time = time.time()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset={"input_ids": train_inputs["input_ids"], "labels": train_labels},
        eval_dataset={"input_ids": val_inputs["input_ids"], "labels": val_labels},
    )
    trainer.train()
    duration = time.time() - start_time

    print(f"Time taken {'with' if use_cudnn else 'without'} cuDNN: {duration:.2f} seconds")
