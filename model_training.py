!pip install datasets rouge-score bert-score evaluate


import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer
from torch.utils.data import Dataset
import json
import numpy as np
#from evaluate import load_metric
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
import nltk
from bert_score import score
import tensorflow as tf

# Download required NLTK data
nltk.download('wordnet')
nltk.download('punkt')

# Load and prepare data
def load_qa_data(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    questions = []
    answers = []
    contexts = []

    for article in data['data']:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                questions.append(qa['question'])
                answers.append(qa['answers'][0]['text'])
                contexts.append(context)

    return questions, answers, contexts

class QADataset(Dataset):
    def __init__(self, questions, answers, contexts, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.questions = questions
        self.answers = answers
        self.contexts = contexts
        self.max_length = max_length

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        answer = self.answers[idx]
        context = self.contexts[idx]

        # Tokenize without moving to device
        encoding = self.tokenizer(
            question,
            context,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        # Get answer span
        answer_start = context.lower().find(answer.lower())
        answer_end = answer_start + len(answer)

        tokens = self.tokenizer.encode_plus(
            context,
            max_length=self.max_length,
            truncation=True,
            return_offsets_mapping=True
        )
        offsets = tokens['offset_mapping']

        start_token = 0
        end_token = 0
        for idx, (start, end) in enumerate(offsets):
            if start <= answer_start < end:
                start_token = idx
            if start < answer_end <= end:
                end_token = idx
                break

        # Create tensors on CPU
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'start_positions': torch.tensor(start_token, dtype=torch.long),
            'end_positions': torch.tensor(end_token, dtype=torch.long)
        }

def train_bert(questions, answers, contexts, model_name='bert-base-uncased', epochs=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Split data
    from sklearn.model_selection import train_test_split

    train_questions, val_questions, train_answers, val_answers, train_contexts, val_contexts = train_test_split(
        questions, answers, contexts, test_size=0.2, random_state=42
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    train_dataset = QADataset(train_questions, train_answers, train_contexts, tokenizer)
    val_dataset = QADataset(val_questions, val_answers, val_contexts, tokenizer)

    # Define data collator
    from transformers import default_data_collator
    data_collator = default_data_collator

    training_args = TrainingArguments(
        output_dir=f"./results_{model_name}",
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f'./logs_{model_name}',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        learning_rate=2e-5,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_pin_memory=False,  # Disable pin memory
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    return model.to(device), tokenizer

def ask_question(question, context, model, tokenizer):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Tokenize input
    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True
    )

    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)

    # Process outputs
    answer_start = torch.argmax(outputs.start_logits).item()
    answer_end = torch.argmax(outputs.end_logits).item()

    # Decode answer
    answer_tokens = inputs['input_ids'][0][answer_start:answer_end + 1]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

    return answer

# Add a custom data collator if needed
def custom_data_collator(features):
    batch = {}

    # Collect all keys from the first feature
    keys = features[0].keys()

    for key in keys:
        batch[key] = torch.stack([f[key] for f in features])

    return batch

# Update the other model training functions similarly
def train_roberta(questions, answers, contexts, epochs=5):
    return train_bert(questions, answers, contexts, model_name='roberta-base', epochs=epochs)

def train_distilbert(questions, answers, contexts, epochs=5):
    return train_bert(questions, answers, contexts, model_name='distilbert-base-uncased', epochs=epochs)


# BiLSTM Model
class BiLSTMQA(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=256):
        super(BiLSTMQA, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        output = self.fc(lstm_out)
        return output

def calculate_metrics(predictions, references):
    # BERT Score
    P, R, F1 = score(predictions, references, lang='en', verbose=True)
    bert_score = F1.mean().item()

    # ROUGE Score
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = [scorer.score(pred, ref) for pred, ref in zip(predictions, references)]
    rouge_1 = np.mean([score['rouge1'].fmeasure for score in rouge_scores])
    rouge_2 = np.mean([score['rouge2'].fmeasure for score in rouge_scores])
    rouge_l = np.mean([score['rougeL'].fmeasure for score in rouge_scores])

    # BLEU Score
    bleu_scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = nltk.word_tokenize(pred.lower())
        ref_tokens = [nltk.word_tokenize(ref.lower())]
        bleu_scores.append(sentence_bleu(ref_tokens, pred_tokens))
    bleu_score = np.mean(bleu_scores)

    # METEOR Score
    meteor_scores = []
    for pred, ref in zip(predictions, references):
        meteor_scores.append(meteor_score([ref], pred))
    meteor_score_avg = np.mean(meteor_scores)

    return {
        'bert_score': bert_score,
        'rouge_1': rouge_1,
        'rouge_2': rouge_2,
        'rouge_l': rouge_l,
        'bleu': bleu_score,
        'meteor': meteor_score_avg
    }

# Load data
questions, answers, contexts = load_qa_data('final_squad_data.json')


# BiLSTM Model Implementation
class BiLSTMQA(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=256, num_layers=2, dropout=0.3):
        super(BiLSTMQA, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.start_linear = nn.Linear(hidden_dim * 2, 1)
        self.end_linear = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask=None):
        # input_ids shape: (batch_size, sequence_length)
        embedded = self.dropout(self.embedding(input_ids))

        # Apply LSTM
        lstm_out, _ = self.lstm(embedded)

        # Get start and end logits
        start_logits = self.start_linear(lstm_out).squeeze(-1)
        end_logits = self.end_linear(lstm_out).squeeze(-1)

        # Mask padded positions
        if attention_mask is not None:
            start_logits = start_logits * attention_mask
            end_logits = end_logits * attention_mask

        return start_logits, end_logits

# BiLSTM training function
def train_bilstm(questions, answers, contexts, epochs=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create vocabulary and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    vocab_size = tokenizer.vocab_size

    # Initialize model
    model = BiLSTMQA(vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    # Split data
    train_questions, val_questions, train_answers, val_answers, train_contexts, val_contexts = train_test_split(
        questions, answers, contexts, test_size=0.2, random_state=42
    )

    # Create datasets
    train_dataset = QADataset(train_questions, train_answers, train_contexts, tokenizer)
    val_dataset = QADataset(val_questions, val_answers, val_contexts, tokenizer)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)

            start_logits, end_logits = model(input_ids, attention_mask)

            loss = criterion(start_logits, start_positions) + criterion(end_logits, end_positions)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                start_positions = batch['start_positions'].to(device)
                end_positions = batch['end_positions'].to(device)

                start_logits, end_logits = model(input_ids, attention_mask)
                loss = criterion(start_logits, start_positions) + criterion(end_logits, end_positions)
                val_loss += loss.item()

        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Training Loss: {total_loss/len(train_loader):.4f}')
        print(f'Validation Loss: {val_loss/len(val_loader):.4f}')

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_bilstm_model.pt')

    return model, tokenizer

# Updated evaluation metrics
def calculate_metrics(predictions, references):
    """
    Calculate various NLP evaluation metrics
    """
    from rouge import Rouge
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    import evaluate

    # Initialize metrics
    rouge = Rouge()
    bertscore = evaluate.load("bertscore")
    smooth = SmoothingFunction().method1

    # Calculate ROUGE scores
    try:
        rouge_scores = rouge.get_scores([p for p in predictions], [r for r in references], avg=True)
    except:
        rouge_scores = {'rouge-1': {'f': 0}, 'rouge-2': {'f': 0}, 'rouge-l': {'f': 0}}

    # Calculate BERT Score
    try:
        bert_scores = bertscore.compute(predictions=predictions,
                                      references=references,
                                      lang="en")
        bert_f1 = sum(bert_scores['f1']) / len(bert_scores['f1'])
    except:
        bert_f1 = 0

    # Calculate BLEU scores
    bleu_scores = []
    for pred, ref in zip(predictions, references):
        try:
            pred_tokens = pred.split()
            ref_tokens = [ref.split()]
            score = sentence_bleu(ref_tokens, pred_tokens,
                                smoothing_function=smooth)
            bleu_scores.append(score)
        except:
            bleu_scores.append(0)
    bleu_avg = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0

    # Calculate METEOR scores
    meteor_scores = []
    for pred, ref in zip(predictions, references):
        try:
            score = meteor_score([ref.split()], pred.split())
            meteor_scores.append(score)
        except:
            meteor_scores.append(0)
    meteor_avg = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0

    return {
        'rouge1_f1': rouge_scores['rouge-1']['f'],
        'rouge2_f1': rouge_scores['rouge-2']['f'],
        'rougeL_f1': rouge_scores['rouge-l']['f'],
        'bert_score': bert_f1,
        'bleu': bleu_avg,
        'meteor': meteor_avg
    }



def evaluate_all_models(test_questions, test_contexts, test_answers, models, tokenizers):
    """
    Evaluate all models and compare their performance
    """
    results = {
        'bert': [],
        'roberta': [],
        'distilbert': [],
        'bilstm': []
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get predictions from each model
    for model_name in results.keys():
        model = models[model_name].to(device)
        tokenizer = tokenizers[model_name]
        model.eval()

        predictions = []
        with torch.no_grad():
            for question, context in zip(test_questions, test_contexts):
                if model_name == 'bilstm':
                    # Special handling for BiLSTM model
                    inputs = tokenizer(
                        question,
                        context,
                        return_tensors="pt",
                        max_length=512,
                        truncation=True,
                        padding=True
                    )
                    input_ids = inputs['input_ids'].to(device)
                    attention_mask = inputs['attention_mask'].to(device)

                    start_logits, end_logits = model(input_ids, attention_mask)
                    start_idx = torch.argmax(start_logits).item()
                    end_idx = torch.argmax(end_logits).item()

                    answer_tokens = input_ids[0][start_idx:end_idx + 1]
                    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
                else:
                    # For transformer models
                    answer = ask_question(question, context, model, tokenizer)

                predictions.append(answer)

        results[model_name] = predictions

    # Calculate metrics for each model
    metrics = {}
    for model_name, predictions in results.items():
        print(f"\nEvaluating {model_name.upper()} model:")
        model_metrics = calculate_metrics(predictions, test_answers)
        metrics[model_name] = model_metrics

        print(f"ROUGE-1 F1: {model_metrics['rouge1_f1']:.4f}")
        print(f"ROUGE-2 F1: {model_metrics['rouge2_f1']:.4f}")
        print(f"ROUGE-L F1: {model_metrics['rougeL_f1']:.4f}")
        print(f"BERT Score: {model_metrics['bert_score']:.4f}")
        print(f"BLEU Score: {model_metrics['bleu']:.4f}")
        print(f"METEOR Score: {model_metrics['meteor']:.4f}")

    return metrics, results

# Function to visualize results
def plot_metrics_comparison(metrics):
    """
    Create visualizations comparing model performances
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Prepare data for plotting
    models = list(metrics.keys())
    metric_names = list(metrics[models[0]].keys())

    # Create subplots for each metric
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Model Performance Comparison')

    for idx, metric in enumerate(metric_names):
        row = idx // 3
        col = idx % 3

        values = [metrics[model][metric] for model in models]

        sns.barplot(x=models, y=values, ax=axes[row, col])
        axes[row, col].set_title(metric)
        axes[row, col].set_xticklabels(models, rotation=45)

    plt.tight_layout()
    plt.show()

# Main execution with all models
def main():
    try:
        # Load data
        questions, answers, contexts = load_qa_data('final_squad_data.json')

        # Split data
        train_questions, test_questions, train_answers, test_answers, train_contexts, test_contexts = train_test_split(
            questions, answers, contexts, test_size=0.2, random_state=42
        )

        # Train all models
        print("Training BERT model...")
        bert_model, bert_tokenizer = train_bert(train_questions, train_answers, train_contexts, epochs=10)

        print("\nTraining RoBERTa model...")
        roberta_model, roberta_tokenizer = train_roberta(train_questions, train_answers, train_contexts, epochs=10)

        print("\nTraining DistilBERT model...")
        distilbert_model, distilbert_tokenizer = train_distilbert(train_questions, train_answers, train_contexts, epochs=10)

        print("\nTraining BiLSTM model...")
        bilstm_model, bilstm_tokenizer = train_bilstm(train_questions, train_answers, train_contexts)

        # Prepare models dictionary
        models = {
            'bert': bert_model,
            'roberta': roberta_model,
            'distilbert': distilbert_model,
            'bilstm': bilstm_model
        }

        tokenizers = {
            'bert': bert_tokenizer,
            'roberta': roberta_tokenizer,
            'distilbert': distilbert_tokenizer,
            'bilstm': bilstm_tokenizer
        }

        # Evaluate all models
        metrics, results = evaluate_all_models(
            test_questions,
            test_contexts,
            test_answers,
            models,
            tokenizers
        )

        # Plot results
        plot_metrics_comparison(metrics)

        # Save results
        save_results(metrics, results, 'qa_model_results.json')

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

def save_results(metrics, results, filename):
    """
    Save evaluation results to a file
    """
    output = {
        'metrics': metrics,
        'predictions': results
    }

    with open(filename, 'w') as f:
        json.dump(output, f, indent=4)


def analyze_errors(predictions, references, threshold=0.5):
    """
    Analyze cases where models performed poorly
    """
    from difflib import SequenceMatcher

    error_cases = []

    for idx, (pred, ref) in enumerate(zip(predictions, references)):
        # Calculate similarity ratio
        similarity = SequenceMatcher(None, pred.lower(), ref.lower()).ratio()

        if similarity < threshold:
            error_cases.append({
                'index': idx,
                'prediction': pred,
                'reference': ref,
                'similarity': similarity
            })

    return error_cases

def generate_error_report(error_cases, output_file='error_analysis.txt'):
    """
    Generate a detailed error analysis report
    """
    with open(output_file, 'w') as f:
        f.write("Error Analysis Report\n")
        f.write("===================\n\n")

        for case in error_cases:
            f.write(f"Case {case['index']}:\n")
            f.write(f"Prediction: {case['prediction']}\n")
            f.write(f"Reference: {case['reference']}\n")
            f.write(f"Similarity: {case['similarity']:.4f}\n")
            f.write("-" * 50 + "\n")


