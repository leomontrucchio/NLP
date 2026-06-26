import pandas as pd
import numpy as np
import os
import json
import nltk
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import tomotopy as tp
import scipy.special

# Ensure punkt is downloaded for tokenization
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

def find_opt_thresh(thresholds, prob, test_labels):
    opt_t = None
    binary_predictions_def = None
    max_f1 = 0
    for t in thresholds:
        binary_predictions = (prob > t).astype(int)
        f1_temp = f1_score(test_labels, binary_predictions, average='weighted', zero_division=0)
        if f1_temp > max_f1:
            opt_t = t
            binary_predictions_def = binary_predictions
            max_f1 = f1_temp
    return opt_t, binary_predictions_def

def train_slda(processed_data_path, models_dir):
    print("Loading processed suspicious tweets...")
    df = pd.read_csv(processed_data_path)
    df = df.dropna(subset=['message', 'label'])
    
    docs = df['message'].tolist()
    labels = df['label'].tolist()
    
    print("Splitting into train and test sets...")
    train_set, test_set, train_labels, test_labels = train_test_split(docs, labels, test_size=0.2, random_state=42)
    
    print("Creating corpus...")
    corpus = tp.utils.Corpus()
    for doc_text, label in zip(train_set, train_labels):
        words = word_tokenize(str(doc_text))
        corpus.add_doc(words, y=[label])
        
    n_topics = range(10, 101, 10)
    n_thresh = np.round(np.arange(0.5, 1, 0.05), 2)
    
    max_f1 = 0
    opt_k = None
    opt_t = None
    binary_predictions_def = None
    slda_model_def = None
    
    print("Training sLDA models to find the optimal number of topics and threshold...")
    for k in n_topics:
        print(f"  Training sLDA with {k} topics...")
        slda_model = tp.SLDAModel(vars=['b'], k=k)
        slda_model.add_corpus(corpus)
        # Train the model with 1000 iterations and 1 worker to suppress RuntimeWarning
        slda_model.train(1000, workers=1)
        
        coef = slda_model.get_regression_coef(0)
        
        # Test predictions
        test_predictions = []
        for doc in test_set:
            tp_doc = slda_model.make_doc(word_tokenize(str(doc)))
            test_predictions.append(slda_model.infer(tp_doc))
            
        y_test = [np.dot(coef, test_predictions[i][0]) for i in range(len(test_predictions))]
        prob = scipy.special.expit(y_test)
        
        current_opt_t, binary_predictions = find_opt_thresh(n_thresh, prob, test_labels)
        temp_f1 = f1_score(test_labels, binary_predictions, average='weighted', zero_division=0)
        
        print(f"  -> Best Threshold: {current_opt_t}, F1-score: {temp_f1:.4f}")
        
        if temp_f1 > max_f1:
            max_f1 = temp_f1
            opt_k = k
            opt_t = current_opt_t
            binary_predictions_def = binary_predictions
            slda_model_def = slda_model
            
    print(f"\nOptimization completed!")
    print(f"Optimal number of topics: {opt_k}")
    print(f"Optimal threshold: {opt_t}")
    
    print("\nComputing final metrics on test set:")
    accuracy = accuracy_score(test_labels, binary_predictions_def)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-score: {max_f1:.4f}")
    
    # Calculate recall and precision for the positive class (1)
    true_positives = sum((true == 1) and (pred == 1) for true, pred in zip(test_labels, binary_predictions_def))
    false_positives = sum((true != 1) and (pred == 1) for true, pred in zip(test_labels, binary_predictions_def))
    false_negatives = sum((true == 1) and (pred != 1) for true, pred in zip(test_labels, binary_predictions_def))
    
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    
    print(f"Recall: {recall:.2f}")
    print(f"Precision: {precision:.2f}")
    
    print("\nSaving the optimal sLDA model and metrics...")
    os.makedirs(models_dir, exist_ok=True)
    # Save the model to a binary file
    slda_model_def.save(os.path.join(models_dir, 'slda_model.bin'), full=True)
    
    # Save metrics to JSON for the final results notebook
    metrics = {
        'optimal_topics': int(opt_k),
        'optimal_threshold': float(opt_t),
        'accuracy': round(accuracy, 4),
        'f1_score': round(max_f1, 4),
        'recall': round(recall, 4),
        'precision': round(precision, 4)
    }
    with open(os.path.join(models_dir, 'slda_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("sLDA training completed successfully!")

if __name__ == '__main__':
    PROCESSED_DATA = os.path.join('data', 'processed', 'suspicious_tweets_off.csv')
    MODELS_DIR = 'models'
    
    if os.path.exists(PROCESSED_DATA):
        train_slda(PROCESSED_DATA, MODELS_DIR)
    else:
        print(f"File not found: {PROCESSED_DATA}. Please run make_dataset.py first.")
