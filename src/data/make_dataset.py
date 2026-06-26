import pandas as pd
import re
import string
import unicodedata
import spacy
from nltk.corpus import stopwords
import nltk
import contractions
from deep_translator import GoogleTranslator
import os

# Download stopwords if they are not already present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load spacy model
nlp = spacy.load("en_core_web_sm")

def text_clean(text):
    text = unicodedata.normalize('NFKD', str(text)).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w', '', text)
    text = re.sub(r'http\w+', '', text)
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F700-\U0001F77F"  # alchemical symbols
                               u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                               u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                               u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                               u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                               u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                               u"\U00002702-\U000027B0"  # Dingbats
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    text = contractions.fix(text)
    text = ' '.join(text.split())
    return text

def translate_text(df, col_name='text'):
    translator = GoogleTranslator(source='auto', target='en')
    translated_texts = []
    non_translatable = []
    
    for i, text in enumerate(df[col_name]):
        try:
            translated_text = translator.translate(text)
            translated_texts.append(translated_text)
        except Exception:
            translated_texts.append(text)
            non_translatable.append(i)
            
    return translated_texts, non_translatable

def stopwords_removal_with_covid(corpus):
    stop_w = stopwords.words('english')
    stop_w.extend(["covid", "coronavirus", "sars", "cov"])
    cleaned_corpus = [[word for word in str(doc).split() if word not in stop_w and len(word) > 2] for doc in corpus]
    cleaned_corpus = [' '.join(doc) for doc in cleaned_corpus if len(doc) > 3]
    return cleaned_corpus

def stopwords_removal_no_covid(corpus):
    stop_w = set(stopwords.words('english'))
    cleaned_corpus = [[word for word in str(doc).split() if word not in stop_w] for doc in corpus]
    cleaned_corpus = [' '.join(x) for x in cleaned_corpus]
    return cleaned_corpus

def spacy_lemma(text):
    lemmas = [word.lemma_ for word in nlp(str(text))]
    return " ".join(lemmas)

def process_covid_tweets(raw_path, processed_dir):
    print("Starting preprocessing of COVID tweets...")
    df = pd.read_csv(raw_path)
    
    print("1. Text cleaning...")
    df['cleaned'] = df['text'].apply(text_clean)
    
    print("2. Translation...")
    translated, _ = translate_text(df, 'cleaned')
    
    print("3. Stopwords removal...")
    # Variant 1: Without COVID terms (for topic modeling)
    corpus_off = stopwords_removal_with_covid(translated)
    df_off = pd.DataFrame(corpus_off, columns=['text'])
    df_off = df_off.drop_duplicates().reset_index(drop=True)
    
    # Variant 2: With COVID terms (for EDA plots)
    corpus_graph = stopwords_removal_no_covid(translated)
    df_graph = pd.DataFrame(corpus_graph, columns=['text'])
    df_graph = df_graph.drop_duplicates().reset_index(drop=True)
    
    print("4. Lemmatization...")
    df_off['text'] = df_off['text'].apply(spacy_lemma)
    df_graph['text'] = df_graph['text'].apply(spacy_lemma)
    
    print("Saving results...")
    df_off.to_csv(os.path.join(processed_dir, 'covid19_tweets_off.csv'), index=False)
    df_graph.to_csv(os.path.join(processed_dir, 'covid19_tweets_for_graph.csv'), index=False)
    print("COVID tweets processing completed!")

def process_suspicious_tweets(raw_path, processed_dir):
    print("Starting preprocessing of Suspicious tweets...")
    df = pd.read_csv(raw_path)
    
    print("1. Text cleaning...")
    df['cleaned'] = df['message'].apply(text_clean)
    
    print("2. Translation...")
    translated, _ = translate_text(df, 'cleaned')
    df['translated'] = translated
    
    print("3. Stopwords removal and length filtering...")
    stop_w = stopwords.words('english')
    stop_w.extend(["cannot", "not", "covid", "coronavirus", "sars", "cov"])
    
    cleaned_corpus = []
    removed_idx = []
    
    for i, row in df.iterrows():
        doc = row['translated']
        words = [word for word in str(doc).split() if word not in stop_w and len(word) > 2]
        cleaned_doc = ' '.join(words)
        if len(cleaned_doc.split()) > 3:
            cleaned_corpus.append(cleaned_doc)
        else:
            removed_idx.append(i)
            
    df_final = pd.DataFrame({
        'message': cleaned_corpus,
        'label': df.drop(removed_idx)['label'].tolist()
    })
    
    print("4. Lemmatization...")
    df_final['message'] = df_final['message'].apply(spacy_lemma)
    
    print("Saving results...")
    df_final.to_csv(os.path.join(processed_dir, 'suspicious_tweets_off.csv'), index=False)
    print("Suspicious tweets processing completed!")

if __name__ == '__main__':
    RAW_DIR = os.path.join('data', 'raw')
    PROCESSED_DIR = os.path.join('data', 'processed')
    
    covid_raw = os.path.join(RAW_DIR, 'covid19_tweets.csv')
    sus_raw = os.path.join(RAW_DIR, 'suspicious_tweets.csv')
    
    if os.path.exists(covid_raw):
        process_covid_tweets(covid_raw, PROCESSED_DIR)
    else:
        print(f"File not found: {covid_raw}")
        
    if os.path.exists(sus_raw):
        process_suspicious_tweets(sus_raw, PROCESSED_DIR)
    else:
        print(f"File not found: {sus_raw}")
