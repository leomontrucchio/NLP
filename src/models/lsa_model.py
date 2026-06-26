import pandas as pd
import os
import nltk
from nltk.tokenize import word_tokenize
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel


try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

def train_lsa(processed_data_path, models_dir):
    print("Loading processed data...")
    df = pd.read_csv(processed_data_path)
    df = df.dropna(subset=['text'])
    documents = df['text'].tolist()
    
    print("Tokenizing documents...")
    tokenized_documents = [word_tokenize(str(doc)) for doc in documents]
    
    print("Creating dictionary and TF-IDF corpus...")
    dictionary = corpora.Dictionary(tokenized_documents)
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_documents]
    
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    
    num_topics_list = [5, 20, 40]
    best_coherence = float('inf')
    best_model = None
    best_num_topics = 0
    
    print("Training Latent Semantic Analysis (LSA) models and calculating Coherence...")
    for num_topics in num_topics_list:
        print(f"  Training LSA with {num_topics} topics...")
        lsa_model = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=num_topics)
        
        coherence_model = CoherenceModel(model=lsa_model, corpus=corpus_tfidf, dictionary=dictionary, coherence='u_mass')
        coherence_score = coherence_model.get_coherence()
        
        print(f"  -> Coherence Score for {num_topics} topics: {coherence_score:.4f}")
        
        if coherence_score < best_coherence:
            best_coherence = coherence_score
            best_model = lsa_model
            best_num_topics = num_topics
            
    print(f"\nBest model found with {best_num_topics} topics (Coherence: {best_coherence:.4f})")
    
    print("Saving the best model, dictionary and TF-IDF model...")
    os.makedirs(models_dir, exist_ok=True)
    best_model.save(os.path.join(models_dir, 'lsa_model.gensim'))
    dictionary.save(os.path.join(models_dir, 'lsa_dictionary.gensim'))
    tfidf.save(os.path.join(models_dir, 'lsa_tfidf.gensim'))
    
    print("LSA training completed successfully!")

if __name__ == '__main__':
    PROCESSED_DATA = os.path.join('data', 'processed', 'covid19_tweets_off.csv')
    MODELS_DIR = 'models'
    
    if os.path.exists(PROCESSED_DATA):
        train_lsa(PROCESSED_DATA, MODELS_DIR)
    else:
        print(f"File not found: {PROCESSED_DATA}. Please run make_dataset.py first.")
