import pandas as pd
import numpy as np
import os
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel

def show_topics(vectorizer, lda_model, n_words):
    """Retrieves n-most frequent words for each topic"""
    keywords = np.array(vectorizer.get_feature_names_out())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords

def train_lda(processed_data_path, models_dir):
    print("Loading processed data...")
    # Load lemmatized corpus
    df = pd.read_csv(processed_data_path)
    # Drop any NaN values that might have resulted from empty strings after preprocessing
    df = df.dropna(subset=['text'])
    
    print("Vectorizing text (CountVectorizer)...")
    vectorizer = CountVectorizer(analyzer="word",
                                 stop_words='english',
                                 max_features=5000)
    
    vectorized_corpus = vectorizer.fit_transform(df['text'])
    
    print("Training Latent Dirichlet Allocation (LDA) model...")
    lda_model = LatentDirichletAllocation(n_components=15,
                                          learning_method='online',
                                          n_jobs=-1,
                                          random_state=42)
    lda_out = lda_model.fit_transform(vectorized_corpus)
    
    print("Saving the trained model and vectorizer...")
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(lda_model, os.path.join(models_dir, 'lda_model.pkl'))
    joblib.dump(vectorizer, os.path.join(models_dir, 'lda_vectorizer.pkl'))
    
    print("Calculating U-Mass Coherence...")
    # Prepare data for Gensim CoherenceModel
    corpus_for_gensim = df['text'].apply(lambda x: str(x).split())
    dictionary = Dictionary(corpus_for_gensim)
    gensim_corpus = [dictionary.doc2bow(doc) for doc in corpus_for_gensim]
    
    topics = show_topics(vectorizer, lda_model, 15)
    
    cm = CoherenceModel(topics=topics, 
                        corpus=gensim_corpus, 
                        dictionary=dictionary, 
                        coherence='u_mass')
    
    coherence_score = cm.get_coherence()
    print(f"LDA U-Mass Coherence Score: {coherence_score:.4f}")
    
    print("LDA training completed successfully!")

if __name__ == '__main__':
    PROCESSED_DATA = os.path.join('data', 'processed', 'covid19_tweets_off.csv')
    MODELS_DIR = 'models'
    
    if os.path.exists(PROCESSED_DATA):
        train_lda(PROCESSED_DATA, MODELS_DIR)
    else:
        print(f"File not found: {PROCESSED_DATA}. Please run make_dataset.py first.")
