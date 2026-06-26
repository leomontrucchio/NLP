# Topic Modeling on COVID-19 Tweets

Comparison of three topic modeling approaches — **LDA**, **LSA**, and **Supervised LDA** — applied to two COVID-19 tweet datasets. The first dataset contains general COVID-related tweets; the second contains tweets labeled as suspicious (potentially hateful or negative) vs. neutral.

## Project Structure

```
NLP/
├── data/
│   ├── raw/                  # Original datasets
│   └── processed/            # Cleaned and lemmatized data
├── src/
│   ├── data/
│   │   └── make_dataset.py   # Text cleaning, translation, and lemmatization
│   └── models/
│       ├── lda_model.py      # LDA training (sklearn + gensim coherence)
│       ├── lsa_model.py      # LSA training (gensim TF-IDF + LsiModel)
│       └── slda_model.py     # Supervised LDA training (tomotopy)
├── models/                   # Saved model weights (.pkl, .gensim, .bin)
├── notebooks/
│   ├── final_results.ipynb                    # Loads trained models and visualizes results
│   ├── exploratory_analysis_covid_tweets.ipynb # EDA on the COVID tweets dataset
│   └── exploratory_analysis_sus_tweets.ipynb   # EDA on the suspicious tweets dataset
├── requirements.txt
└── .gitignore
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Usage

Train the models:

```bash
python src/models/lda_model.py
python src/models/lsa_model.py
python src/models/slda_model.py
```

Then open `notebooks/final_results.ipynb` to explore the results.
