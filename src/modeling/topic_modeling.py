import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from src.utils.file_operations import read_csv, save_csv

def load_processed_data(input_path):
    """Load preprocessed data"""
    with open(input_path, 'r', encoding='utf-8') as f:
        return f.read().split('\n')

def create_vectorizer():
    """Create custom vectorizer"""
    return CountVectorizer(
        ngram_range=(1, 2),
        stop_words=None,  # Using custom filtering in BERTopic
        tokenizer=lambda x: x.split()  # Direct split after preprocessing
    )

def load_embedding_model(model_path):
    """Load embedding model"""
    return pipeline(
        'feature-extraction',
        model=model_path
    )

def train_topic_model(docs, embeddings, output_path):
    """Train BERTopic model"""
    model = BERTopic(
        language="japanese",
        embedding_model=embeddings,
        vectorizer_model=create_vectorizer(),
        verbose=True
    )
    topics, probabilities = model.fit_transform(docs)
    
    # Save results
    save_csv(model.get_document_info(docs), output_path)
    print(f"\nDocument topics saved to {output_path}")
    
    return model

def main():
    """Main topic modeling workflow"""
    processed_docs = load_processed_data("data/processed/data_new.txt")
    embeddings = load_embedding_model("models/bertopic_japanese_v3")
    
    # Train model
    model = train_topic_model(processed_docs, embeddings, "results/csv/topicdocs.csv")
    
    # Get topic info
    topic_freq = model.get_topic_freq()
    save_csv(topic_freq, "results/csv/topic_frequency.csv")
    
    # Get all topics
    all_topics = model.get_topics()
    save_csv(pd.DataFrame(all_topics), "results/csv/all_topics.csv")

if __name__ == "__main__":
    main()