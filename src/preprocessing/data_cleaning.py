import os
import re
import MeCab
from bertopic import BERTopic
from transformers import pipeline
from sklearn.feature_extraction.text import CountVectorizer

# Custom tokenizer filtering words containing digits
def custom_tokenizer(text: str) -> list:
    """Tokenize text while filtering words containing digits"""
    return re.findall(r'\b[a-zA-Zぁ-んァ-ン一-龥]+\b', text)

# Initialize MeCab tokenizer with dictionary path
def initialize_mecab(dic_directory: str) -> MeCab.Tagger:
    """Initialize MeCab tagger with specified dictionary"""
    try:
        tagger = MeCab.Tagger(f"--rcfile /dev/null --dicdir {dic_directory}")
        test_node = tagger.parseToNode("Test")
        if test_node.surface == "Test":
            print("MeCab initialized successfully")
            return tagger
        else:
            raise RuntimeError("MeCab initialization failed")
    except Exception as e:
        print(f"Error initializing MeCab: {str(e)}")
        return None

# Load raw documents from file
def load_documents(file_path: str) -> list:
    """Load documents from text file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []

# Process stopwords
def process_stopwords(stopwords_file: str) -> set:
    """Process stopwords file"""
    stopwords = set()
    try:
        with open(stopwords_file, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word and not re.search(r'\d', word):
                    stopwords.add(word)
        print(f"Loaded {len(stopwords)} stopwords")
        return stopwords
    except FileNotFoundError:
        print(f"Warning: Stopwords file not found at {stopwords_file}")
        return set()

# Preprocess individual document
def preprocess_document(doc: str, mecab_tagger: MeCab.Tagger, stopwords: set) -> str:
    """Preprocess single document"""
    if not doc:
        return ""
    
    words = []
    if mecab_tagger:
        node = mecab_tagger.parseToNode(doc)
        while node:
            surface = node.surface
            if surface and not re.search(r'\d', surface) and surface not in stopwords:
                words.append(surface)
            node = node.next
    else:
        # Fallback tokenizer (simple whitespace split)
        words = re.findall(r'\b[a-zA-Zぁ-んァ-ン一-龥]+\b', doc)
        words = [w for w in words if not re.search(r'\d', w)]
    
    return ' '.join(words)

# Main processing pipeline
def main():
    # Configure base directories
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_RAW = os.path.join(BASE_DIR, "data/raw/data.txt")
    STOPWORDS_FILE = os.path.join(BASE_DIR, "data/processed/stopwords.txt")
    PROCESSED_DATA = os.path.join(BASE_DIR, "data/processed/data_new.txt")
    MODEL_PATH = os.path.join(BASE_DIR, "models/bertopic_japanese_v3")
    OUTPUT_CSV = os.path.join(BASE_DIR, "results/csv/topicdocs.csv")
    
    # Create necessary directories
    os.makedirs(os.path.dirname(PROCESSED_DATA), exist_ok=True)
    
    # Load raw data
    raw_documents = load_documents(DATA_RAW)
    print(f"Loaded {len(raw_documents)} raw documents")
    
    # Initialize MeCab
    mecab = initialize_mecab(os.path.join(MODEL_PATH, "dic/ipadic"))
    
    # Load stopwords
    stopwords = process_stopwords(STOPWORDS_FILE)
    
    # Preprocess documents
    processed_docs = []
    for idx, doc in enumerate(raw_documents, 1):
        if idx % 1000 == 0:
            print(f"Processing document {idx}/{len(raw_documents)}")
        processed_doc = preprocess_document(doc, mecab, stopwords)
        processed_docs.append(processed_doc)
    
    # Save processed data
    with open(PROCESSED_DATA, 'w', encoding='utf-8') as f:
        f.write('\n'.join(processed_docs))
    print(f"Processed data saved to {PROCESSED_DATA}")
    
    # Prepare vectorizer
    vectorizer = CountVectorizer(
        ngram_range=(1, 2),
        stop_words=stopwords,
        tokenizer=lambda x: x.split()
    )
    
    # Load embedding model
    embedding_model = pipeline(
        'feature-extraction',
        model=MODEL_PATH
    )
    
    # Train BERTopic model
    model = BERTopic(
        language="japanese",
        embedding_model=embedding_model,
        vectorizer_model=vectorizer,
        verbose=True
    )
    
    # Process documents
    docs = [doc for doc in processed_docs if doc]
    topics, probabilities = model.fit_transform(docs)
    
    # Save results
    model.save(f"{BASE_DIR}/results/models/topic_model")
    print(f"Model saved to {BASE_DIR}/results/models/topic_model")
    
    # Output statistics
    print("\nTopic Frequency Distribution:")
    print(model.get_topic_freq())
    
    print("\nSample Document Topics:")
    for doc_id, (topic, prob) in enumerate(zip(topics, probabilities), 1):
        if doc_id > 10:
            break
        print(f"Document {doc_id}: Topic {topic} (Probability: {prob:.3f})")
    
    # Save document info
    model.get_document_info(docs).to_csv(OUTPUT_CSV, encoding='utf-8-sig')
    print(f"Document info saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()