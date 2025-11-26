# A Sino Japanese Public Opinion Analysis Project Based on BERTopic Modeling

## Project Structure

```bash
project_root/
│
├── mecab-ipadic-neologd
│
├── data/
│   ├── raw/
│   │   └── data.txt
│   ├── processed/
│   │   ├── stopwords.txt
│   │   └── data_new.txt
│   └── dataset/
│
├── models/
│   └── bertopic_japanese_v3/
│
├── results/
│   ├── plots/
│   │   ├── topic_frequency_time_trend.png
│   │   ├── clustered_topic_trend.png
│   │   └── semantic_similarity_heatmap.png
│   └── csv/
│       ├── topicdocs.csv
│       ├── topic_frequency.csv
│       └── BERTopic_key_word.csv
│
├── src/
│   ├── preprocessing/
│   │   └── data_cleaning.py
│   ├── modeling/
│   │   ├── topic_modeling.py
│   │   └── sentiment_analysis.py
│   ├── visualization/
│   │   ├── trend_visualization.py
│   │   └── clustering_visualization.py
│   └── utils/
│       └── file_operations.py
│
├── requirements.txt
└── README.md
```

## Getting Started
1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Set environment variables:
```bash
export OPENAI_API_KEY="your_api_key_here"
```
3. Data Preprocessing:
```bash
python src/preprocessing/data_cleaning.py
```
4. Topic Modeling:
```bash
python src/modeling/topic_modeling.py
```
5. Sentiment Analysis:
```bash
python src/modeling/sentiment_analysis.py
```
6. Visualization:
```bash
python src/visualization/trend_visualization.py
python src/visualization/clustering_visualization.py
python src/visualization/semantic_heatmap.py
```

## Data Requirements
- Raw data should be placed in `data/raw/data.txt`
- Stopwords list should be placed in `data/processed/stopwords.txt`

## Model Requirements
Pre-trained BERTopic model should be placed in `models/bertopic_japanese_v3`

