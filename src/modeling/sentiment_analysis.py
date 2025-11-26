import pandas as pd
from openai import OpenAI
from src.utils.file_operations import read_csv, save_csv
import os

# Initialize OpenAI client
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),  # Use environment variable
    base_url="https://api.deepseek.com"
)

def analyze_sentiment(text):
    """Analyze text sentiment"""
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个情感分析助手，仅返回中文的'积极'、'中性'或'消极'三个词中的一个，不要返回其他任何内容。"},
                {"role": "user", "content": f"请分析以下日文文本的情感倾向：{text}"}
            ],
            stream=False,
            temperature=0.2,
            max_tokens=10
        )
        
        sentiment = response.choices[0].message.content.strip()
        return sentiment if sentiment in ["积极", "中性", "消极"] else "中性"
    except Exception as e:
        print(f"Error analyzing sentiment: {str(e)}")
        return "中性"

def process_data(csv_path):
    """Process data"""
    df = read_csv(csv_path)
    texts = df.iloc[:, 0].tolist()
    topics = df.iloc[:, 2].astype(str).tolist()
    
    # Filter invalid topics
    valid_data = [(t, tp) for t, tp in zip(texts, topics) if tp != '-1']
    return valid_data

def analyze_and_save_results(data, output_path):
    """Analyze sentiments and save results"""
    stats = {}
    for text, topic in data:
        sentiment = analyze_sentiment(text)
        if topic not in stats:
            stats[topic] = {"积极":0, "中性":0, "消极":0}
        stats[topic][sentiment] +=1
    
    save_csv(pd.DataFrame(stats).T, output_path)
    print(f"Sentiment analysis saved to {output_path}")

def main():
    """Main sentiment analysis workflow"""
    data = process_data("results/csv/topicdocs.csv")
    analyze_and_save_results(data, "results/csv/sentiment_stats.csv")

if __name__ == "__main__":
    main()