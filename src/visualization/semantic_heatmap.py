import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
from src.utils.file_operations import read_csv  # 如果需要读取CSV的话

def load_topic_data(excel_path: str) -> tuple:
    """
    Load topic labels from Excel file and map to English labels
    
    Args:
        excel_path: Path to the Excel file containing topic labels
        
    Returns:
        Tuple of (Chinese labels, English labels)
    """
    df = pd.read_excel(excel_path, sheet_name="Sheet1")
    valid_df = df[
        df["主题标签"].notna() &
        ~df["主题标签"].isin(["合计", "其他（-1）", "总计"])
    ].reset_index(drop=True)
    
    # Extract Chinese labels and create English mappings
    cn_labels = valid_df["主题标签"].tolist()
    en_mappings = {
        1: "China-Japan-US Affairs",
        2: "China-Japan-US Relations",
        3: "Japanese Economy",
        4: "Government Publicity",
        5: "Summit Meetings",
        6: "Japan-China Summit",
        7: "International Summit",
        8: "Pandemic Countermeasures",
        9: "China-Japan Development",
        10: "Maritime Security",
        11: "Hong Kong Affairs",
        12: "Treaty Security",
        13: "Space Issues",
        14: "China-Japan Friendship",
        15: "Diaoyu Islands Sovereignty",
        16: "China-Japan Exchange",
        17: "Japan-US Cooperation",
        18: "Chinese Military"
    }
    en_labels = [en_mappings[idx] for idx in valid_df["主题序号"].tolist()]
    
    return cn_labels, en_labels

def generate_similarity_matrix(cn_labels: list) -> np.ndarray:
    """
    Generate semantic similarity matrix using multilingual BERT embeddings
    
    Args:
        cn_labels: List of Chinese topic labels
        
    Returns:
        Cosine similarity matrix (numpy array)
    """
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    embeddings = model.encode(cn_labels, convert_to_tensor=False)
    return cosine_similarity(embeddings)

def plot_heatmap(
    similarity_matrix: np.ndarray,
    en_labels: list,
    output_path: str
) -> None:
    """
    Plot and save semantic similarity heatmap
    
    Args:
        similarity_matrix: Cosine similarity matrix
        en_labels: List of English topic labels
        output_path: Path to save the generated heatmap
    """
    plt.figure(figsize=(20, 16))
    ax = sns.heatmap(
        similarity_matrix,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        xticklabels=en_labels,
        yticklabels=en_labels,
        vmin=0,
        vmax=1,
        square=True,
        cbar_kws={
            "label": "Semantic Similarity Score",
            "shrink": 0.85
        },
        annot_kws={"fontsize": 6.5}
    )
    
    # Customize plot appearance
    ax.set_title('Semantic Similarity Heatmap of Topics', fontsize=16, pad=20)
    ax.set_xlabel('Topic Labels', fontsize=14)
    ax.set_ylabel('Topic Labels', fontsize=14)
    
    # Adjust axis labels and layout
    plt.xticks(rotation=90, ha='right', fontsize=8.5)
    plt.yticks(rotation=0, ha='right', fontsize=8.5)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to {output_path}")

def main() -> None:
    """Main execution function"""
    excel_path = "../results/csv/BERTopic_key_word.csv"
    output_path = "../results/plots/semantic_similarity_heatmap.png"
    
    try:
        # Load and preprocess data
        cn_labels, en_labels = load_topic_data(excel_path)
        
        # Generate similarity matrix
        similarity_matrix = generate_similarity_matrix(cn_labels)
        
        # Create heatmap
        plot_heatmap(similarity_matrix, en_labels, output_path)
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main()