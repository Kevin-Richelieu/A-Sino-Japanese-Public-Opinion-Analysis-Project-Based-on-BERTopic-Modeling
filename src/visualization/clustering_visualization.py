import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.cluster.hierarchy import linkage, dendrogram

topics = [
    "China-Japan-US Affairs",
    "China-Japan-US Relations",
    "Japanese Economy",
    "Government Publicity",
    "Summit Meetings",
    "Japan-China Summit",
    "International Summit",
    "Pandemic Countermeasures",
    "China-Japan Development",
    "Maritime Security",
    "Hong Kong Affairs",
    "Treaty Security",
    "Space Issues",
    "China-Japan Friendship",
    "Diaoyu Islands Sovereignty",
    "China-Japan Exchange",
    "Japan-US Cooperation",
    "Chinese Military"
]
topic_labels = [f"{i+1}: {topic}" for i, topic in enumerate(topics)]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(topics)
X = tfidf_matrix.toarray()  


linked = linkage(X, method='ward')


plt.figure(figsize=(12, 8))
dendrogram(
    linked,
    orientation='right',  
    labels=topic_labels,       
    distance_sort='descending',  
    show_leaf_counts=True,       
    leaf_font_size=10            
)
#plt.title('Hierarchical Clustering Dendrogram of Topics')
plt.xlabel('Cluster Distance')
plt.ylabel('Topics')
plt.tight_layout()
plt.savefig('../results/plots/cluster_dendrogram.png', dpi=300, bbox_inches='tight')
plt.show()