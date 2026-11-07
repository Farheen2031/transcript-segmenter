import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('punkt', quiet=True)

def segment_with_statistics(text, window_size=2):
    sentences = nltk.sent_tokenize(text)
    windows = []
    for i in range(len(sentences) - window_size + 1):
        windows.append(" ".join(sentences[i:i+window_size]))
    
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(windows)
    
    similarities = []
    for i in range(len(windows) - 1):
        sim = cosine_similarity(tfidf_matrix[i], tfidf_matrix[i+1])[0][0]
        similarities.append(sim)
    
    threshold = np.mean(similarities) - (0.2 * np.std(similarities))
    boundaries_idx = [i+1 for i, sim in enumerate(similarities) if sim < threshold]
    
    segments = []
    start = 0
    for b in boundaries_idx:
        segments.append(" ".join(sentences[start:b]))
        start = b
    segments.append(" ".join(sentences[start:]))
    
    return segments, similarities

if __name__ == "__main__":
    with open("sample_transcript.txt", "r", encoding="utf-8") as f:
        text = f.read()
    
    segments, sims = segment_with_statistics(text)
    
    print("=== Statistical Segmentation ===")
    print(f"Number of segments: {len(segments)}")
    for i, seg in enumerate(segments):
        print(f"\nSegment {i+1}:\n{seg[:200]}...")