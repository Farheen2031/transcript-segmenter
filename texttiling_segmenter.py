import nltk
nltk.download('stopwords', quiet=True)

from nltk.tokenize import TextTilingTokenizer

def segment_with_texttiling(text):
    tt = TextTilingTokenizer(w=5, k=3)
    result = tt.tokenize(text)
    
    # This handles the list error perfectly
    if isinstance(result, list):
        segments = result
    else:
        segments = result.split("\n\n\n")
        
    return segments

if __name__ == "__main__":
    print("Starting TextTiling script...\n")
    
    with open("sample_transcript.txt", "r", encoding="utf-8") as f:
        text = f.read()
    
    segments = segment_with_texttiling(text)
    
    print("=== TextTiling Segmentation ===")
    print(f"Number of segments found: {len(segments)}")
    print("\n--- Segmented Topics ---")
    
    for i, seg in enumerate(segments):
        print(f"\nTopic {i+1}:\n{seg.strip()[:200]}...")
