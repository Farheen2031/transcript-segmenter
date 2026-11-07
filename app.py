import streamlit as st
from texttiling_segmenter import segment_with_texttiling
from statistical_segmenter import segment_with_statistics

# Page setup
st.set_page_config(page_title="Transcript Segmenter", page_icon="📝")
st.title("📝 Lecture & Meeting Transcript Segmenter")
st.write("Paste a transcript below to automatically divide it into topics.")

# Text input box
text_input = st.text_area("Paste Transcript Here:", height=200, 
    placeholder="Welcome to today's lecture on artificial intelligence...")

# Dropdown to select method
method = st.selectbox("Select Segmentation Method:", 
    ["Statistical Method (TF-IDF)", "TextTiling (NLTK)"])

# Button to run
if st.button("Segment Transcript", type="primary"):
    if text_input.strip() == "":
        st.warning("Please paste some text first!")
    else:
        with st.spinner("Analyzing topics..."):
            if "Statistical" in method:
                segments, sims = segment_with_statistics(text_input)
            else:
                segments = segment_with_texttiling(text_input)
            
            st.success(f"Success! Found {len(segments)} segments.")
            
            # Display the results beautifully
            for i, seg in enumerate(segments):
                with st.expander(f"Topic {i+1}"):
                    st.write(seg)