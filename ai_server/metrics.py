import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def main():
    st.set_page_config(page_title="Model Evaluation Metrics", layout="centered")
    st.title("Model Evaluation Metrics")
    
    # Metrics values
    metrics = {
        "BLEU Score": 0.1179,
        "ROUGE-1": 0.3071,
        "ROUGE-2": 0.2844,
        "ROUGE-L": 0.3017,
        "Perplexity (PPL)": 4.8845
    }
    
    # Display metrics as cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="BLEU Score", value=f"{metrics['BLEU Score']:.4f}")
    with col2:
        st.metric(label="ROUGE-1", value=f"{metrics['ROUGE-1']:.4f}")
    with col3:
        st.metric(label="ROUGE-2", value=f"{metrics['ROUGE-2']:.4f}")
    
    col4, col5 = st.columns(2)
    with col4:
        st.metric(label="ROUGE-L", value=f"{metrics['ROUGE-L']:.4f}")
    with col5:
        st.metric(label="Perplexity (PPL)", value=f"{metrics['Perplexity (PPL)']:.4f}")
    
    # Convert metrics to DataFrame for plotting
    df = pd.DataFrame(metrics.items(), columns=["Metric", "Score"])
    
    # Bar chart visualization
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(df["Metric"], df["Score"], color=["#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#E91E63"])
    ax.set_xlabel("Score")
    ax.set_title("Evaluation Metrics")
    st.pyplot(fig)
    
if __name__ == "__main__":
    main()
