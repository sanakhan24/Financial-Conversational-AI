import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time
import json

# Load and preprocess the QA data
@st.cache_data
def load_qa_data():
    qa_df = pd.read_csv('merged_file.csv')
    return qa_df

@st.cache_resource
def create_vectorizer():
    return TfidfVectorizer(stop_words='english')

# Load and prepare data
def load_qa_data(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    questions = []
    answers = []
    contexts = []

    for article in data['data']:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                questions.append(qa['question'])
                answers.append(qa['answers'][0]['text'])
                contexts.append(context)

    return questions, answers, contexts


# Main execution with all models
def model_run():
    try:
        # Load data
        questions, answers, contexts = load_qa_data('final_squad_data.json')

        # Prepare models dictionary
        models = {
            'bert': bert_model,
            'roberta': roberta_model,
            'distilbert': distilbert_model,
            'bilstm': bilstm_model
        }

        tokenizers = {
            'bert': bert_tokenizer,
            'roberta': roberta_tokenizer,
            'distilbert': distilbert_tokenizer,
            'bilstm': bilstm_tokenizer
        }

        # Evaluate all models
        metrics, results = evaluate_all_models(
            test_questions,
            test_contexts,
            test_answers,
            models,
            tokenizers
        )

        # Plot results
        plot_metrics_comparison(metrics)

        # Save results
        save_results(metrics, results, 'qa_model_results.json')

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

def save_results(metrics, results, filename):
    """
    Save evaluation results to a file
    """
    output = {
        'metrics': metrics,
        'predictions': results
    }

    with open(filename, 'w') as f:
        json.dump(output, f, indent=4)


def analyze_errors(predictions, references, threshold=0.5):
    """
    Analyze cases where models performed poorly
    """
    from difflib import SequenceMatcher

    error_cases = []

    for idx, (pred, ref) in enumerate(zip(predictions, references)):
        # Calculate similarity ratio
        similarity = SequenceMatcher(None, pred.lower(), ref.lower()).ratio()

        if similarity < threshold:
            error_cases.append({
                'index': idx,
                'prediction': pred,
                'reference': ref,
                'similarity': similarity
            })

    return error_cases

def generate_error_report(error_cases, output_file='error_analysis.txt'):
    """
    Generate a detailed error analysis report
    """
    with open(output_file, 'w') as f:
        f.write("Error Analysis Report\n")
        f.write("===================\n\n")

        for case in error_cases:
            f.write(f"Case {case['index']}:\n")
            f.write(f"Prediction: {case['prediction']}\n")
            f.write(f"Reference: {case['reference']}\n")
            f.write(f"Similarity: {case['similarity']:.4f}\n")
            f.write("-" * 50 + "\n")


def get_most_similar_response(user_question, qa_df, vectorizer, top_k=3):
    question_vectors = vectorizer.fit_transform(qa_df['question'].tolist())
    user_vector = vectorizer.transform([user_question])
    similarities = cosine_similarity(user_vector, question_vectors)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    responses = []
    for idx in top_indices:
        if similarities[idx] > 0.2:
            responses.append({
                'question': qa_df['question'].iloc[idx],
                'answer': qa_df['answer'].iloc[idx],
                'similarity': similarities[idx]
            })
    return responses

# Set page config
st.set_page_config(
    page_title="Loblaws Financial Assistant",
    page_icon="💰",
    layout="wide"
)

# Custom CSS - only modifying input and response text colors
st.markdown("""
<style>
.big-font {
    font-size:30px !important;
    font-weight: bold;
    color: #002B49;
}
.user-container {
    background-color: #e1e9f0;
    padding: 20px;
    border-radius: 10px;
    margin: 10px 0;
}
.bot-container {
    background-color: #f5f5f5;
    padding: 20px;
    border-radius: 10px;
    margin: 10px 0;
}
/* Dark color only for input field */
.stTextInput>div>div>input {
    background-color: white;
    color: #000000 !important;
    font-weight: 500;
}
/* Ensuring question and answer text are dark */
.user-text {
    color: #000000 !important;
    font-weight: 500;
}
.bot-text {
    color: #000000 !important;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Create two columns for layout
col1, col2 = st.columns([2,1])

with col1:
    st.markdown('<p class="big-font">Loblaws Financial Assistant</p>', unsafe_allow_html=True)
    st.markdown("Get instant answers about Loblaws' financial information")
    st.markdown("---")

# Load data and create vectorizer
qa_df = load_qa_data()
vectorizer = create_vectorizer()

with col1:
    user_question = st.text_input(
        "Ask me anything about Loblaws' financial information:",
        key="user_input",
        placeholder="Example: What was the revenue for 2024?"
    )
    
    if user_question:
        time.sleep(3)
        responses = get_most_similar_response(user_question, qa_df, vectorizer)
        
        if responses:
            st.session_state.chat_history.append({
                "user": user_question,
                "bot": responses[0]['answer'],
                "similarity": responses[0]['similarity']
            })
        else:
            st.session_state.chat_history.append({
                "user": user_question,
                "bot": "I'm sorry, I couldn't find a relevant answer to your question.",
                "similarity": 0
            })
    
    st.markdown("### Conversation History")
    for chat in reversed(st.session_state.chat_history):
        # User message with dark text
        st.markdown(
            f'<div class="user-container"><b>You:</b><br><span class="user-text">{chat["user"]}</span></div>', 
            unsafe_allow_html=True
        )
        # Bot response with dark text
        st.markdown(
            f'<div class="bot-container"><b>Assistant:</b><br><span class="bot-text">{chat["bot"]}</span></div>', 
            unsafe_allow_html=True
        )

# Side information
with col2:
    st.markdown("### About")
    st.markdown("""
    This assistant can help you with:
    - Financial metrics
    - Revenue information
    - Profit margins
    - Operating costs
    - Business performance
    - And more!
    """)
    

# Footer
st.markdown("---")
st.markdown("*Powered by Loblaws Financial Data*")