import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import seaborn as sns
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

st.set_page_config(
    page_title="Sentiment Analysis Application", 
    page_icon="💬",  
)

# Memuat dataset
df = pd.read_csv('./data/indobert_sentiment_predictions.csv')

# Cache model loading to prevent reloading on every interaction
@st.cache_resource
def load_model():
    model_path = './model/indobert_model/'
    tokenizer_path = './model/indobert_tokenizer/'
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # Set device (CPU atau GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    return tokenizer, model, device

# Load model once
tokenizer, model, device = load_model()

stopwords_df = pd.read_csv('./data/stopwords.csv', header=None, names=['stopword'])
list_stopwords = stopwords_df['stopword'].tolist()

df['cleaned_comments'] = df['cleaned_comments'].fillna('').astype(str)

# Menambahkan variabel sesi untuk mengatur landing page dan status About
if 'landing_done' not in st.session_state:
    st.session_state['landing_done'] = False

if 'show_about' not in st.session_state:
    st.session_state['show_about'] = False

# Halaman Landing Page
if not st.session_state['landing_done']:
    st.markdown("<h1>Selamat Datang di Aplikasi Analisis Sentimen💬</h1>", unsafe_allow_html=True)
    st.markdown("<p>Aplikasi ini menggunakan Pemrosesan Bahasa Alami (NLP) untuk menganalisis sentimen dari pengguna, apakah itu positif, negatif, atau netral.</p>", unsafe_allow_html=True)

    # Menampilkan fitur-fitur aplikasi
    st.markdown(""" 
    ### Fitur Utama:
    - **Analisis Sentimen Komentar**: Masukkan teks untuk mendapatkan prediksi sentimen.
    - **Gambaran Umum Analisis**: Visualisasi seperti awan kata (word cloud) dan diagram batang untuk memahami kata-kata yang paling sering digunakan dalam komentar positif, negatif, dan netral dari dataset yang digunakan author.
    - **Kata Paling Umum**: Menampilkan kata-kata yang paling sering muncul berdasarkan sentimen.
                
    ### Sumber Dataset: 
    Dataset ini diambil dari komentar YouTube mengenai berita tentang Prabowo Subianto dan Bahlil. \n
    ### Disclaimer:
    Hasil analisis sentimen tidak sepenuhnya akurat dan mungkin tidak mencerminkan pendapat pengguna secara utuh. Model ini menggunakan teknik Pemrosesan Bahasa Alami (NLP) yang mungkin memiliki keterbatasan dalam memahami konteks dan nuansa bahasa.
                
    **Tekan tombol 'Next' hingga hilang untuk mengakses aplikasi.**
    """)

    # Tombol untuk melanjutkan ke halaman analisis
    if st.button("Next"):
        st.session_state['landing_done'] = True

# Jika landing page sudah dilewati, tampilkan fitur analisis
if st.session_state['landing_done']:
    st.markdown("""<style>
        <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap" rel="stylesheet">
        .subheader { color: #ffff; font-family: 'Poppins', sans-serif; }
        .stDataFrame { background-color: #ffffff; border-radius: 15px; padding: 10px; }
        .stSelectbox div { color: #ffff; }
    </style>""", unsafe_allow_html=True)

    # Sidebar menu untuk navigasi
    menu = st.sidebar.selectbox(
        'Select an Option', 
        ['🔍Sentiment Analysis', '📊Analysis Overview']
    )

    # Add About button to sidebar
    if st.sidebar.button("ℹ️ About"):
        st.session_state['show_about'] = not st.session_state['show_about']

    # Menampilkan panel "About" jika statusnya True
    if st.session_state['show_about']:
        with st.sidebar.expander("About This App", expanded=True):
            st.write("""
            Aplikasi ini menganalisis komentar pengguna untuk menentukan sentimen mereka—apakah positif, negatif, atau netral. 
            Aplikasi ini dibangun menggunakan model Pemrosesan Bahasa Alami (NLP) yang memprediksi sentimen teks berdasarkan isi komentar.

            **Fitur:**
            - **Analisis Sentimen**: Masukkan teks, dan aplikasi akan memprediksi sentimennya.
            - **Gambaran Umum Analisis**: Jelajahi kata-kata paling umum, word cloud, dan kata-kata teratas dalam komentar positif, negatif, dan netral dari dataset yang digunakan author.
            
            **Sumber Dataset**: Dataset ini diambil dari komentar YouTube mengenai berita tentang Prabowo Subianto dan Bahlil.
            """)
            st.markdown("**Author**: Faaiz Akhdan Fatkhi")
            st.markdown("**NIM**: 23115791")

    # Fungsi untuk memprediksi sentimen berdasarkan model IndoBERT
    def predict_sentiment(text, tokenizer, model, device):
        if text:
            # Tokenize input
            inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Get sentiment from logits
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()
            
            # Map to sentiment label (0: positive, 1: neutral, 2: negative)
            sentiment_map = {0: 'positive', 1: 'neutral', 2: 'negative'}
            sentiment = sentiment_map[predicted_class]
            
            # Convert to capitalized format to match CSV
            sentiment = sentiment.capitalize()
            
            # Mengembalikan sentimen
            return sentiment
        return None

    # Fungsi untuk mendapatkan kata-kata teratas dari komentar
    def get_top_n_words(comments_series, n=10):
        all_comments = ' '.join(comments_series.fillna('').astype(str))
        words = [word for word in all_comments.split() if word not in list_stopwords]
        word_count = Counter(words)
        return word_count.most_common(n)

    # Fungsi untuk menghasilkan insight berdasarkan visualisasi yang dipilih
    def generate_insights(sentiment_option, top_words, visualization_type):
        if visualization_type == 'Word Cloud':
            if sentiment_option == 'All':
                insight = "Kata kunci yang sering muncul dalam komentar akan ditampilkan dalam bentuk word cloud. Kata-kata yang lebih besar menunjukkan frekuensi yang lebih tinggi, mengindikasikan bahwa mereka adalah topik sentral yang dibahas di seluruh sentimen."
            elif sentiment_option == 'Positive':
                insight = "Word Cloud ini menyoroti kata-kata yang paling sering digunakan dalam komentar positif. Kata-kata yang lebih besar menunjukkan topik yang sangat disukai oleh pengguna dengan sentimen positif."
            elif sentiment_option == 'Negative':
                insight = "Word Cloud ini menyoroti kata-kata yang paling sering digunakan dalam komentar negatif. Kata-kata yang lebih besar menunjukkan masalah berulang atau area ketidakpuasan."
            elif sentiment_option == 'Neutral':
                insight = "Word Cloud ini menampilkan kata-kata yang paling sering digunakan dalam komentar netral, menunjukkan topik yang memicu reaksi seimbang dari audiens."

        elif visualization_type == 'Bar Chart':
            if sentiment_option == 'All':
                insight = f"Diagram batang ini menggambarkan kata-kata teratas dari semua komentar, dengan kata-kata seperti '{top_words[0][0]}' dan '{top_words[1][0]}' menjadi yang paling menonjol."
            elif sentiment_option == 'Positive':
                insight = f"Pada diagram batang komentar positif, kata-kata seperti '{top_words[0][0]}' dan '{top_words[1][0]}' menonjol, menunjukkan bahwa kata-kata ini terkait dengan pengalaman positif."
            elif sentiment_option == 'Negative':
                insight = f"Pada diagram batang komentar negatif, kata-kata seperti '{top_words[0][0]}' dan '{top_words[1][0]}' muncul dengan frekuensi tinggi, menyoroti kekhawatiran umum atau area kritik."
            elif sentiment_option == 'Neutral':
                insight = f"Diagram batang komentar netral menunjukkan kata-kata seperti '{top_words[0][0]}' dan '{top_words[1][0]}', mengindikasikan topik yang dibahas secara seimbang."

        elif visualization_type == 'Most Common Words':
            if sentiment_option == 'All':
                insight = "Tabel kata-kata paling umum menampilkan frekuensi kata yang digunakan di semua komentar, memberikan gambaran tentang topik yang paling dibahas."
            elif sentiment_option == 'Positive':
                insight = "Tabel kata-kata paling umum untuk komentar positif menyoroti kata-kata yang sering muncul dalam diskusi yang menguntungkan."
            elif sentiment_option == 'Negative':
                insight = "Tabel kata-kata paling umum untuk komentar negatif mengungkapkan kata-kata yang umum dikaitkan dengan kritik atau umpan balik negatif."
            elif sentiment_option == 'Neutral':
                insight = "Tabel kata-kata paling umum untuk komentar netral menunjukkan kata-kata yang digunakan dalam konteks seimbang tanpa konotasi positif atau negatif yang kuat."

        return insight

    if menu == '🔍Sentiment Analysis':
        # Judul Sentiment Analysis
        st.markdown('<h1 class="title">💬 Sentiment Analysis Application</h1>', unsafe_allow_html=True)

        # User Input Text
        st.subheader('User Input Text')
        user_input = st.text_area("Enter a comment for analysis:")

        if st.button('Submit'):
            if user_input:
                predicted_sentiment = predict_sentiment(user_input, tokenizer, model, device)
                
                # Menentukan emoji berdasarkan sentimen
                emoji_map = {'Positive': '😁', 'Negative': '😡', 'Neutral': '😐'}
                emoji = emoji_map.get(predicted_sentiment, '')

                # Display the prediction
                st.subheader('Sentiment Analysis Result')
                st.write(f"Predicted Sentiment for the entered comment: **{predicted_sentiment} {emoji}**")

    elif menu == '📊Analysis Overview':
        # Judul Analysis Overview
        st.markdown('<h1 class="title">📊Analysis Overview of YouTube Comments</h1>', unsafe_allow_html=True)

        # Select sentiment option for analysis
        sentiment_option = st.selectbox("Select Sentiment Option", 
                                         ['All', 'Positive', 'Negative', 'Neutral'])

        # Word Cloud Visualization
        st.subheader('Word Cloud Visualization')

        # Filter comments based on sentiment
        if sentiment_option == 'All':
            comments_to_display = df['cleaned_comments']
        else:
            comments_to_display = df[df['predicted_sentiment'] == sentiment_option.lower()]['cleaned_comments']

        # Kombinasi semua comment menjadi single string
        all_comments = ' '.join(comments_to_display)

        # Generate the word cloud
        wordcloud = WordCloud(width=800, height=400,
                            background_color='white',
                            stopwords=list_stopwords,
                            colormap='viridis',
                            random_state=42).generate(all_comments)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)

        # Menampilkan word cloud insight
        wordcloud_insight = generate_insights(sentiment_option, top_words=None, visualization_type='Word Cloud')
        st.write(wordcloud_insight)

        # Top Words berdasarkan Sentiment
        st.subheader('Top Words by Sentiment')

        # Mendapatkan top words
        top_words = get_top_n_words(comments_to_display)

        # Membuat bar chart
        fig, ax = plt.subplots(figsize=(10, 6))

        # Memplot bar chart
        sns.barplot(x=[x[1] for x in top_words], y=[x[0] for x in top_words], ax=ax)
        ax.set_title(f'Top Words in {sentiment_option} Comments')
        ax.set_xlabel('Count')
        ax.set_ylabel('Words')

        # Menampilkan bar chart
        st.pyplot(fig)

        # Menampilkan insight dari bar chart
        barchart_insight = generate_insights(sentiment_option, top_words=top_words, visualization_type='Bar Chart')
        st.write(barchart_insight)

        # Most Common Words
        st.subheader('Most Common Words')
        filtered_words = [word for word in all_comments.split() if word not in list_stopwords]
        word_count = Counter(filtered_words)
        common_words_df = pd.DataFrame(word_count.most_common(20), columns=['Word', 'Frequency'])

        half = len(common_words_df) // 2
        left_column = common_words_df.iloc[:half]
        right_column = common_words_df.iloc[half:]

        # Menaampilkan tabel kata-kata paling umum ke dalam 2 kolom
        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(left_column)

        with col2:
            st.dataframe(right_column)

        # Menampilkan insight dari tabel kata-kata paling umum
        common_words_insight = generate_insights(sentiment_option, top_words=None, visualization_type='Most Common Words')
        st.write(common_words_insight)
