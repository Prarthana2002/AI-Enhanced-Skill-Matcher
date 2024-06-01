import streamlit as st
import pandas as pd
import os
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Set the theme to light
st.set_page_config(layout="wide")

# Function to load resume CSV
def load_resume_data():
    resume_csv_path = "C:\\Users\\prart\\Downloads\\archive 3\\UpdatedResumeDataSet.csv"
    resume_df = pd.read_csv(resume_csv_path)
    return resume_df

# Function to load job description CSV
def load_job_description_data():
    job_description_csv_path = "C:\\Users\\prart\\Downloads\\archive (1) 3\\jobs.csv"
    job_description_df = pd.read_csv(job_description_csv_path)
    return job_description_df

# Function to load resume PDFs folder
def load_resume_pdfs():
    resume_pdfs_folder_path = "C:\\Users\\prart\\Downloads\\Resume-screening-master\\Resume-screening-master\\dataset"
    resume_pdfs = os.listdir(resume_pdfs_folder_path)
    return resume_pdfs

# Load datasets
resume_df = load_resume_data()
job_description_df = load_job_description_data()
resume_pdfs = load_resume_pdfs()

# Function to preprocess text using regex
def preprocess_text(text):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    lemmatizer = nltk.WordNetLemmatizer()
    
    # Remove non-alphanumeric characters
    text = re.sub(r'\W+', ' ', text)
    
    # Tokenize text using regex
    tokens = re.findall(r'\b\w+\b', text.lower())
    
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

# Function to calculate matching skill percentage using TF-IDF
def calculate_matching_skill_percentage(resume_text, job_description_text):
    # Preprocess text
    resume_text = preprocess_text(resume_text)
    job_description_text = preprocess_text(job_description_text)
    
    # Vectorize text using TfidfVectorizer
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, job_description_text])
    
    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(vectors)
    matching_skill_percentage = similarity_matrix[0][1] * 100
    return matching_skill_percentage

# Function to calculate similarity for resume ranking using Gale-Shapely algorithm
def calculate_similarity(job_description, df):
    if 'Resume_str' not in df.columns:
        st.error("The CSV file must contain a column named 'Resume_str' with resume text.")
        return pd.DataFrame()

    # Tokenize job description
    job_tokens = nltk.word_tokenize(job_description.lower())

    # Convert resumes to lowercase
    df['Resume_str'] = df['Resume_str'].str.lower()

    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')

    # Fit and transform resumes
    resume_vectors = vectorizer.fit_transform(df['Resume_str'])

    # Transform job description
    job_vector = vectorizer.transform([job_description])

    # Calculate cosine similarity between job description and resumes
    similarity_scores = cosine_similarity(job_vector, resume_vectors)

    # Add similarity scores to DataFrame
    df['Similarity'] = similarity_scores.flatten()

    # Sort DataFrame by similarity scores
    df.sort_values(by='Similarity', ascending=False, inplace=True)

    return df[['ID', 'Resume_str', 'Similarity']].head(10)

# Streamlit UI
st.title("AI Enhanced Skill Matcher")

# Sidebar options
st.sidebar.header("Options")
selected_option = st.sidebar.selectbox("Select an option", ["Skill Matcher", "Resume Ranking"])

# Personalized Options for Resume Ranking
if selected_option == "Resume Ranking":
    st.sidebar.subheader("Personalized Options")
    location = st.sidebar.text_input("Location", "Enter location")
    keyword_targeting = st.sidebar.text_input("Keyword Targeting", "Enter keywords")

# Skill Matcher option
if selected_option == "Skill Matcher":
    st.header("Skill Matcher")
    
    # Enter Job Description
    st.subheader("Enter Job Description")
    job_description_text = st.text_area("Enter Job Description here")
    
    # Upload Resume PDF
    st.subheader("Upload Resume (PDF)")
    uploaded_resume = st.file_uploader("Upload Resume (PDF)", type="pdf")
    
    # Analyze Button
    if st.button("Analyze"):
        if uploaded_resume is not None and job_description_text != "":
            try:
                resume_text = uploaded_resume.getvalue().decode("utf-8")
            except UnicodeDecodeError:
                resume_text = uploaded_resume.getvalue().decode("latin-1", errors="replace")
            matching_skill_percentage = calculate_matching_skill_percentage(resume_text, job_description_text)
            # Multiply by 100 to convert to percentage without decimal points
            matching_skill_percentage = matching_skill_percentage * 100
            st.success(f"Matching Skill Percentage: {int(matching_skill_percentage)}%")
            # Add a progress bar
            st.progress(int(matching_skill_percentage))
        else:
            st.error("Please upload a resume PDF and enter a job description.")
            
# Resume Ranking option
elif selected_option == "Resume Ranking":
    st.header("Resume Ranking")
    
    # Enter Job Description
    st.subheader("Enter Job Description")
    job_description_text = st.text_area("Enter Job Description here")
    
    # Upload Resumes CSV
    st.subheader("Upload Resume Data (CSV)")
    uploaded_file = st.file_uploader("Upload Resume Data (CSV)", type="csv")
    
    # Analyze Button
    if st.button("Analyze"):
        if uploaded_file is not None and job_description_text != "":
            df = pd.read_csv(uploaded_file)
            st.write(df.head())

            top_matches = calculate_similarity(job_description_text, df)
            if not top_matches.empty:
                st.subheader("Top 10 Matches 🚀")
                st.write(top_matches)
            else:
                st.write("Please ensure the CSV file contains 'ID' and 'Resume_str' columns.")
        else:
            st.error("Please upload resume data (CSV) and enter a job description.")

# Display the loaded datasets
st.subheader("Loaded Datasets")
st.write("Resume Data:")
st.write(resume_df)
st.write("Job Description Data:")
st.write(job_description_df)
st.write("Resume PDFs:")
st.write(resume_pdfs)

