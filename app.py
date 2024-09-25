import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from tqdm import tqdm

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the CSV file
df = pd.read_csv(r"C:\Users\ADMIN\Desktop\Semantic Search\LinkedIn_Jobs_Data_India.csv")

# Initial data inspection
print("Shape:", df.shape)
print("Size:", df.size)
print("Columns:", df.columns)
print("Info:")
print(df.info())
print("Count of null values:\n", df.isnull().sum())

# Drop rows with null values
df_cleaned = df.dropna()

# Count after dropping nulls
print("Count after dropping null values:\n", df_cleaned.isnull().sum())

# Preprocessing function for text
def preprocess_text(text):
    text = str(text).lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

# Function to generate text embeddings
def generate_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

# Extract relevant sentences based on cosine similarity
def extract_relevant_sentences(summary, query_embedding, tokenizer, model, k=2):
    sentences = summary.split(".")
    sentence_embeddings = [generate_embedding(sentence, tokenizer, model) for sentence in sentences]
    similarities = [cosine_similarity(query_embedding.reshape(1, -1), sentence_embedding.reshape(1, -1))[0, 0] for sentence_embedding in sentence_embeddings]
    top_k_indices = np.argsort(similarities)[-k:]
    return [sentences[index] for index in top_k_indices]

# Search for authors and relevant parts
def search_authors_and_relevant_parts(df, query, tokenizer, model, n=3, k=2):
    query_embedding = generate_embedding(query, tokenizer, model)
    df["similarity"] = df["embedding"].apply(lambda x: cosine_similarity(x.reshape(1, -1), query_embedding.reshape(1, -1))[0, 0])
    top_n_results = df.sort_values("similarity", ascending=False).head(n)
    
    for _, row in top_n_results.iterrows():
        relevant_parts = extract_relevant_sentences(row["summary"], query_embedding, tokenizer, model, k)
        print(f"Job Title: {row['title']} (similarity: {row['similarity']:.4f})")
        print(f"Relevant parts: {'.'.join(relevant_parts)}...")

# Load pretrained model and tokenizer
model_name = "sentence-transformers/paraphrase-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Create a summary column with relevant metadata concatenated
df_cleaned['summary'] = df_cleaned.apply(lambda x: f"{x['id']} {x['title']} {x['companyName']} {x['postedTime']} {x['applicationsCount']} {x['description']} {x['contractType']} {x['experienceLevel']} {x['workType']} {x['sector']} {x['companyId']} {x['city']} {x['state']} {x['recently_posted_jobs']}", axis=1)

# Preprocess the summary text
df_cleaned['preprocessed_summary'] = df_cleaned['summary'].apply(preprocess_text)

# Generate embeddings
embeddings = []
for summary in tqdm(df_cleaned['preprocessed_summary'], desc="Generating embeddings"):
    embedding = generate_embedding(summary, tokenizer, model)
    embeddings.append(embedding)

# Save embeddings into the DataFrame
df_cleaned['embedding'] = embeddings
df_cleaned['embedding'] = df_cleaned['embedding'].apply(lambda x: ', '.join(map(str, x[0])))

# Save the DataFrame with embeddings to a CSV file
df_cleaned.to_csv("summaries_with_embeddings.csv", index=False)

# Load the CSV file with embeddings
df_with_embeddings = pd.read_csv("summaries_with_embeddings.csv")

# Convert the embeddings from strings back to numpy arrays
df_with_embeddings["embedding"] = df_with_embeddings["embedding"].apply(lambda x: np.array([float(num) for num in x.split(', ')]))

# Search function with semantic similarity
def search(query, df, tokenizer, model, top_n=5):
    # Preprocess the query
    preprocessed_query = preprocess_text(query)
    
    # Generate the embedding for the query
    query_embedding = generate_embedding(preprocessed_query, tokenizer, model)[0]
    
    # Compute cosine similarity between the query embedding and the job embeddings
    similarities = cosine_similarity([query_embedding], list(df["embedding"]))
    
    # Add similarity scores to the DataFrame
    df["similarity_score"] = similarities[0]
    
    # Get the top N most similar results
    top_n_indices = np.argsort(similarities[0])[-top_n:][::-1]
    top_n_results = df.iloc[top_n_indices]
    
    return top_n_results

# Main loop to interact with the user
while True:
    query = input("Enter your query (or 'exit' to quit): ").strip().lower()
    if query == 'exit':
        print("Exiting...")
        break
    
    if query:
        results = search(query, df_with_embeddings, tokenizer, model)
        if not results.empty:
            print(f"Top results for '{query}':")
            for index, result in results.iterrows():
                print(f"Job Title: {result['title']}")
                print(f"Company: {result['companyName']}")
                print(f"Location: {result['city']}, {result['state']}")
                print(f"Similarity Score: {result['similarity_score']:.4f}")
                print("-------------------------")
        else:
            print("No results found.")
    else:
        print("Please enter a valid query.")
