import pymongo
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Connect to MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["search_engine"]
terms_collection = db["index_terms"]
documents_collection = db["index_documents"]

# Sample documents
documents = [
    "The doctor prescribed medication for the patient with dizziness and headache.",
    "Medication caused nausea, and the patient reported discomfort with dizziness.",
    "Nausea and headache are side effects commonly caused by medication.",
    "The dizziness was due to the medication, but no headache was reported."
]

# Text preprocessing: remove punctuation, convert to lowercase
def preprocess_text(text):
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    return text.lower()  # Convert to lowercase

# Tokenizer for unigrams, bigrams, trigrams
def generate_tokens(text):
    words = text.split()
    unigrams = words
    bigrams = [" ".join(words[i:i+2]) for i in range(len(words)-1)]
    trigrams = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
    return unigrams + bigrams + trigrams

# Build the inverted index
def build_index():
    terms_collection.delete_many({})
    documents_collection.delete_many({})

    vectorizer = TfidfVectorizer()
    processed_docs = [preprocess_text(doc) for doc in documents]
    tfidf_matrix = vectorizer.fit_transform(processed_docs)
    feature_terms = vectorizer.get_feature_names_out()
    vocab = {term: idx for idx, term in enumerate(feature_terms)}

    # Process each document and build the index
    for doc_id, content in enumerate(documents, start=1):
        processed_content = preprocess_text(content)
        tokens = generate_tokens(processed_content)

        # Add document to MongoDB
        documents_collection.insert_one({"_id": doc_id, "content": content})

        # Count token positions
        term_positions = {}
        for position, token in enumerate(tokens):
            if token not in term_positions:
                term_positions[token] = []
            term_positions[token].append(position)

        # Populate the inverted index
        for term, positions in term_positions.items():
            tfidf_value = (
                tfidf_matrix[doc_id-1, vocab.get(term, -1)]
                if term in vocab else 0
            )
            if tfidf_value > 0:
                term_entry = terms_collection.find_one({"term": term})
                if term_entry:
                    term_entry["docs"].append({"doc_id": doc_id, "positions": positions, "tfidf": tfidf_value})
                    terms_collection.replace_one({"term": term}, term_entry)
                else:
                    terms_collection.insert_one({
                        "_id": vocab[term],
                        "term": term,
                        "pos": vocab[term],
                        "docs": [{"doc_id": doc_id, "positions": positions, "tfidf": tfidf_value}]
                    })

# Query the inverted index and rank results
def query_index(queries):
    vectorizer = TfidfVectorizer()
    processed_docs = [preprocess_text(doc) for doc in documents]
    tfidf_matrix = vectorizer.fit_transform(processed_docs)

    for query_id, query_text in enumerate(queries, start=1):
        processed_query = preprocess_text(query_text)
        query_vector = vectorizer.transform([processed_query])

        # Calculate similarity
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

        # Rank documents
        ranked_results = sorted(
            [(doc, score) for doc, score in zip(documents, similarities) if score > 0],
            key=lambda x: x[1],
            reverse=True
        )

        # Print results
        print(f"Query {query_id}: {query_text}")
        for doc, score in ranked_results:
            print(f"  Document: \"{doc}\", Score: {score:.4f}")
        print()

# Main execution
if __name__ == "__main__":
    build_index()

    # Define queries
    queries = [
        "dizziness and headache",
        "nausea caused",
        "side effects",
        "reported headache",
        "medication"
    ]

    query_index(queries)
