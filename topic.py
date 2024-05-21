import requests
import logging
import random
import json
from typing import List, Tuple, Dict, Any
from datetime import datetime
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import sqlite3
import config
import concurrent.futures

# Initialize logging
logging.basicConfig(level=config.LOGGING_LEVEL, format=config.LOGGING_FORMAT)

# Initialize BERT model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def send_email(subject: str, message: str, message_type: str = 'html') -> None:
    """Send an email with the specified subject and message."""
    payload = {
        'subject': subject,
        'message': message,
        'message_type': message_type
    }
    try:
        response = requests.post(config.EMAIL_URL, headers=config.EMAIL_HEADERS, data=json.dumps(payload))
        response.raise_for_status()
        logging.info('Email sent successfully!')
    except requests.RequestException as e:
        logging.error(f'Failed to send email: {e}')

def generate_dynamic_topic_vector(topic_interests: Dict[str, float]) -> np.ndarray:
    """Generate a topic vector based on user interests."""
    topic_embeddings = model.encode(config.TOPICS)
    user_topic_embeddings = model.encode(list(topic_interests.keys()))

    weights = np.array(list(topic_interests.values()))
    weighted_avg_embedding = np.average(user_topic_embeddings, axis=0, weights=weights)

    norm = np.linalg.norm(weighted_avg_embedding)
    if norm != 0:
        weighted_avg_embedding /= norm

    return weighted_avg_embedding

def fetch_recent_papers(categories: List[str], num_papers: int = 10) -> List[Tuple[str, str, str, str, datetime]]:
    """Fetch recent papers from arXiv."""
    papers = []
    for category in categories:
        url = f"http://export.arxiv.org/rss/{category}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "xml")
            items = soup.find_all("item")[:num_papers]
            papers.extend([
                (item.title.text.strip() if item.title else "No Title",
                 item.summary.text.strip() if item.summary else "No Summary",
                 item.link.text.strip() if item.link else "No Link",
                 category,
                 datetime.strptime(item.pubDate.text.strip(), '%a, %d %b %Y %H:%M:%S %Z') if item.pubDate else datetime.now())
                for item in items
            ])
        except requests.RequestException as e:
            logging.error(f"Failed to fetch papers from ArXiv category {category}: {e}")
    return sorted(papers, key=lambda x: x[4], reverse=True)

def fetch_blogs_from_source(source: str, num_blogs_per_source: int) -> List[Tuple[str, str, str, str, str]]:
    """Fetch blogs from a single source."""
    blogs = []
    try:
        logging.info(f"Fetching blogs from source: {source}")
        response = requests.get(source)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "xml")
        items = soup.find_all("item")[:num_blogs_per_source]
        blogs.extend([
            (item.title.text.strip() if item.title else "No Title",
             item.description.text.strip() if item.description else "No Description",
             item.link.text.strip() if item.link else "No Link",
             source,
             item.pubDate.text.strip() if item.pubDate else "No Date")
            for item in items
        ])
    except requests.RequestException as e:
        logging.error(f"Failed to fetch blogs from {source}: {e}")
    return blogs

def fetch_engineering_blogs(num_blogs: int = 10) -> List[Tuple[str, str, str, str, str]]:
    """Fetch recent engineering blogs using parallel requests."""
    num_sources = len(config.BLOG_SOURCES)
    num_blogs_per_source = max(1, num_blogs // num_sources)

    blogs = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(fetch_blogs_from_source, source, num_blogs_per_source) for source in config.BLOG_SOURCES]
        for future in concurrent.futures.as_completed(futures):
            try:
                blogs.extend(future.result())
            except Exception as e:
                logging.error(f"Error occurred during fetching blogs: {e}")

    return blogs

def save_entries_to_db(entries: List[Tuple[str, str, str, Any, Any]], db_path: str = config.DB_PATH) -> List[Tuple[str, str, str, Any, Any]]:
    """Save new entries to the database."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS entries
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, summary TEXT, link TEXT UNIQUE, source TEXT, pubDate TEXT)''')
    existing_links = set(row[0] for row in c.execute("SELECT link FROM entries"))
    new_entries = [entry for entry in entries if entry[2] not in existing_links]
    c.executemany('INSERT OR IGNORE INTO entries (title, summary, link, source, pubDate) VALUES (?, ?, ?, ?, ?)', new_entries)
    conn.commit()
    conn.close()
    return new_entries

def generate_bert_vectors(entries: List[Tuple[str, str, str, Any, Any]]) -> np.ndarray:
    """Generate BERT vectors for the given entries."""
    corpus = [entry[1] for entry in entries]
    return model.encode(corpus)

def recommend_entries_bert(topic_vector: np.ndarray, entries: List[Tuple[str, str, str, Any, Any]], bert_matrix: np.ndarray, num_recommendations: int = 10) -> List[Tuple[str, str, str, str]]:
    """Recommend entries based on cosine similarity."""
    topic_embedding = np.array([topic_vector])
    similarity_scores = cosine_similarity(topic_embedding, bert_matrix).flatten()
    sorted_indices = similarity_scores.argsort()[::-1]

    # Ensure diversity using Maximal Marginal Relevance (MMR)
    lambda_param = 0.5  # Control diversity
    selected_indices = []
    for _ in range(num_recommendations):
        if len(selected_indices) == 0:
            selected_indices.append(sorted_indices[0])
        else:
            remaining_indices = [idx for idx in sorted_indices if idx not in selected_indices]
            if not remaining_indices:
                break  # No more entries to select
            mmr_scores = [
                lambda_param * similarity_scores[idx] - (1 - lambda_param) * max(cosine_similarity([bert_matrix[idx]], bert_matrix[selected_indices]).flatten())
                for idx in remaining_indices
            ]
            next_idx = remaining_indices[np.argmax(mmr_scores)]
            selected_indices.append(next_idx)
    return [(entries[idx][0], entries[idx][1], entries[idx][2], entries[idx][3]) for idx in selected_indices]

def generate_email_content(recommendations: List[Tuple[str, str, str, str]]) -> str:
    """Generate HTML email content from recommendations."""
    email_message = """
    <html>
    <head>
    <style>
        body { font-family: Arial, sans-serif; background-color: #f4f4f9; padding: 20px; }
        h2 { color: #333333; }
        ul { list-style-type: none; padding: 0; }
        li { background-color: #ffffff; margin: 10px 0; padding: 15px; border-radius: 5px; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1); }
        strong { font-size: 1.1em; }
        a { color: #1a73e8; text-decoration: none; }
        a:hover { text-decoration: underline; }
        .summary { color: #555555; margin-top: 10px; }
        .source { color: #777777; margin-top: 5px; font-style: italic; }
    </style>
    </head>
    <body>
        <h2>Top 10 Recommended Papers and Blogs</h2>
        <ul>
    """
    for idx, entry in enumerate(recommendations, start=1):
        email_message += f"""
        <li>
            <strong>{entry[0]}</strong><br>
            <p class="summary">{entry[1]}</p>
            <a href="{entry[2]}">Read more</a><br>
            <span class="source">Source: {entry[3]}</span>
        </li>
        """
    email_message += """
        </ul>
    </body>
    </html>
    """
    return email_message

def main() -> None:
    """Main function to fetch, process, recommend, and email entries."""
    topic_interests = {
        "artificial intelligence": 1,
        "machine learning": 1,
        "natural language processing": 0.8,
        "computer vision": 0.5,
        "data science": 0.3
    }
    topic_vector = generate_dynamic_topic_vector(topic_interests)

    # Fetch and process papers
    papers = fetch_recent_papers(config.ARXIV_CATEGORIES, num_papers=10)
    new_papers = save_entries_to_db(papers)
    
    if new_papers:
        paper_bert_matrix = generate_bert_vectors(new_papers)
        recommended_papers = recommend_entries_bert(topic_vector, new_papers, paper_bert_matrix, num_recommendations=10)
    else:
        recommended_papers = []

    # Fetch and process blogs
    blogs = fetch_engineering_blogs(num_blogs=10)
    new_blogs = save_entries_to_db(blogs)
    if new_blogs:
        blog_bert_matrix = generate_bert_vectors(new_blogs)
        recommended_blogs = recommend_entries_bert(topic_vector, new_blogs, blog_bert_matrix, num_recommendations=10)
    else:
        recommended_blogs = []

    # Combine and select balanced recommendations
    num_recommendations = 10
    recommendations = recommended_papers[:num_recommendations // 2] + recommended_blogs[:num_recommendations // 2]

    # If we don't have enough from either, fill in the rest
    if len(recommendations) < num_recommendations:
        recommendations += recommended_papers[len(recommendations):num_recommendations]
    if len(recommendations) < num_recommendations:
        recommendations += recommended_blogs[len(recommendations):num_recommendations]

    random.shuffle(recommendations)

    # Ensure we only have 10 recommendations
    recommendations = recommendations[:num_recommendations]

    # Generate and send email with balanced recommendations
    email_subject = "Top 10 Recommended Papers and Blogs"
    email_message = generate_email_content(recommendations)
    send_email(email_subject, email_message)
    logging.info("Email Sent")

if __name__ == "__main__":
    main()
