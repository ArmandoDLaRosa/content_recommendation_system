# Personalized Article Recommendation System

## Overview

The Personalized Article Recommendation System is designed to revolutionize the reading experience by offering a singular, highly relevant article recommendation daily. Tailored to individual interests in specialized fields such as Distributed Systems, AI Scaling, Low-Level Engineering, and System Design, this system leverages advanced Natural Language Processing (NLP), dynamic feedback loops, and Retrieval-Augmented Generation (RAG) technologies. Our goal is to provide unparalleled personalization in content recommendation and discovery. By having Topic Vectors, it's able to search and discover new articles in the predefined sources that match those keywords mixed with niche words selected by the user.

## Key Features

- **Singular Daily Article Recommendation**: Delivers one article per day, chosen for its relevance and alignment with the user's interests by pondering the most recent articles from all sources and delivering only the most relevant towards the interests. We make sure it hasn't been read by the user and it's as recent as possible in the origin source.
- **Similar Article Discovery**: Offers a selection of related articles to encourage exploration of topics in depth. Sortable by Most Recent.
- **RAG-Powered Search**: Allows for natural language queries, enhancing the discovery of articles that match user interests closely. Sortable by Most Recent.
- **User Feedback Integration**: Utilizes direct user inputs and article rankings to continuously refine the recommendation process.

## Idea of the App

The Personalized Article Recommendation System aims to enhance user engagement and learning by providing tailored content recommendations based on individual interests. By analyzing user preferences and search behaviors, the system identifies relevant articles from various sources and delivers a daily recommendation that aligns with the user's specific fields of interest. This approach ensures users receive the most pertinent and recent articles, promoting continuous learning and exploration.

## Implementation Overview

### System Architecture

The architecture integrates various technologies, including Topic Model Vectors, Semantic Relevance Analysis, Text Similarity Clusters, a Vector Database (VectorDB), Retrieval-Augmented Generation (RAG), User Feedback Mechanisms, and Knowledge Graphs. These components work together to facilitate personalized article recommendations and enable effective content discovery.

### Technologies and Methodologies

#### Content Categorization
- **Purpose**: To accurately tag articles with relevant topics, improving search and recommendation accuracy.
- **Implementation**: Use of NLP models like BERT for extracting and categorizing article topics, ensuring precise association with user interests.

#### Semantic Relevance & Topic Model Vectors
- **Purpose**: To assess the relevance of articles to a user's specific interests, aiding the daily recommendation process.
- **Implementation**: Generation of article embeddings with transformer models, stored in VectorDB for efficient similarity-based retrieval.

#### Text Similarity Clusters
- **Purpose**: To identify semantically related groups of articles, enriching the similar article discovery feature.
- **Implementation**: Application of clustering algorithms on article embeddings to suggest related content, enhancing the exploration experience.

#### Knowledge Graphs
- **Purpose**: To leverage rich contextual relationships between entities (articles, authors, topics) for enhanced recommendations.
- **Implementation**: Construction and traversal of a knowledge graph using libraries like `networkx`, integrating domain-specific knowledge and ensuring diverse and novel recommendations.

#### VectorDB
- **Purpose**: To store and query vector embeddings of articles, serving as the system's data retrieval backbone.
- **Implementation**: Deployment of a vector database capable of fast nearest neighbor searches (e.g., Elasticsearch or Milvus), facilitating real-time, similarity-based article retrieval.

#### Retrieval-Augmented Generation (RAG)
- **Purpose**: To improve natural language search capabilities, enabling precise article discovery through user queries.
- **Implementation**: Integration of RAG for query generation and similar article recommendations, combining retrieved information with generative modeling for nuanced understanding and response.

#### User Feedback Integration
- **Purpose**: To refine the recommendation engine based on user interactions and feedback, ensuring continuous improvement.
- **Implementation**: Development of a feedback loop that adjusts recommendation algorithms based on user ratings and inputs.

### Workflow and Data Flow

1. **Article Processing**: Automatic preprocessing, categorization, and embedding generation for each article, followed by storage in VectorDB.
2. **Daily Recommendation**: Utilization of semantic relevance, knowledge graphs, and user interest vectors to select the most pertinent article for each user daily.
3. **Similar Article Discovery**: Provision of related articles through text similarity analysis, knowledge graph traversal, and RAG-powered recommendations, fostering content exploration.
4. **User Feedback Collection**: Incorporation of user feedback to fine-tune the recommendation process, enhancing personalization and relevance.

## Getting Started

This section should include detailed instructions on system setup, including environment configuration, dependencies installation, and initial data ingestion. Provide steps for starting the recommendation system, ensuring users can begin receiving personalized content recommendations efficiently.

## Extended Article Retention and Discovery

### Overview

In addition to providing a singular, highly relevant article recommendation each day, our system introduces an advanced feature for storing and discovering multiple articles. This capability allows users to engage with a curated selection of three articles daily, chosen based on their unique interests and search queries. This feature is designed to enrich the user's exploration experience, offering them the opportunity to dive deeper into their areas of interest.

### Implementation

#### Storing Multiple Articles
- **Purpose**: To augment the daily reading experience by providing users with additional articles that align closely with their interests, encouraging further exploration and engagement.
- **Implementation**: Every day, alongside the primary article recommendation, the system automatically stores a selection of three additional articles. This selection is derived from sophisticated analysis of user-generated search queries, incorporating a blend of main topic interests and niche keywords. These articles are then stored in a dedicated section within the user's profile, allowing for easy access and exploration at their convenience.

#### Technology Integration
- **Search Query Analysis**: Utilizes advanced NLP techniques to analyze search queries, extracting main topic interests and niche keywords. This analysis guides the selection of additional articles, ensuring they are highly relevant to the user's expressed interests.
- **Article Selection Algorithm**: Employs a combination of user interest profiles, past interactions, and semantic relevance evaluations to curate a list of three articles daily. This algorithm prioritizes content that not only matches the user's main interests but also introduces them to related topics and ideas.
- **User Interface Design**: The dedicated section for stored articles is designed with user experience in mind, ensuring easy access and navigation. This allows users to seamlessly transition between their daily recommended article and their curated list of additional reads.

### Benefits

- **Enhanced User Engagement**: By offering users more than just a singular daily recommendation, the system fosters a deeper level of engagement and interaction with the content.
- **Personalized Exploration**: The curated selection of additional articles allows users to explore related topics in depth, facilitating a personalized exploration experience that goes beyond their initial queries.
- **Continuous Learning**: This feature supports users in their quest for knowledge, providing them with multiple perspectives and insights on topics of interest. It encourages continuous learning and discovery within their preferred domains.

### Getting Started with Extended Article Discovery

- **Initial Setup**: Detailed instructions on how to enable and configure the extended article retention feature within the system.
- **User Guide**: A step-by-step guide for users on how to access and utilize the dedicated section for additional article exploration. Tips on how to make the most of this feature for personalized content discovery.

## System Architecture

```plaintext
[Content Sources] (e.g., arXiv, Medium)
       |
       | Articles fetched and processed
       v
[Content Processing Pipeline] (NLP Models for Categorization & Embedding)
       |
       | Embeddings and metadata
       v
         [Vector Database (VectorDB)]-----------------+
       |                   ^                         |
       |                   | Similarity Searches     |
       |                   | and Retrieval           |
       v                   v                         |
[Knowledge Graph]---->[Retrieval-Augmented Generation (RAG)]------>[Recommendation Engine]
       |                   ^                         |
       |                   |                         |
       | User Queries      | Feedback Loop           | Recommendations
       |                   |                         |
       v                   v                         v
                      [User Interface]
                        (Frontend)
```

## Data Flow Diagram

```plaintext
                             [Fetch Articles]
                                  |
                                  | (Articles for processing)
                                  v
                            [Preprocess Articles]
                                  |
                                  | (Cleaned and formatted articles)
                                  v
                       [Generate Embeddings (NLP)]
                                  |
      +---------------------------+---------------------------+
      |                           |                           |
      |                           |                           |
      v                           v                           v
[Store Embeddings]        [Identify Similar]             [User Feedback]
  (in VectorDB)                 (Articles)                  (on Articles)
      |                           |                           |
      |                           |                           |
      |      +--------------------+                           |
      |      |                                                 |
      +------>              [RAG for Search] <----------------+
             |                     |
             +----> [Recommendation Engine] <---+
                               |
                               | (Personalized recommendations)
                               v
                          [User Interface]
```

## Component Interaction Diagram

```plaintext
            +------------------------------------------------+
            |                    [User Interface]            |
            |                        (FE)                    |
            |                                                |
+-----------v----------+                         +-----------v-----------+
|   [Backend (BE)]     | <------ APIs ------>    | [RAG & ML Models]     |
| - API management     |                         | - Embedding generation|
| - Auth & session mgmt|                         | - Similarity matching |
| - Feedback processing|                         | - Recommendation algo |
+-----------^----------+                         +-----------

^-----------+
            |                                                     |
            |                                                     |
            |                +---------------------+              |
            +---------------->  [Vector Database]  <-------------+
                             |       (DE)          |
                             | - Article storage   |
                             | - Embedding storage |
                             | - Query processing  |
                             +---------------------+
```

## How Each Section Works?

### Article Processing

```plaintext
                      [Fetch Articles]
                           |
                           v
                 [Preprocessing Pipeline]
                /       |              \
               v        v               v
       [Tokenization] [Stopword] [Lemmatization]
               \        |              /
                v       v             v
            [Combine Preprocessed Text]
                           |
                           v
               [Generate Embeddings (BERT)]
                           |
                           v
                 [Store in VectorDB]
```

### Recommendation Generation Process

```plaintext
            [Start Recommendation Process]
                           |
                           v
            [User Interest Profile Vector]
                           |
             +-------------+--------------+
             |                            |
             v                            v
    [Fetch Candidate Articles]    [Check User History]
             |                            |
             +-------------+--------------+
                           |
                           v
          [Compute Similarity Scores with RAG]
                           |
                           v
                [Rank Articles Based on Score]
                           |
                           v
                [Select Top-N Recommendations]
                           |
                           v
                  [Present to User via UI]
```

### User Feedback Loop

```plaintext
              [User Interacts with Article]
                           |
                           v
                  [Collect User Feedback]
                           |
            +--------------+--------------+
            |                             |
            v                             v
    [Adjust User Profile]         [Re-rank Similar Articles]
            |                             |
            v                             v
    [Update Recommendation]       [Store Feedback for Future Learning]
            |                             |
            +--------------+--------------+
                           |
                           v
             [Refine Future Recommendations]
```

### Full System

```plaintext
                              +---------------------+
                              |   Enhanced Big      |
                              |      Texts          |
                              +----------+----------+
                                         |
                                         | Embedding Generation (BERT)
                                         v
                    +--------------------+--------------------+
                    |              Vector Database            |
                    | (Stores embeddings & article metadata)  |
                    +--------+----------------------+---------+
                             |                      |
         +-------------------+                      +
         |                                          |                  
 +-------v-------+                          +-------v-------+    +-----+----+
 |  Search Query |                          |  Fetch & Rank |    | User     |
 |  Construction +------------------------->+  Articles     +--->+ Ranking  |
 |  (with RAG)   |    (arXiv, Medium, etc.) |               |    | (1 to 5) |
 |               |                          +-------+-------+    +-----+----+
 +-------+-------+                                  |                  |
         |            Similar Articles              | Similar articles |
         |            with RAG                      | with RAG         |
         v                                          v                  |
 +-------+--------+                         +-------+--------+         |
 | Natural        |                         | Article-based  |<--------+
 | Language       |                         | Recommendations|
 | Search         |                         +----------------+
 +----------------+
```

## MVP - Single Article Recommendation

### MAIN TOPIC VECTOR - Distributed Systems

Distributed Systems encompass a broad array of technologies and concepts focused on managing and operating multiple interlinked computing nodes that work together to perform complex tasks. These systems are characterized by their scalability, reliability, and efficiency. Key aspects include:

- **Scalability**: Ability to handle increasing workloads by adding resources either horizontally (more nodes) or vertically (more powerful nodes).
- **Concurrency**: Managing multiple parts of a computation simultaneously to optimize processing speed.
- **Fault Tolerance**: Ensuring the system continues to operate even if one or more nodes fail.
- **Load Balancing**: Distributing work evenly across all nodes to prevent any single node from being overwhelmed, which ensures smooth operation and optimal resource utilization.
- **Cloud Computing**: Utilizing online computational resources that can be scaled and shared among users, often encompassing distributed databases and applications.
- **Decentralization**: Removing a central point of control or failure by distributing data and tasks across many locations.
- **Synchronization**: Managing the timing of processes so that operations across nodes are coordinated.
- **Data Replication**: Duplicating data across different nodes to increase reliability and accessibility.

This topic vector encompasses the primary elements and challenges associated with designing, implementing, and managing distributed systems, targeting advancements and solutions that address these core areas.

### NICHE WORDS - Construction

Distributed Systems encompass a broad array of technologies and concepts focused on managing and operating multiple interlinked computing nodes that work together to perform complex tasks. These systems are characterized by their scalability, reliability, and efficiency. Key aspects include:

- **Scalability**: Ability to handle increasing workloads by adding resources either horizontally (more nodes) or vertically (more powerful nodes).
- **Concurrency**: Managing multiple parts of a computation simultaneously to optimize processing speed.
- **Fault Tolerance**: Ensuring the system continues to operate even if one or more nodes fail.
- **Load Balancing**: Distributing work evenly across all nodes to prevent any single node from being overwhelmed, which ensures smooth operation and optimal resource utilization.
- **Cloud Computing**: Utilizing online computational resources that can be scaled and shared among users, often encompassing distributed databases and applications.
- **Decentralization**: Removing a central point of control or failure by distributing data and tasks across many locations.
- **Synchronization**: Managing the timing of processes so that operations across nodes are coordinated.
- **Data Replication**: Duplicating data across different nodes to increase reliability and accessibility.

This topic vector encompasses the primary elements and challenges associated with designing, implementing, and managing distributed systems, targeting advancements and solutions that address these core areas.

### Diagram

```plaintext
                                +-----------------------------+
                                |     Random Query Generator  |
                                |  (MAIN TOPIC VECTOR +       |
                                |   NICHE Words)              |
                                +--------------+--------------+
                                               |
                                               v
                                +--------------+--------------+
                                |         arXiv API           |
                                |  (Fetch top 3 recent        |
                                |   articles per query)       |
                                +--------------+--------------+
                                               |
                                               | (15 articles)
                                               v
                                +--------------+--------------+
                                |        Preprocessing        |
                                |   (Tokenization, Encoding)  |
                                +--------------+--------------+
                                               |
                                               | (Encoded texts)
                                               v
                                +--------------+--------------+
                                |           BERT Model        |
                                |  (Generate embeddings for   |
                                |   each article summary)     |
                                +--------------+--------------+
                                               |
                                               | (Embeddings)
                                               v
                                +--------------+--------------+
                                |      Similarity Computation |
                                |  (Compare embeddings to     |
                                |   'Distributed Systems'     |
                                |  MAIN topic vector)         |
                                +--------------+--------------+
                                               |
                                               | (Best match)
                                               v
                                +--------------+--------------+
                                |        Email System         |
                                |  (Send recommended article  |
                                |   to the user via email)    |
                                +-----------------------------+
```

---

# Text Similarity Methods

This repository provides examples of various methods to find related texts using different similarity measures and machine learning models.

## Table of Contents

1. [Cosine Similarity](#cosine-similarity)
2. [Jaccard Similarity](#jaccard-similarity)
3. [Euclidean Distance](#euclidean-distance)
4. [Manhattan Distance](#manhattan-distance)
5. [TF-IDF Vectorization](#tf-idf-vectorization)
6. [BM25 (Best Matching 25)](#bm25-best-matching-25)
7. [Topic Modeling (LDA - Latent Dirichlet Allocation)](#topic-modeling-lda---latent-dirichlet-allocation)
8. [Sentence Transformers (SBERT)](#sentence-transformers-sbert)
9. [Doc2Vec](#doc2vec)
10. [FastText](#fasttext)
11. [BERT](#bert)

## Cosine Similarity

Cosine similarity measures the cosine of the angle between two non-zero vectors. It is used to determine how similar two documents are, irrespective of their size.

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def cosine_sim(vec1, vec2):
    return cosine_similarity([vec1], [vec2])[0][0]

# Example usage
embedding1 = np.array([1, 2, 3])
embedding2 = np.array([2, 3, 4])
similarity = cosine_sim(embedding1, embedding2)
print(f"Cosine Similarity: {similarity:.4f}")
```

## Jaccard Similarity

Jaccard similarity measures the similarity between two sets by comparing the size of the intersection to the size of the union of the sets.

```python
def jaccard_similarity(text1, text2):
    set1 = set(text1.split())
    set2 = set(text2.split())
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)

# Example usage
text1 = "Machine learning is a field of artificial intelligence."
text2 = "Deep learning is a subset of machine learning."
similarity = jaccard_similarity(text1, text2)
print(f"Jaccard Similarity: {similarity:.4f}")
```

## Euclidean Distance

Euclidean distance measures the straight-line distance between two points in a vector space.

```python
from scipy.spatial.distance import euclidean

def euclidean_dist(vec1, vec2):
    return euclidean(vec1, vec2)

# Example usage
embedding1 = np.array([1, 2, 3])
embedding2 = np.array([2, 3, 4])
distance = euclidean_dist(embedding1, embedding2)
print(f"Euclidean Distance: {distance:.4f}")
```

## Manhattan Distance

Manhattan distance, also known as L1 distance, measures the absolute differences between two points in a vector space.

```python
from scipy.spatial.distance import cityblock

def manhattan_dist(vec1, vec2):
    return cityblock(vec1, vec2)

# Example usage
embedding1 = np.array([1, 2, 3])
embedding2 = np.array([2, 3, 4])
distance = manhattan_dist(embedding1, embedding2)
print(f"Manhattan Distance: {distance:.4f}")
```

## TF-IDF Vectorization

TF-IDF evaluates the importance of a word in a document relative to a collection of documents. It can be used to find related texts by calculating cosine similarity on TF-IDF vectors.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_similarity(corpus):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return cosine_similarity(tfidf_matrix)

# Example usage
corpus = ["Machine learning is a field of artificial intelligence.",
          "Deep learning is a subset of machine learning.",
          "Cooking recipes and techniques."]
similarity_matrix = tfidf_similarity(corpus)
print(f"TF-IDF Cosine Similarity:\n{similarity_matrix}")
```

## BM25 (Best Matching 25)

BM25 is a ranking function used by search engines to estimate the relevance of documents to a given search query.

```python
import rank_bm25

def bm25_similarity(corpus, query):
    tokenized_corpus = [doc.split() for doc in corpus]
    bm25 = rank_bm25.BM25Okapi(tokenized_corpus)
    tokenized_query = query.split()
    return bm25.get_scores(tokenized_query)

# Example usage
corpus = ["Machine learning is a field of artificial intelligence.",
          "Deep learning is a subset of machine learning.",
          "Cooking recipes and techniques."]
query = "Deep learning techniques"
scores = bm25_similarity(corpus, query)
print(f"BM25 Scores: {scores}")
```

## Topic Modeling (LDA - Latent Dirichlet Allocation)

LDA can be used to identify the underlying topics in a collection of documents. You can then compare the topic distributions of different documents to find related texts.

```python
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

def lda_similarity(corpus):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    lda = LatentDirichletAllocation(n_components=2, random_state=0)
    lda_features = lda.fit_transform(X)
    return cosine_similarity(lda_features)

# Example usage
corpus = ["Machine learning is a field of artificial intelligence.",
          "Deep learning is a subset of machine learning.",
          "Cooking recipes and techniques."]
similarity_matrix = lda_similarity(corpus)
print(f"LDA Cosine Similarity:\n{similarity_matrix}")
```

## Sentence Transformers (SBERT)

Sentence-BERT is a modification of the BERT network that uses siamese and triplet network structures to derive semantically meaningful sentence embeddings that can be compared using cosine similarity.

```python
from sentence_transformers import SentenceTransformer

def sbert_similarity(texts):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = model.encode(texts)
    return cosine_similarity(embeddings)

# Example usage
texts = ["Machine learning is a field of artificial intelligence.",
         "Deep learning is a subset of machine learning.",
         "Cooking recipes and techniques."]
similarity_matrix = sbert_similarity(texts)
print(f"SBERT Cosine Similarity:\n{similarity_matrix}")
```

## Doc2Vec

Doc2Vec is an extension of Word2Vec that generates vectors for entire documents.

```python
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

def doc2vec_similarity(corpus):
    documents = [TaggedDocument(doc.split(), [i]) for i, doc in enumerate(corpus)]
    model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)
    vectors = [model.infer_vector(doc.split()) for doc in corpus]
    return cosine_similarity(vectors)

# Example usage
corpus = ["Machine learning is a field of artificial intelligence.",
          "Deep learning is a subset of machine learning.",
          "Cooking recipes and techniques."]
similarity_matrix = doc2vec_similarity(corpus)
print(f"Doc2Vec Cosine Similarity:\n{similarity_matrix}")
```

## FastText

FastText is an efficient text classification and representation learning method, particularly useful for handling out-of-vocabulary words.

```python
import fasttext
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Ensure the FastText model is downloaded
model_path = 'cc.en.300.bin'
if not os.path.exists(model_path):
    fasttext.util.download_model('en', if_exists='ignore')

# Load FastText model
ft = fasttext.load_model(model_path)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def get_fasttext_embedding(text):
    tokens = preprocess_text(text).split()
    embeddings = [ft.get_word_vector(token) for token in tokens]
    return np.mean(embeddings, axis=0)

def fasttext_similarity(texts):
    embeddings = [get_fasttext_embedding(text) for text in texts]
    return cosine_similarity(embeddings)

# Example usage
texts = ["Machine learning is a field of artificial intelligence.",
         "Deep learning is a subset of machine learning.",
         "Cooking recipes and techniques."]
similarity_matrix = fasttext_similarity(texts)
print(f"FastText Cosine Similarity:\n{similarity_matrix}")
```

## BERT

BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained transformer model designed to understand the context of words in search queries.

```python
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

def get_bert_embedding(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
    return embeddings.numpy()

def bert_similarity(texts):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    embeddings = [get_bert_embedding(text, model, tokenizer) for text in texts]
    return cosine_similarity(embeddings)

# Example usage
texts = ["Machine learning is a field of artificial intelligence.",
         "Deep learning is a subset of machine learning.",
         "Cooking recipes and techniques."]
similarity_matrix = bert_similarity(texts)
print(f"BERT Cosine Similarity:\n{similarity_matrix}")
```
