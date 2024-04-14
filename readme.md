
# Personalized Article Recommendation System

## Overview

The Personalized Article Recommendation System is designed to revolutionize the reading experience by offering a singular, highly relevant article recommendation daily. Tailored to individual interests in specialized fields such as Distributed Systems, AI Scaling, Low-Level Engineering, and System Design, this system leverages advanced Natural Language Processing (NLP), dynamic feedback loops, and Retrieval-Augmented Generation (RAG) technologies. Our goal is to provide unparalleled personalization in content recommendation and discovery. By having Topic Vectors its able to search and discover new articles in the predefined sources that match those keywords mixed with niching words selected by the user.

## Key Features

- **Singular Daily Article Recommendation**: Delivers one article per day, chosen for its relevance and alignment with the user's interests by ponderating
                                             the most recent articles from all sources and delivering only the most relevant towards the interests.
                                             We make sure it hasnt been read by the user and it's as recent as possible in the origin source.
- **Similar Article Discovery**: Offers a selection of related articles to encourage exploration of topics in depth. Sortable by Most Recent.
- **RAG-Powered Search**: Allows for natural language queries, enhancing the discovery of articles that match user interests closely. Sortable by Most Recent
- **User Feedback Integration**: Utilizes direct user inputs and article rankings to continuously refine the recommendation process.

## Implementation Overview

### System Architecture

The architecture integrates various technologies, including Topic Model Vectors, Semantic Relevance Analysis, Text Similarity Clusters, a Vector Database (VectorDB), Retrieval-Augmented Generation (RAG), and User Feedback Mechanisms. These components work together to facilitate personalized article recommendations and enable effective content discovery.

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
2. **Daily Recommendation**: Utilization of semantic relevance and user interest vectors to select the most pertinent article for each user daily.
3. **Similar Article Discovery**: Provision of related articles through text similarity analysis and RAG-powered recommendations, fostering content exploration.
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
```

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
[Retrieval-Augmented Generation (RAG)]------>[Recommendation Engine]
       |                   ^                         |
       |                   |                         |
       | User Queries      | Feedback Loop           | Recommendations
       |                   |                         |
       v                   v                         v
                      [User Interface]
                        (Frontend)
```

## Data Flow Diagram
```
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
```
            +------------------------------------------------+
            |                    [User Interface]            |
            |                        (FE)                    |
            |                                                 |
+-----------v----------+                         +-----------v-----------+
|   [Backend (BE)]     | <------ APIs ------>    | [RAG & ML Models]     |
| - API management     |                         | - Embedding generation|
| - Auth & session mgmt|                         | - Similarity matching |
| - Feedback processing|                         | - Recommendation algo |
+-----------^----------+                         +-----------^-----------+
            |                                                     |
            |                                                     |
            |                +---------------------+             |
            +---------------->  [Vector Database]  <-------------+
                             |       (DE)          |
                             | - Article storage   |
                             | - Embedding storage |
                             | - Query processing  |
                             +---------------------+
```

## How each section works?
### Article Processing 
```
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

###  Recommendation Generation Process
```
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
```
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

```
                              +---------------------+
                              |   Enhanced Big      |
                              |      Texts          |
                              +----------+----------+
                                         |
                                         | Embedding Generation (BERT)
                                         v
                    +--------------------+--------------------+
                    |              Vector Database              |
                    | (Stores embeddings & article metadata)    |
                    +--------+---------------------+--------+
                             |                     |
         +-------------------+                     +-------------------+
         |                                          |                  
 +-------v-------+                          +-------v-------+    +-----+----+
 |  Search Query |                          |  Fetch & Rank |    | User     |
 |  Construction +------------------------->+  Articles     +--->+ Ranking  |
 |  (with RAG)   |    (arXiv, Medium, etc.) |               |    | (1 to 5)  |
 |               |                          +-------+-------+    +-----+----+
 +-------+-------+                                  |                  |
         |            Similar Articles              | Similar articles |
         |            with RAG                      | with RAG         |
         v                                          v                  |
 +-------+--------+                         +-------+--------+        |
 | Natural        |                         | Article-based  |<-------+
 | Language       |                         | Recommendations|
 | Search         |                         +----------------+
 +----------------+
```


## MVP - Single Article Recommendation

### MAIN TOPIC VECTOR - Distributed Systems
Distributed Systems encompass a broad array of technologies and concepts focused on managing and operating multiple interlinked computing nodes that work together to perform complex tasks. These systems are characterized by their scalability, reliability, and efficiency. Key aspects include:

Scalability: Ability to handle increasing workloads by adding resources either horizontally (more nodes) or vertically (more powerful nodes).
Concurrency: Managing multiple parts of a computation simultaneously to optimize processing speed.
Fault Tolerance: Ensuring the system continues to operate even if one or more nodes fail.
Load Balancing: Distributing work evenly across all nodes to prevent any single node from being overwhelmed, which ensures smooth operation and optimal resource utilization.
Cloud Computing: Utilizing online computational resources that can be scaled and shared among users, often encompassing distributed databases and applications.
Decentralization: Removing a central point of control or failure by distributing data and tasks across many locations.
Synchronization: Managing the timing of processes so that operations across nodes are coordinated.
Data Replication: Duplicating data across different nodes to increase reliability and accessibility.
This topic vector encompasses the primary elements and challenges associated with designing, implementing, and managing distributed systems, targeting advancements and solutions that address these core areas.

### NICHE WORDS - Construction
Distributed Systems encompass a broad array of technologies and concepts focused on managing and operating multiple interlinked computing nodes that work together to perform complex tasks. These systems are characterized by their scalability, reliability, and efficiency. Key aspects include:

Scalability: Ability to handle increasing workloads by adding resources either horizontally (more nodes) or vertically (more powerful nodes).
Concurrency: Managing multiple parts of a computation simultaneously to optimize processing speed.
Fault Tolerance: Ensuring the system continues to operate even if one or more nodes fail.
Load Balancing: Distributing work evenly across all nodes to prevent any single node from being overwhelmed, which ensures smooth operation and optimal resource utilization.
Cloud Computing: Utilizing online computational resources that can be scaled and shared among users, often encompassing distributed databases and applications.
Decentralization: Removing a central point of control or failure by distributing data and tasks across many locations.
Synchronization: Managing the timing of processes so that operations across nodes are coordinated.
Data Replication: Duplicating data across different nodes to increase reliability and accessibility.
This topic vector encompasses the primary elements and challenges associated with designing, implementing, and managing distributed systems, targeting advancements and solutions that address these core areas.

### Diagram
                                +-----------------------------+
                                |     Random Query Generator  |
                                |  (MAIN TOPIC VECTOR +     |
                                |   NICHE Words)       |
                                +--------------+--------------+
                                               |
                                               v
                                +--------------+--------------+
                                |         arXiv API          |
                                |  (Fetch top 3 recent       |
                                |   articles per query)      |
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
                                |  MAIN topic vector)             |
                                +--------------+--------------+
                                               |
                                               | (Best match)
                                               v
                                +--------------+--------------+
                                |        Email System         |
                                |  (Send recommended article  |
                                |   to the user via email)    |
                                +------------------------------+
