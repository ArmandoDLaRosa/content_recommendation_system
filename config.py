import logging

# Logging configuration
LOGGING_LEVEL = logging.INFO
LOGGING_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

# Email configuration
EMAIL_URL = 'http://192.168.1.181:5000/send-email'
EMAIL_HEADERS = {'Content-Type': 'application/json'}

# Topics for BERT model
TOPICS = [
    "artificial intelligence", "machine learning", "natural language processing",
    "computer vision", "data science", "robotics", "bioinformatics",
    "theoretical computer science", "network security", "quantum computing",
    "software engineering", "distributed systems", "cybersecurity",
    "blockchain", "cloud computing", "edge computing", "big data",
    "IoT", "AR/VR", "ethical AI", "deep learning", "reinforcement learning",
    "speech recognition", "genomics", "computational biology", "information retrieval",
    "data mining", "algorithm design", "human-computer interaction", "software architecture",
    "privacy and security", "computer graphics", "programming languages", "systems programming",
    "Bayesian methods", "quantum algorithms", "distributed algorithms",
    "theoretical machine learning", "computational complexity", "information theory",
    "cryptography", "formal languages and automata theory", "logic in computer science",
    "multimedia computing", "operating systems", "performance evaluation",
    "programming language theory", "scientific computing", "social and information networks",
    "symbolic computation", "computer systems and networks", "formal verification",
    "hardware architecture", "mobile computing", "software testing and verification",
    "computational geometry", "bioinformatics algorithms", "computer-aided design",
    "data structures", "data privacy", "embedded systems", "high performance computing",
    "mobile robotics", "numerical analysis", "software metrics"
]


# RSS feed URLs
ARXIV_CATEGORIES = [
    "cs.AI",  # Artificial Intelligence
    "cs.LG",  # Machine Learning
    "cs.CL",  # Computation and Language (Natural Language Processing)
    "cs.CV",  # Computer Vision and Pattern Recognition
    "stat.ML",  # Statistics and Machine Learning
    "cs.CR",  # Cryptography and Security
    "cs.DC",  # Distributed, Parallel, and Cluster Computing
    "cs.DS",  # Data Structures and Algorithms
    "cs.NI",  # Networking and Internet Architecture
    "cs.SE",  # Software Engineering
    "cs.RO",  # Robotics
    "cs.SI",  # Social and Information Networks
    "cs.CY",  # Computers and Society (Cybersecurity, Ethical AI)
    "cs.DM",  # Discrete Mathematics (Theoretical Computer Science)
    "cs.GL",  # General Literature (Multidisciplinary topics)
    "cs.MA",  # Multiagent Systems
    "cs.MM",  # Multimedia (Computer Graphics, AR/VR)
    "cs.SY",  # Systems and Control (Control Systems, Edge Computing)
    "cs.OS",  # Operating Systems
    "cs.PL"   # Programming Languages
]

BLOG_SOURCES = [
    # Engineering Blogs
    "https://aws.amazon.com/blogs/architecture/feed/",
    "https://slack.engineering/feed",
    "https://engineering.linkedin.com/blog.rss",
    "https://engineering.fb.com/feed/",
    "https://engineering.atspotify.com/feed",
    "https://netflixtechblog.com/feed",
    "https://medium.com/feed/airbnb-engineering",
    "https://medium.com/feed/@pinterest",
    "https://medium.com/feed/paypal-engineering",
    "https://medium.com/feed/@cloudflare",
    "https://eng.lyft.com/feed",
    "https://developers.googleblog.com/feed/",
    "https://developer.ibm.com/blogs/feed/",
    "https://blog.acolyer.org/feed/",
    "https://www.statistics.com/blog/feed/",
    # Medium Tagged
    # Towards Data Science Tagged
    "https://towardsdatascience.com/feed/tagged/bayesian",
    "https://towardsdatascience.com/feed/tagged/distributed",
    "https://towardsdatascience.com/feed/tagged/optimization",
    "https://towardsdatascience.com/feed/tagged/computation"
]

# Database configuration
DB_PATH = 'papers.db'
