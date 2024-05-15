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
    "privacy and security", "computer graphics", "programming languages", "systems programming"
]

# RSS feed URLs
ARXIV_CATEGORIES = [
    "cs.AI", "cs.LG", "cs.CL", "cs.CV", "stat.ML",
    "cs.CR", "cs.DC", "cs.DS", "cs.NI", "cs.SE",
    "cs.RO", "cs.SI", "cs.CY", "cs.DM", "cs.GL",
    "cs.MA", "cs.MM", "cs.SY", "cs.OS", "cs.PL"
]

BLOG_SOURCES = [
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
    "https://blog.acolyer.org/feed/"
]

# Database configuration
DB_PATH = 'papers.db'
