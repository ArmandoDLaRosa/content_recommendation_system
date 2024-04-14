#!/bin/bash

if [ "$(id -u)" != "0" ]; then
    echo "run as root" 1>&2
    exit 1
fi

if ! command -v docker &> /dev/null; then
    echo "Docker, installing..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    echo "Docker installed successfully."
else
    echo "Docker already installed."
fi

echo "Checking Docker service..."
if command -v systemctl &> /dev/null && systemctl is-active --quiet docker; then
    echo "Docker already running."
elif command -v service &> /dev/null && service docker status &> /dev/null; then
    echo "Docker running."
else
    echo "Attempting to start Docker..."
    if command -v systemctl &> /dev/null; then
        systemctl start docker || echo "Failed to start Docker with systemctl."
    elif command -v service &> /dev/null; then
        service docker start || echo "Failed to start Docker with service."
    else
        echo "No known service management tool available to start Docker. Manual intervention required."
        exit 1
    fi
fi

sleep 10  


echo "Installing necessary system dependencies for grpcio and other packages..."
apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    python3-dev \
    gcc \
    g++ \
    make \
    cmake \
    libffi-dev

echo "Setting up Milvus Docker container..."

docker pull milvusdb/milvus:v2.0.0-rc8
docker run -d --name milvus_cpu_2.0 \
  -p 19530:19530 \
  -p 19121:19121 \
  -v ~/milvus/db:/var/lib/milvus/db \
  -v ~/milvus/conf:/var/lib/milvus/conf \
  -v ~/milvus/logs:/var/lib/milvus/logs \
  -v ~/milvus/wal:/var/lib/milvus/wal \
  milvusdb/milvus:v2.0.0-rc8

echo "Waiting for Milvus to start..."
sleep 10 

pip install pymilvus==2.0.0rc8
pip install torch transformers

echo "Setting up collections..."
python3 - <<EOF
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection

# Connect to Milvus
connections.connect("default", host='localhost', port='19530')

def create_collection(collection_name, fields):
    schema = CollectionSchema(fields=fields, description=f"{collection_name} collection")
    collection = Collection(name=collection_name, schema=schema)
    print(f"Collection {collection_name} created successfully.")

# Fields for main_topic and niche
text_field = FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1024, is_primary=False, auto_id=True)
vector_field = FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768)  # BERT typical dimension

# Fields for content_table
datetime_field = FieldSchema(name="datetime", dtype=DataType.DATETIME, is_primary=False)
source_field = FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=255, is_primary=False)
url_field = FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=1024, is_primary=False)

# Create collections
create_collection("main_topic", [text_field, vector_field])
create_collection("niche", [text_field, vector_field])
create_collection("content_table", [datetime_field, source_field, url_field, vector_field])

EOF

echo "Milvus setup and collection creation complete."

