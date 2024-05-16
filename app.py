from flask import Flask, request, jsonify, render_template, abort
from flask_cors import CORS
import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

DATABASE = 'papers.db'

# Database helper functions
def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

# CRUD API endpoints
@app.route('/papers', methods=['GET'])
def get_papers():
    conn = get_db_connection()
    sort_by = request.args.get('sort_by', 'pubDate')
    order = request.args.get('order', 'desc')
    source = request.args.get('source')
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 10))
    offset = (page - 1) * per_page

    query = 'SELECT * FROM entries'
    params = []

    if source:
        query += ' WHERE source = ?'
        params.append(source)

    query += f' ORDER BY {sort_by} {order.upper()} LIMIT ? OFFSET ?'
    params.extend([per_page, offset])

    papers = conn.execute(query, params).fetchall()
    conn.close()
    return jsonify([dict(paper) for paper in papers])

@app.route('/papers/<int:id>', methods=['GET'])
def get_paper(id):
    conn = get_db_connection()
    paper = conn.execute('SELECT * FROM entries WHERE id = ?', (id,)).fetchone()
    conn.close()
    if paper is None:
        abort(404, description="Paper not found")
    return jsonify(dict(paper))

@app.route('/papers', methods=['POST'])
def create_paper():
    new_paper = request.json
    conn = get_db_connection()
    conn.execute(
        'INSERT INTO entries (title, summary, link, source, pubDate) VALUES (?, ?, ?, ?, ?)',
        (new_paper['title'], new_paper['summary'], new_paper['link'], new_paper['source'], new_paper['pubDate'])
    )
    conn.commit()
    conn.close()
    return jsonify(new_paper), 201

@app.route('/papers/<int:id>', methods=['PUT'])
def update_paper(id):
    updated_paper = request.json
    conn = get_db_connection()
    conn.execute(
        'UPDATE entries SET title = ?, summary = ?, link = ?, source = ?, pubDate = ? WHERE id = ?',
        (updated_paper['title'], updated_paper['summary'], updated_paper['link'], updated_paper['source'], updated_paper['pubDate'], id)
    )
    conn.commit()
    conn.close()
    return jsonify(updated_paper)

@app.route('/papers/<int:id>', methods=['DELETE'])
def delete_paper(id):
    conn = get_db_connection()
    conn.execute('DELETE FROM entries WHERE id = ?', (id,))
    conn.commit()
    conn.close()
    return '', 204

@app.route('/sources', methods=['GET'])
def get_sources():
    conn = get_db_connection()
    sources = conn.execute('SELECT DISTINCT source FROM entries').fetchall()
    conn.close()
    return jsonify([source['source'] for source in sources])

@app.route('/papers/search', methods=['GET'])
def search_papers():
    conn = get_db_connection()
    query = request.args.get('query', '')
    papers = conn.execute(
        'SELECT * FROM entries WHERE title LIKE ? OR summary LIKE ? OR source LIKE ?',
        (f'%{query}%', f'%{query}%', f'%{query}%')
    ).fetchall()
    conn.close()
    return jsonify([dict(paper) for paper in papers])

@app.route('/papers/<int:id>/recommendations', methods=['GET'])
def get_recommendations(id):
    conn = get_db_connection()
    papers = conn.execute('SELECT * FROM entries').fetchall()
    conn.close()

    df = pd.DataFrame(papers, columns=['id', 'title', 'summary', 'link', 'source', 'pubDate'])
    if df.empty:
        return jsonify({'recent': [], 'old': []})

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform( df.iloc[:, 1] )

    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    sim_scores = list(enumerate(cosine_sim[id]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [i for i in sim_scores if i[0] != id]

    recent_papers = df.loc[[i[0] for i in sim_scores[:3]]]
    old_papers = df.loc[[i[0] for i in sim_scores[-3:]]]

    return jsonify({
        'recent': recent_papers.to_dict(orient='records'),
        'old': old_papers.to_dict(orient='records')
    })

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port = 8080)
