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
    try:
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
        total_papers = conn.execute('SELECT COUNT(*) FROM entries').fetchone()[0]
        total_pages = (total_papers + per_page - 1) // per_page
        conn.close()
        return jsonify({'papers': [dict(paper) for paper in papers], 'page': page, 'total_pages': total_pages})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/papers/<int:id>', methods=['GET'])
def get_paper(id):
    try:
        conn = get_db_connection()
        paper = conn.execute('SELECT * FROM entries WHERE id = ?', (id,)).fetchone()
        conn.close()
        if paper is None:
            abort(404, description="Paper not found")
        return jsonify(dict(paper))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/papers', methods=['POST'])
def create_paper():
    try:
        new_paper = request.json
        conn = get_db_connection()
        conn.execute(
            'INSERT INTO entries (title, summary, link, source, pubDate) VALUES (?, ?, ?, ?, ?)',
            (new_paper['title'], new_paper['summary'], new_paper['link'], new_paper['source'], new_paper['pubDate'])
        )
        conn.commit()
        conn.close()
        return jsonify(new_paper), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/papers/<int:id>', methods=['PUT'])
def update_paper(id):
    try:
        updated_paper = request.json
        conn = get_db_connection()
        conn.execute(
            'UPDATE entries SET title = ?, summary = ?, link = ?, source = ?, pubDate = ? WHERE id = ?',
            (updated_paper['title'], updated_paper['summary'], updated_paper['link'], updated_paper['source'], updated_paper['pubDate'], id)
        )
        conn.commit()
        conn.close()
        return jsonify(updated_paper)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/papers/<int:id>', methods=['DELETE'])
def delete_paper(id):
    try:
        conn = get_db_connection()
        conn.execute('DELETE FROM entries WHERE id = ?', (id,))
        conn.commit()
        conn.close()
        return '', 204
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/sources', methods=['GET'])
def get_sources():
    try:
        conn = get_db_connection()
        sources = conn.execute('SELECT DISTINCT source FROM entries').fetchall()
        conn.close()
        return jsonify([source['source'] for source in sources])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/papers/search', methods=['GET'])
def search_papers():
    try:
        conn = get_db_connection()
        query = request.args.get('query', '')
        papers = conn.execute(
            'SELECT * FROM entries WHERE title LIKE ? OR summary LIKE ? OR source LIKE ?',
            (f'%{query}%', f'%{query}%', f'%{query}%')
        ).fetchall()
        conn.close()
        return jsonify([dict(paper) for paper in papers])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/papers/<int:id>/recommendations', methods=['GET'])
def get_recommendations(id):
    try:
        conn = get_db_connection()
        papers = conn.execute('SELECT * FROM entries').fetchall()
        conn.close()

        if not papers:
            return jsonify({'recent': [], 'old': []})

        df = pd.DataFrame(papers, columns=['id', 'title', 'summary', 'link', 'source', 'pubDate'])
        if df.empty:
            return jsonify({'recent': [], 'old': []})

        df['content'] = df['title'] + ' ' + df['summary']

        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['content'])

        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

        idx = df.index[df['id'] == id].tolist()
        if not idx:
            return jsonify({'recent': [], 'old': []})
        idx = idx[0]

        sim_scores = list(enumerate(cosine_sim[idx]))

        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        sim_scores = [i for i in sim_scores if i[0] != idx]

        recent_papers_idx = [i[0] for i in sim_scores[:3]]
        old_papers_idx = [i[0] for i in sim_scores[-3:]]

        recent_papers = df.iloc[recent_papers_idx]
        old_papers = df.iloc[old_papers_idx]

        return jsonify({
            'recent': recent_papers.to_dict(orient='records'),
            'old': old_papers.to_dict(orient='records')
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=8080)
