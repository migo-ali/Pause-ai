import os
import sqlite3
from flask import Flask, request, render_template, redirect, url_for
from openai import OpenAI

app = Flask(__name__)
DATABASE = os.path.join(os.path.dirname(__file__), 'dashboard.db')
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute(
        """CREATE TABLE IF NOT EXISTS projects (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        prompt TEXT NOT NULL,
        last_response TEXT
    )"""
    )
    conn.commit()
    conn.close()


@app.route('/', methods=['GET', 'POST'])
def index():
    conn = get_db()
    if request.method == 'POST':
        name = request.form['name']
        prompt = request.form['prompt']
        conn.execute(
            'INSERT INTO projects (name, prompt, last_response) VALUES (?, ?, ?)',
            (name, prompt, '')
        )
        conn.commit()
        return redirect(url_for('index'))
    projects = conn.execute('SELECT * FROM projects').fetchall()
    conn.close()
    return render_template('index.html', projects=projects)


@app.route('/update/<int:project_id>', methods=['POST'])
def update(project_id):
    conn = get_db()
    project = conn.execute('SELECT * FROM projects WHERE id=?', (project_id,)).fetchone()
    if project:
        response = client.responses.create(
            model="gpt-4o-mini",
            input=project['prompt']
        )
        conn.execute(
            'UPDATE projects SET last_response=? WHERE id=?',
            (response.output_text, project_id)
        )
        conn.commit()
    conn.close()
    return redirect(url_for('index'))


if __name__ == '__main__':
    init_db()
    app.run(debug=True)
