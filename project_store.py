"""SQLite metadata and explicit ownership of generated project files."""
import json
import re
import shutil
import sqlite3
import uuid
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from threading import RLock

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / 'outputs' / 'projects.sqlite3'
LOCK = RLock()
ACTIVE_REQUESTS = 0


def protect(function):
    """Close the startup/upload race with project deletion."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        global ACTIVE_REQUESTS
        with LOCK:
            ACTIVE_REQUESTS += 1
        try:
            return function(*args, **kwargs)
        finally:
            with LOCK:
                ACTIVE_REQUESTS -= 1
    return wrapped



def clean(value):
    """Never persist access tokens with experiment metadata."""
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()
                if k.lower() not in {'hf_key', 'hfkey', 'extractionhfkey', 'token', 'authorization'}}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


@contextmanager
def _connection():
    db = sqlite3.connect(DB_PATH, timeout=30)
    try:
        with db:
            yield db
    finally:
        db.close()


@contextmanager
def database():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCK, _connection() as db:
        db.row_factory = sqlite3.Row
        db.executescript('''
            PRAGMA foreign_keys=ON;
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, details TEXT NOT NULL DEFAULT '{}',
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS artifacts (
                path TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                kind TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                kind TEXT NOT NULL, details TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS deleted_projects (id TEXT PRIMARY KEY);
            CREATE VIEW IF NOT EXISTS project_details AS SELECT
                id AS project_id, name,
                json_extract(details, '$.data') AS dataset,
                CASE WHEN json_extract(details, '$.data.sourceMode') = 'extract'
                     THEN json_extract(details, '$.data.selectionMethod') ELSE 'upload' END AS extract_method,
                json_extract(details, '$.model.modelName') AS pre_unlearning_model_name,
                json_extract(details, '$.model.method') = 'adaptor' AS is_adaptor_unlearning,
                json_extract(details, '$.model.adaptorPath') AS adaptor_path,
                json_extract(details, '$.hyperparameters.unlearningMethod') AS unlearning_algorithm,
                json_extract(details, '$.hyperparameters') AS hyperparams,
                json_extract(details, '$.run.evaluation') AS eval_results,
                updated_at FROM projects;
        ''')
        yield db


def ensure_project(project_id, name=None):
    if not project_id or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', project_id):
        raise ValueError('A file-safe project_id (1–128 characters) is required.')
    with database() as db:
        if db.execute('SELECT 1 FROM deleted_projects WHERE id=?', (project_id,)).fetchone():
            raise ValueError('This project has been deleted.')
        db.execute('INSERT OR IGNORE INTO projects(id,name) VALUES (?,?)',
                   (project_id, name or project_id))
        if name:
            db.execute('UPDATE projects SET name=?,updated_at=CURRENT_TIMESTAMP WHERE id=?', (name, project_id))
        return db.execute('SELECT name FROM projects WHERE id=?', (project_id,)).fetchone()[0]


def project_slug(project_id, name=None):
    name = ensure_project(project_id, name)
    slug = re.sub(r'[^A-Za-z0-9_-]+', '-', name).strip('-_')[:80] or 'project'
    return f'{slug}--{project_id}'


def register(project_id, path, kind):
    path = Path(path).absolute()
    # Only generated storage is eligible for ownership and recursive deletion.
    roots = [ROOT / 'outputs' / 'runs', ROOT / 'outputs' / 'gradients', ROOT / 'uploaded_datasets']
    if not any(path.resolve().is_relative_to(root.resolve()) and path.resolve() != root.resolve() for root in roots):
        raise ValueError(f'Not a managed artifact path: {path}')
    with database() as db:
        db.execute('INSERT INTO artifacts(path,project_id,kind) VALUES (?,?,?)', (str(path), project_id, kind))
    return path


def allocate(project_id, name, kind):
    slug = project_slug(project_id, name)
    root = ROOT / ('uploaded_datasets' if kind == 'datasets' else 'outputs/runs')
    path = root / slug / f'{kind}-{uuid.uuid4().hex}'
    register(project_id, path, kind)
    path.mkdir(parents=True, exist_ok=False)
    return path


def record(project_id, kind, details):
    if not project_id:
        return
    with database() as db:
        if not db.execute('SELECT 1 FROM projects WHERE id=?', (project_id,)).fetchone():
            return
        db.execute('INSERT INTO events(project_id,kind,details) VALUES (?,?,?)',
                   (project_id, kind, json.dumps(clean(details), default=str)))


def save_project(project_id, details):
    with LOCK:
        try:
            existing = get_project(project_id)['details']
        except KeyError:
            existing = {}
        if existing.get('updatedAt', '') > details.get('updatedAt', ''):
            return
        ensure_project(project_id, details.get('name'))
        _save_details(project_id, details)


def _save_details(project_id, details):
    details = dict(details, id=project_id)
    with database() as db:
        db.execute('UPDATE projects SET details=?,updated_at=CURRENT_TIMESTAMP WHERE id=?',
                   (json.dumps(clean(details)), project_id))


def list_projects():
    with database() as db:
        return [dict(row, details=json.loads(row['details'])) for row in db.execute('SELECT * FROM projects ORDER BY updated_at DESC')]


def get_project(project_id):
    with database() as db:
        row = db.execute('SELECT * FROM projects WHERE id=?', (project_id,)).fetchone()
        if row is None:
            raise KeyError(project_id)
        return dict(row, details=json.loads(row['details']),
                    artifacts=[dict(r) for r in db.execute('SELECT * FROM artifacts WHERE project_id=?', (project_id,))],
                    events=[dict(r, details=json.loads(r['details'])) for r in db.execute('SELECT * FROM events WHERE project_id=? ORDER BY id', (project_id,))])


def delete_project(project_id):
    with database() as db:
        paths = db.execute('SELECT path FROM artifacts WHERE project_id=?', (project_id,)).fetchall()
        for row in paths:
            path = Path(row['path'])
            # Recheck containment: directories could have been replaced with symlinks.
            roots = [ROOT / 'outputs/runs', ROOT / 'outputs/gradients', ROOT / 'uploaded_datasets']
            if not any(path.resolve().is_relative_to(r.resolve()) and path.resolve() != r.resolve() for r in roots):
                raise ValueError(f'Refusing to delete artifact outside managed storage: {path}')
            if path.is_symlink() or path.is_file():
                path.unlink()
            elif path.exists():
                shutil.rmtree(path)
            if path.parent.exists() and not any(path.parent.iterdir()) and path.parent not in roots:
                path.parent.rmdir()
        db.execute('DELETE FROM projects WHERE id=?', (project_id,))
        db.execute('INSERT OR IGNORE INTO deleted_projects(id) VALUES (?)', (project_id,))
