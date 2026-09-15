import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import UploadFile, HTTPException
import project_store as store
from dataset.dataset_loader import process_dataset


class ProjectStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.patches = [patch.object(store, 'ROOT', self.root),
                        patch.object(store, 'DB_PATH', self.root / 'outputs/projects.sqlite3')]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in self.patches:
            item.stop()
        self.temp.cleanup()

    def test_names_unique_runs_and_delete_only_owned_files(self):
        first = store.allocate('one', '../../Same project', 'training')
        second = store.allocate('one', '../../Same project', 'training')
        other = store.allocate('two', '../../Same project', 'training')
        self.assertNotEqual(first, second)
        self.assertIn('Same-project--one', str(first))
        for path in (first, second, other):
            (path / 'model.safetensors').write_text('weights')
        source = self.root / 'base_model'
        source.mkdir()
        store.save_project('one', {'name': 'Renamed', 'model': {'adaptorPath': str(source)}})
        store.record('one', 'training_result', {'output_dir': str(first)})
        store.delete_project('one')
        self.assertFalse(first.exists())
        self.assertFalse(second.exists())
        self.assertTrue(other.exists())
        self.assertTrue(source.exists())
        with store.database() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM events').fetchone()[0], 0)
        with self.assertRaises(ValueError):
            store.save_project('one', {'name': 'Stale browser save'})

    def test_upload_raw_and_processed_are_deleted(self):
        directory = store.allocate('upload', 'My dataset', 'datasets')
        upload = UploadFile(filename='questions.csv', file=io.BytesIO(b'question,answer\nWhy?,Because\n'))
        df, path = process_dataset(upload, '{question}', 'forget_set', directory)
        self.assertEqual(len(df), 1)
        self.assertTrue(Path(path).exists())
        self.assertEqual(len(list(directory.iterdir())), 2)
        store.delete_project('upload')
        self.assertFalse(directory.exists())

    def test_extraction_uploads_and_artifacts_have_one_owner(self):
        from data_selection import caching
        def upload():
            return UploadFile(filename='rows.csv', file=io.BytesIO(b'question,answer\nWhy?,Because\n'))
        with patch.object(caching, 'PROJECT_ROOT', self.root), patch.object(caching, 'GRADIENTS_ROOT', self.root / 'outputs/gradients'), patch.object(caching, 'RASLIK_UPLOAD_ROOT', self.root / 'uploaded_datasets/raslik'):
            prepared = caching.prepare_uploaded_datasets(
                full_dataset=upload(), poison_set=upload(), prompt_template='{question}',
                experiment_name='unused', model_name='test/model', max_length=32,
                adaptor_path=None, project_id='extract', project_name='Extract project')
        self.assertIn('Extract-project--extract-', prepared['run_name'])
        artifacts = store.get_project('extract')['artifacts']
        self.assertEqual(len(artifacts), 2)
        raw_files = list((self.root / 'uploaded_datasets').rglob('*_raw_*'))
        self.assertEqual(len(raw_files), 2)
        store.delete_project('extract')
        self.assertTrue(all(not Path(item['path']).exists() for item in artifacts))

    def test_sqlite_metadata_and_secrets(self):
        store.save_project('id', {'name': 'Experiment', 'model': {'modelName': 'base', 'method': 'adaptor', 'adaptorPath': '/adapter', 'hfKey': 'secret'}, 'data': {'sourceMode': 'extract', 'selectionMethod': 'grace'}, 'hyperparameters': {'unlearningMethod': 'npo', 'learningRate': 0.01}, 'run': {'evaluation': {'score': 0.5}}})
        store.record('id', 'training_config', {'model': {'hf_key': 'secret'}, 'hyperparams': {'beta': 1}})
        record = store.get_project('id')
        self.assertNotIn('secret', json.dumps(record))
        self.assertEqual(record['events'][0]['details']['hyperparams']['beta'], 1)
        with store.database() as db:
            row = db.execute('SELECT * FROM project_details').fetchone()
            self.assertEqual(row['extract_method'], 'grace')
            self.assertEqual(row['pre_unlearning_model_name'], 'base')
            self.assertEqual(row['is_adaptor_unlearning'], 1)
            self.assertEqual(json.loads(row['eval_results']), {'score': 0.5})

    def test_unsafe_paths_and_symlinks_are_rejected(self):
        with self.assertRaises(ValueError):
            store.allocate('../escape', 'unsafe', 'datasets')
        directory = store.allocate('safe', 'Safe', 'datasets')
        with self.assertRaises(ValueError):
            store.register('safe', self.root, 'model')
        directory.rmdir()
        outside = self.root / 'source'
        outside.mkdir()
        directory.symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError):
            store.delete_project('safe')
        self.assertTrue(outside.exists())

    def test_failed_cleanup_keeps_metadata_for_retry(self):
        store.allocate('id', 'Retry', 'training')
        with patch.object(store.shutil, 'rmtree', side_effect=OSError('busy')):
            with self.assertRaises(OSError):
                store.delete_project('id')
        self.assertEqual(store.get_project('id')['name'], 'Retry')
        store.delete_project('id')
        with self.assertRaises(KeyError):
            store.get_project('id')

    def test_delete_is_blocked_during_upload_startup(self):
        from routes.project_routes import delete_project
        @store.protect
        def upload():
            with self.assertRaises(HTTPException) as error:
                delete_project('id')
            self.assertEqual(error.exception.status_code, 409)
        upload()
        self.assertEqual(store.ACTIVE_REQUESTS, 0)

    def test_api_upload_and_deletion(self):
        from fastapi.testclient import TestClient
        from main import app
        client = TestClient(app)
        response = client.put('/api/projects/api', json={'name': 'API project'})
        self.assertEqual(response.status_code, 200)
        response = client.post('/api/dataset/upload', data={'project_id': 'api', 'project_name': 'API project', 'prompt_template': '{question}'}, files={'forget_set': ('sample.csv', b'question,answer\nWhy?,Because\n', 'text/csv')})
        self.assertEqual(response.status_code, 200, response.text)
        path = Path(response.json()['forget_set_path'])
        self.assertTrue(path.exists())
        project = client.get('/api/projects/api').json()
        self.assertEqual(project['events'][0]['kind'], 'dataset_upload')
        self.assertEqual(client.delete('/api/projects/api').status_code, 200)
        self.assertFalse(path.exists())
        self.assertEqual(client.get('/api/projects/api').status_code, 404)
        self.assertEqual(client.put('/api/projects/api', json={'name': 'stale'}).status_code, 409)
