from fastapi import APIRouter, HTTPException
import project_store

router = APIRouter(prefix='/projects', tags=['Projects'])


@router.get('')
def list_projects():
    return project_store.list_projects()


@router.get('/deleted/ids')
def deleted_ids():
    with project_store.database() as db:
        return [row[0] for row in db.execute('SELECT id FROM deleted_projects')]


@router.get('/{project_id}')
def get_project(project_id: str):
    try:
        return project_store.get_project(project_id)
    except KeyError as exc:
        raise HTTPException(404, 'Project not found') from exc


@router.put('/{project_id}')
def save_project(project_id: str, details: dict):
    try:
        project_store.save_project(project_id, details)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {'status': 'saved'}


@router.delete('/{project_id}')
def delete_project(project_id: str):
    from training_process import training_process_manager
    from evaluation_process import evaluation_process_manager
    from data_selection.caching import gradient_cache_manager
    with project_store.LOCK:
        if project_store.ACTIVE_REQUESTS or any(manager.is_running for manager in (training_process_manager, evaluation_process_manager, gradient_cache_manager)):
            raise HTTPException(409, 'Stop active training, extraction, or evaluation before deleting a project.')
        try:
            project_store.delete_project(project_id)
        except (ValueError, OSError) as exc:
            raise HTTPException(409, str(exc)) from exc
    return {'status': 'deleted'}
