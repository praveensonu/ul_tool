import { apiUrl } from "../apiConfig";
import { normalizeProject } from "../defaults";
import type { Project } from "../types";

const DB_NAME = "ascent-unlearning-projects";
const DB_VERSION = 1;
const STORE_NAME = "projects";

function openDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        const store = db.createObjectStore(STORE_NAME, { keyPath: "id" });
        store.createIndex("updatedAt", "updatedAt");
      }
    };

    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("Could not open project storage."));
  });
}

function requestToPromise<T>(request: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("IndexedDB request failed."));
  });
}

export async function saveProject(project: Project): Promise<void> {
  await syncProject(project);
  const db = await openDatabase();
  try {
    const transaction = db.transaction(STORE_NAME, "readwrite");
    const store = transaction.objectStore(STORE_NAME);
    await requestToPromise(store.put(project));
  } finally {
    db.close();
  }
}

export async function getProject(projectId: string): Promise<Project | null> {
  const db = await openDatabase();
  try {
    const transaction = db.transaction(STORE_NAME, "readonly");
    const store = transaction.objectStore(STORE_NAME);
    const project = await requestToPromise(store.get(projectId));
    const response = await fetch(apiUrl(`projects/${encodeURIComponent(projectId)}`));
    if (response.status === 404) {
      const deleted = await backendJson(await fetch(apiUrl("projects/deleted/ids"))) as string[];
      return project && !deleted.includes(projectId) ? normalizeProject(project) : null;
    }
    const record = await backendJson(response);
    if (project && (!record.details?.id || project.updatedAt >= record.details.updatedAt)) return normalizeProject(project);
    return record.details?.id ? normalizeProject(record.details) : null;
  } finally {
    db.close();
  }
}

export async function listProjects(): Promise<Project[]> {
  const db = await openDatabase();
  try {
    const transaction = db.transaction(STORE_NAME, "readonly");
    const store = transaction.objectStore(STORE_NAME);
    const cached = (await requestToPromise(store.getAll())).map(normalizeProject);
    const deleted = new Set(await backendJson(await fetch(apiUrl("projects/deleted/ids"))) as string[]);
    const local = cached.filter((project) => !deleted.has(project.id));
    for (const project of cached.filter((item) => deleted.has(item.id))) {
      await requestToPromise(db.transaction(STORE_NAME, "readwrite").objectStore(STORE_NAME).delete(project.id));
    }
    const remote = await backendJson(await fetch(apiUrl("projects"))) as { id: string; details: Project }[];
    const projects = remote.filter((row) => row.details?.id).map((row) => normalizeProject(row.details));
    for (const project of local) {
      const index = projects.findIndex((item) => item.id === project.id);
      if (index < 0 || project.updatedAt >= projects[index].updatedAt) {
        await syncProject(project);
        if (index < 0) projects.push(project);
        else projects[index] = project;
      }
    }
    return projects.sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));
  } finally {
    db.close();
  }
}

export async function deleteProject(projectId: string): Promise<void> {
  await backendJson(await fetch(apiUrl(`projects/${encodeURIComponent(projectId)}`), { method: "DELETE" }));
  const db = await openDatabase();
  try {
    const transaction = db.transaction(STORE_NAME, "readwrite");
    const store = transaction.objectStore(STORE_NAME);
    await requestToPromise(store.delete(projectId));
  } finally {
    db.close();
  }
}

async function backendJson(response: Response) {
  const payload = await response.json();
  if (!response.ok) throw new Error(typeof payload.detail === "string" ? payload.detail : "Could not save project metadata.");
  return payload;
}

async function syncProject(project: Project): Promise<void> {
  const uploadedFiles = Object.fromEntries(Object.entries(project.data)
    .filter(([, value]) => value instanceof File)
    .map(([key, value]) => {
      const file = value as File;
      return [key, { name: file.name, size: file.size, type: file.type, lastModified: file.lastModified }];
    }));
  const body = JSON.stringify({ ...project, uploadedFiles }, (key, value) => {
    if (["hfKey", "hf_key", "extractionHfKey"].includes(key)) return undefined;
    if (value instanceof Blob) return null;
    return value;
  });
  await backendJson(await fetch(apiUrl(`projects/${encodeURIComponent(project.id)}`), {
    method: "PUT", headers: { "Content-Type": "application/json" }, body
  }));
}
