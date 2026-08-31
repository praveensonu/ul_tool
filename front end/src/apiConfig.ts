const DEFAULT_API_URL = "http://localhost:8000";

function withoutTrailingSlash(value: string): string {
  return value.replace(/\/+$/, "");
}

const configuredApiUrl = import.meta.env.VITE_API_URL;
const legacyApiBaseUrl = import.meta.env.VITE_API_BASE_URL;
const apiOrigin = withoutTrailingSlash(
  configuredApiUrl === undefined ? DEFAULT_API_URL : configuredApiUrl
);

// VITE_API_BASE_URL remains a compatibility fallback for older deployments.
export const API_BASE_URL =
  configuredApiUrl === undefined && legacyApiBaseUrl
    ? withoutTrailingSlash(legacyApiBaseUrl)
    : `${apiOrigin}/api`;

export function apiUrl(path: string): string {
  return `${API_BASE_URL}/${path.replace(/^\/+/, "")}`;
}
