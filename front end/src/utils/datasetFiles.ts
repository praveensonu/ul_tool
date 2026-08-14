import type { ProjectPreviewRow } from "../types";

type LocalRow = ProjectPreviewRow;

function extensionOf(file: File) {
  const name = file.name.toLowerCase();
  const dot = name.lastIndexOf(".");
  return dot >= 0 ? name.slice(dot + 1) : "";
}

function parseCsv(text: string): Record<string, unknown>[] {
  const rows: string[][] = [];
  let currentRow: string[] = [];
  let currentField = "";
  let quoted = false;

  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];

    if (quoted) {
      if (char === '"') {
        if (text[index + 1] === '"') {
          currentField += '"';
          index += 1;
        } else {
          quoted = false;
        }
      } else {
        currentField += char;
      }
      continue;
    }

    if (char === '"') {
      quoted = true;
    } else if (char === ",") {
      currentRow.push(currentField);
      currentField = "";
    } else if (char === "\n") {
      currentRow.push(currentField.replace(/\r$/, ""));
      rows.push(currentRow);
      currentRow = [];
      currentField = "";
    } else {
      currentField += char;
    }
  }

  if (currentField.length > 0 || currentRow.length > 0) {
    currentRow.push(currentField.replace(/\r$/, ""));
    rows.push(currentRow);
  }

  if (rows.length === 0) return [];
  const headers = rows[0].map((header) => header.replace(/^\uFEFF/, "").trim());

  return rows
    .slice(1)
    .filter((row) => row.some((value) => value.trim().length > 0))
    .map((row) =>
      Object.fromEntries(headers.map((header, index) => [header, row[index] ?? ""]))
    );
}

function parseJson(text: string): Record<string, unknown>[] {
  const value = JSON.parse(text) as unknown;
  if (Array.isArray(value)) {
    return value.filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === "object");
  }

  if (value && typeof value === "object") {
    const object = value as Record<string, unknown>;
    if (Array.isArray(object.data)) {
      return object.data.filter(
        (item): item is Record<string, unknown> => Boolean(item) && typeof item === "object"
      );
    }
    return [object];
  }

  throw new Error("JSON dataset must contain objects or an array of objects.");
}

function parseJsonl(text: string): Record<string, unknown>[] {
  return text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line, index) => {
      const value = JSON.parse(line) as unknown;
      if (!value || typeof value !== "object" || Array.isArray(value)) {
        throw new Error(`JSONL row ${index + 1} is not an object.`);
      }
      return value as Record<string, unknown>;
    });
}

function stringValue(record: Record<string, unknown>, names: string[]) {
  for (const name of names) {
    const value = record[name];
    if (value !== undefined && value !== null) return String(value);
  }
  return "";
}

function toPreviewRows(records: Record<string, unknown>[]): LocalRow[] {
  return records.map((raw, index) => ({
    key: String(index),
    id: stringValue(raw, ["id", "ID", "Id", "uid", "uuid"]) || String(index + 1),
    question: stringValue(raw, ["question"]),
    answer: stringValue(raw, ["answer"]),
    raw
  }));
}

export async function parseDatasetFile(file: File): Promise<LocalRow[]> {
  const extension = extensionOf(file);
  if (extension === "parquet") {
    throw new Error("Parquet preview/filtering is not available in the browser yet.");
  }

  const text = await file.text();
  let records: Record<string, unknown>[];

  if (extension === "csv") records = parseCsv(text);
  else if (extension === "json") records = parseJson(text);
  else if (extension === "jsonl" || extension === "ndjson") records = parseJsonl(text);
  else throw new Error("Use CSV, JSON, JSONL, or Parquet datasets.");

  if (records.length === 0) throw new Error("The selected dataset has no rows.");
  return toPreviewRows(records);
}

function stableRecordKey(record: Record<string, unknown>) {
  const sorted = Object.keys(record)
    .sort()
    .reduce<Record<string, unknown>>((result, key) => {
      result[key] = record[key];
      return result;
    }, {});
  return JSON.stringify(sorted);
}

function rowIdentityKeys(row: LocalRow) {
  const keys: string[] = [];
  const explicitId = stringValue(row.raw, ["id", "ID", "Id", "uid", "uuid"]);
  if (explicitId) keys.push(`id:${explicitId}`);
  if (row.question || row.answer) keys.push(`qa:${row.question}\u0000${row.answer}`);
  keys.push(`raw:${stableRecordKey(row.raw)}`);
  return keys;
}

export function rowsToJsonlFile(rows: LocalRow[], filename: string) {
  const content = rows.map((row) => JSON.stringify(row.raw)).join("\n") + "\n";
  return new File([content], filename, { type: "application/x-ndjson" });
}

export async function extractForgetAndRetain(fullFile: File, poisonFile: File) {
  const fullRows = await parseDatasetFile(fullFile);
  const poisonRows = await parseDatasetFile(poisonFile);
  const poisonKeys = new Set(poisonRows.flatMap(rowIdentityKeys));

  const forgetRows = fullRows.filter((row) => rowIdentityKeys(row).some((key) => poisonKeys.has(key)));
  const retainRows = fullRows.filter((row) => !rowIdentityKeys(row).some((key) => poisonKeys.has(key)));

  if (forgetRows.length === 0) {
    throw new Error(
      "No poison rows matched the full dataset. Matching uses id when available, otherwise question + answer."
    );
  }

  return {
    forgetRows,
    retainRows,
    forgetFile: rowsToJsonlFile(forgetRows, "forget_extracted.jsonl"),
    retainFile: rowsToJsonlFile(retainRows, "retain_extracted.jsonl")
  };
}

export function selectedRows(rows: LocalRow[], selectedKeys: string[]) {
  const selected = new Set(selectedKeys);
  return rows.filter((row) => selected.has(row.key));
}

export function backendPreviewToRows(rows: Array<Record<string, unknown>>): LocalRow[] {
  return rows.map((raw, index) => ({
    key: String(index),
    id: stringValue(raw, ["id", "ID", "Id", "uid", "uuid"]) || String(index + 1),
    question: stringValue(raw, ["question"]),
    answer: stringValue(raw, ["answer"]),
    raw
  }));
}
