import AdmZip from "adm-zip";
import { InferenceResultsPayload } from "./results";

type SubmitPayload = {
  videoUrl: string;
  brand: string;
  fps?: number;
  confidence_threshold?: number;
  batch_size?: number;
  save_to_s3?: boolean;
};

function requiredEnv(key: string): string {
  const val = process.env[key];
  if (!val) {
    throw new Error(`Missing env var ${key}`);
  }
  return val;
}

export async function submitJob(payload: SubmitPayload): Promise<string> {
  const url = requiredEnv("MODAL_SUBMIT_URL");
  const search = payload.save_to_s3 === false ? "?save_to_s3=false" : "";
  const res = await fetch(`${url}${search}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      video_url: payload.videoUrl,
      fps: payload.fps,
      confidence_threshold: payload.confidence_threshold,
      batch_size: payload.batch_size,
      brand: payload.brand,
    }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Submit failed (${res.status}): ${text}`);
  }
  const data = await res.json();
  if (!data.job_id) throw new Error("Missing job_id in submit response");
  return data.job_id as string;
}

export async function getJobStatus(jobId: string): Promise<Record<string, any>> {
  const url = requiredEnv("MODAL_CHECK_STATUS_URL");
  const res = await fetch(`${url}?job_id=${jobId}`, { cache: "no-store" });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Status failed (${res.status}): ${text}`);
  }
  const data = await res.json();
  // check_status currently returns a JSON string; normalize to object
  if (typeof data === "string") {
    try {
      return JSON.parse(data);
    } catch {
      return { cur_status: data };
    }
  }
  return data;
}

export async function downloadResultZip(jobId: string): Promise<Buffer> {
  const url = requiredEnv("MODAL_DOWNLOAD_URL");
  const res = await fetch(`${url}?job_id=${jobId}`, { cache: "no-store" });
  const contentType = (res.headers.get("content-type") || "").toLowerCase();

  if (!res.ok) {
    let text = "";
    try {
      if (contentType.includes("json") || contentType.includes("text")) {
        text = await res.text();
      }
    } catch {
      text = "";
    }
    throw new Error(`Download failed (${res.status}): ${text || "unexpected response"}`);
  }

  if (
    !contentType.includes("zip") &&
    !contentType.includes("octet-stream")
  ) {
    throw new Error(`Unexpected content-type: ${contentType || "unknown"}`);
  }

  const arrayBuffer = await res.arrayBuffer();
  return Buffer.from(arrayBuffer);
}

export function parseResultsJsonFromZip(zipBuffer: Buffer): InferenceResultsPayload {
  const zip = new AdmZip(zipBuffer);
  const entry = zip
    .getEntries()
    .find((e) => e.entryName.toLowerCase().endsWith("results.json"));
  if (!entry) {
    throw new Error("results.json not found in zip");
  }
  const content = entry.getData().toString("utf-8");
  return JSON.parse(content) as InferenceResultsPayload;
}

export async function fetchInferencePayload(
  jobId: string
): Promise<InferenceResultsPayload> {
  const buffer = await downloadResultZip(jobId);
  return parseResultsJsonFromZip(buffer);
}
