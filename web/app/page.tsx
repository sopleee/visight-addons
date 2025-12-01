'use client';

import { useEffect, useMemo, useState } from "react";
import { BrandName, brands } from "@/lib/brands";
import { BrandMetrics } from "@/lib/results";

type Status = {
  cur_status?: string;
  cur_status_progress?: number;
  updated_at?: string;
  error?: string;
};

const defaults = {
  fps: 12,
  confidence_threshold: 0.5,
  batch_size: 100,
};

function formatSeconds(seconds: number): string {
  if (!Number.isFinite(seconds)) return "—";
  const minutes = Math.floor(seconds / 60);
  const secs = seconds % 60;
  if (minutes === 0) return `${secs.toFixed(1)}s`;
  return `${minutes}m ${secs.toFixed(0)}s`;
}

function Stat({
  label,
  value,
  hint,
}: {
  label: string;
  value: string | number;
  hint?: string;
}) {
  return (
    <div className="flex flex-col bg-white/5 rounded-xl px-4 py-3 border border-white/10">
      <p className="text-sm text-slate-300">{label}</p>
      <p className="text-xl font-semibold text-white">{value}</p>
      {hint && <p className="text-xs text-slate-400 mt-1">{hint}</p>}
    </div>
  );
}

export default function Page() {
  const [brand, setBrand] = useState<BrandName>(brands[0]);
  const [file, setFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState("");
  const [manualUrl, setManualUrl] = useState("");
  const [jobId, setJobId] = useState("");
  const [status, setStatus] = useState<Status | null>(null);
  const [metrics, setMetrics] = useState<BrandMetrics | null>(null);
  const [uploading, setUploading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [polling, setPolling] = useState(false);
  const [error, setError] = useState("");

  const currentVideoUrl = useMemo(
    () => manualUrl.trim() || videoUrl,
    [manualUrl, videoUrl]
  );

  useEffect(() => {
    if (!jobId) return;
    if (status?.cur_status === "completed") return;
    let timer: NodeJS.Timeout;
    const poll = async () => {
      setPolling(true);
      try {
        const res = await fetch(`/api/jobs/${jobId}`);
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        setStatus(data.status);
      } catch (err: any) {
        setError(err?.message || "Failed to poll status");
      } finally {
        setPolling(false);
      }
    };
    poll();
    timer = setInterval(poll, 4000);
    return () => clearInterval(timer);
  }, [jobId, status?.cur_status]);

  useEffect(() => {
    if (!jobId || status?.cur_status !== "completed" || !brand) return;
    const fetchResult = async () => {
      try {
        const res = await fetch(
          `/api/jobs/${jobId}/result?brand=${encodeURIComponent(brand)}`
        );
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        setMetrics(data.metrics);
      } catch (err: any) {
        setError(err?.message || "Failed to read results");
      }
    };
    fetchResult();
  }, [jobId, status?.cur_status, brand]);

  const handleUpload = async () => {
    if (!file) {
      setError("Please choose an mp4 file (max ~4 minutes)");
      return;
    }
    setError("");
    setUploading(true);
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch("/api/upload", { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Upload failed");
      setVideoUrl(data.videoUrl);
    } catch (err: any) {
      setError(err?.message || "Upload failed");
    } finally {
      setUploading(false);
    }
  };

  const handleSubmitJob = async () => {
    if (!currentVideoUrl) {
      setError("Upload a video or paste a URL first");
      return;
    }
    setError("");
    setSubmitting(true);
    setMetrics(null);
    setStatus(null);
    try {
      const res = await fetch("/api/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          videoUrl: currentVideoUrl,
          brand,
          fps: defaults.fps,
          confidence_threshold: defaults.confidence_threshold,
          batch_size: defaults.batch_size,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Failed to submit job");
      setJobId(data.jobId);
    } catch (err: any) {
      setError(err?.message || "Failed to submit");
    } finally {
      setSubmitting(false);
    }
  };

  const statusLabel = status?.cur_status
    ? status.cur_status.replace(/_/g, " ")
    : "idle";

  return (
    <main className="space-y-8">
      <section className="glass rounded-2xl p-6">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3 mb-4">
          <div>
            <h2 className="text-xl font-semibold text-white">1) Upload</h2>
            <p className="text-sm text-slate-300">
              MP4 up to ~4 minutes. Or paste a direct URL (S3/Drive).
            </p>
          </div>
          {videoUrl && (
            <span className="text-xs text-emerald-300 bg-emerald-500/10 px-3 py-1 rounded-full border border-emerald-500/30">
              Uploaded
            </span>
          )}
        </div>
        <div className="flex flex-col gap-4 md:flex-row">
          <label className="flex-1 border border-dashed border-white/20 rounded-xl p-4 cursor-pointer hover:border-brand-600 transition">
            <input
              type="file"
              accept="video/mp4"
              className="hidden"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
            />
            <p className="text-white font-medium">
              {file ? file.name : "Choose MP4 file"}
            </p>
            <p className="text-xs text-slate-400 mt-1">
              Max ~4 minutes. Uploaded videos are stored temporarily.
            </p>
          </label>
          <button
            onClick={handleUpload}
            disabled={!file || uploading}
            className="md:w-40 bg-brand-600 hover:bg-brand-700 text-white font-semibold rounded-xl px-4 py-3 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {uploading ? "Uploading..." : "Upload"}
          </button>
        </div>
        <div className="mt-4 flex flex-col gap-2">
          <label className="text-sm text-slate-200">Or paste a video URL</label>
          <input
            value={manualUrl}
            onChange={(e) => setManualUrl(e.target.value)}
            placeholder="https://..."
            className="bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-brand-600"
          />
        </div>
      </section>

      <section className="glass rounded-2xl p-6 space-y-4">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3">
          <h2 className="text-xl font-semibold text-white">2) Pick brand</h2>
          <p className="text-sm text-slate-300">
            The detector knows {brands.length} brand classes.
          </p>
        </div>
        <div className="flex flex-col md:flex-row gap-4 md:items-center">
          <select
            value={brand}
            onChange={(e) => setBrand(e.target.value as BrandName)}
            className="bg-white/5 text-white border border-white/10 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-brand-600 flex-1"
          >
            {brands.map((b) => (
              <option key={b} value={b}>
                {b}
              </option>
            ))}
          </select>
          <button
            onClick={handleSubmitJob}
            disabled={submitting || !currentVideoUrl}
            className="bg-brand-600 hover:bg-brand-700 text-white font-semibold rounded-xl px-5 py-3 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {submitting ? "Starting..." : "Start inference"}
          </button>
        </div>
        {currentVideoUrl && (
          <p className="text-xs text-slate-400 break-all">
            Using video: {currentVideoUrl}
          </p>
        )}
      </section>

      <section className="glass rounded-2xl p-6 space-y-4">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3">
          <h2 className="text-xl font-semibold text-white">3) Status</h2>
          {jobId && (
            <span className="text-xs text-slate-300">
              Job ID: <code className="text-white">{jobId}</code>
            </span>
          )}
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <Stat
            label="State"
            value={statusLabel}
            hint={polling ? "Polling..." : status?.updated_at}
          />
          <Stat
            label="Progress"
            value={`${status?.cur_status_progress ?? 0}%`}
            hint="Reported by Modal worker"
          />
          <Stat
            label="Brand"
            value={brand || "—"}
            hint="Applied when computing metrics"
          />
        </div>
      </section>

      <section className="glass rounded-2xl p-6 space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-semibold text-white">4) Report</h2>
          {jobId && status?.cur_status === "completed" && (
            <a
              href={`/api/jobs/${jobId}/report?brand=${encodeURIComponent(
                brand
              )}`}
              className="text-sm text-white bg-white/10 border border-white/20 rounded-lg px-3 py-2 hover:bg-white/20"
              target="_blank"
              rel="noreferrer"
            >
              Download PDF
            </a>
          )}
        </div>
        {status?.cur_status !== "completed" && (
          <p className="text-slate-300 text-sm">
            Metrics will appear once the job finishes.
          </p>
        )}
        {metrics && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <Stat
              label="On-screen time"
              value={formatSeconds(metrics.totalOnScreenSeconds)}
              hint={`${(metrics.detectionRate * 100).toFixed(2)}% of frames`}
            />
            <Stat
              label="Avg coverage"
              value={`${metrics.avgCoveragePercent.toFixed(2)}%`}
              hint={`Peak ${metrics.peakCoveragePercent.toFixed(2)}%`}
            />
            <Stat
              label="Detections"
              value={metrics.detectionCount}
              hint={`Frames: ${metrics.framesWithBrand}/${metrics.totalFrames}`}
            />
            <Stat
              label="First seen"
              value={
                metrics.firstSeenSeconds !== undefined
                  ? formatSeconds(metrics.firstSeenSeconds)
                  : "—"
              }
            />
            <Stat
              label="Last seen"
              value={
                metrics.lastSeenSeconds !== undefined
                  ? formatSeconds(metrics.lastSeenSeconds)
                  : "—"
              }
            />
            <Stat
              label="Video"
              value={metrics.videoResolution}
              hint={`${metrics.videoDurationSeconds.toFixed(1)}s @ ${metrics.fps.toFixed(
                2
              )}fps`}
            />
          </div>
        )}
        {error && (
          <p className="text-sm text-rose-300 bg-rose-500/10 border border-rose-500/20 rounded-lg px-3 py-2">
            {error}
          </p>
        )}
      </section>
    </main>
  );
}
