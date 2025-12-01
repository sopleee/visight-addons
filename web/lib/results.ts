import { BrandName } from "./brands";

export type Detection = {
  bbox: [number, number, number, number];
  confidence: number;
  class_name: string;
  class_id?: number;
};

export type FrameResult = {
  frame_id: string;
  frame_number: number;
  timestamp: number;
  detections: Detection[];
  detection_count: number;
  annotated_frame_path?: string;
};

export type VideoInfo = {
  fps: number;
  total_frames: number;
  width: number;
  height: number;
  duration_seconds: number;
  resolution: string;
};

export type InferenceResultsPayload = {
  video_id: string;
  video_info: VideoInfo;
  video_s3_path?: string | null;
  confidence_threshold: number;
  total_frames: number;
  frames: Array<{
    frame_id: string;
    frame_number: number;
    timestamp: number;
    file_path?: string;
  }>;
  inference_results: FrameResult[];
  summary_stats: Record<string, unknown>;
};

export type BrandMetrics = {
  brand: string;
  videoId: string;
  fps: number;
  videoResolution: string;
  videoDurationSeconds: number;
  totalFrames: number;
  framesWithBrand: number;
  detectionCount: number;
  detectionRate: number;
  totalOnScreenSeconds: number;
  avgCoveragePercent: number;
  peakCoveragePercent: number;
  firstSeenSeconds?: number;
  lastSeenSeconds?: number;
};

export function computeBrandMetrics(
  payload: InferenceResultsPayload,
  brand: BrandName | string
): BrandMetrics {
  const fps = payload.video_info?.fps ?? 0;
  const totalFrames =
    payload.total_frames ??
    payload.video_info?.total_frames ??
    payload.inference_results?.length ??
    0;
  const frameArea =
    (payload.video_info?.width ?? 0) * (payload.video_info?.height ?? 0);

  const framesWithBrand = payload.inference_results.filter((frame) =>
    frame.detections.some((d) => d.class_name === brand)
  );

  const detectionCount = framesWithBrand.reduce((sum, frame) => {
    return (
      sum + frame.detections.filter((d) => d.class_name === brand).length
    );
  }, 0);

  const coverageFractions = framesWithBrand.map((frame) => {
    if (!frameArea) return 0;
    const area = frame.detections
      .filter((d) => d.class_name === brand)
      .reduce((acc, det) => {
        const [x1, y1, x2, y2] = det.bbox;
        const w = Math.max(0, x2 - x1);
        const h = Math.max(0, y2 - y1);
        return acc + w * h;
      }, 0);
    return frameArea > 0 ? area / frameArea : 0;
  });

  const avgCoveragePercent =
    coverageFractions.length > 0
      ? (coverageFractions.reduce((a, b) => a + b, 0) /
          coverageFractions.length) *
        100
      : 0;
  const peakCoveragePercent =
    coverageFractions.length > 0
      ? Math.max(...coverageFractions) * 100
      : 0;

  const totalOnScreenSeconds =
    fps > 0 ? framesWithBrand.length / fps : framesWithBrand.length;
  const detectionRate =
    totalFrames > 0 ? framesWithBrand.length / totalFrames : 0;

  const timestamps = framesWithBrand.map((f) => f.timestamp);
  const firstSeenSeconds = timestamps.length ? Math.min(...timestamps) : undefined;
  const lastSeenSeconds = timestamps.length ? Math.max(...timestamps) : undefined;

  return {
    brand,
    videoId: payload.video_id,
    fps,
    videoResolution: payload.video_info?.resolution ?? "unknown",
    videoDurationSeconds: payload.video_info?.duration_seconds ?? 0,
    totalFrames,
    framesWithBrand: framesWithBrand.length,
    detectionCount,
    detectionRate,
    totalOnScreenSeconds,
    avgCoveragePercent,
    peakCoveragePercent,
    firstSeenSeconds,
    lastSeenSeconds,
  };
}
