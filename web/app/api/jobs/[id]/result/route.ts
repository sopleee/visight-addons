import { NextRequest, NextResponse } from "next/server";
import { fetchInferencePayload } from "@/lib/modal";
import { computeBrandMetrics } from "@/lib/results";

export const runtime = "nodejs";

export async function GET(
  req: NextRequest,
  { params }: { params: { id: string } }
) {
  const jobId = params.id;
  const brand = new URL(req.url).searchParams.get("brand");

  if (!jobId) {
    return NextResponse.json({ error: "jobId is required" }, { status: 400 });
  }
  if (!brand) {
    return NextResponse.json({ error: "brand is required" }, { status: 400 });
  }

  try {
    const payload = await fetchInferencePayload(jobId);
    const metrics = computeBrandMetrics(payload, brand);

    return NextResponse.json({
      jobId,
      brand,
      metrics,
      video: payload.video_info,
      summary: payload.summary_stats,
    });
  } catch (err: any) {
    console.error("Result error", err);
    return NextResponse.json(
      { error: err?.message || "Failed to read results" },
      { status: 500 }
    );
  }
}
