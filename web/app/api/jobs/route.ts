import { NextRequest, NextResponse } from "next/server";
import { submitJob } from "@/lib/modal";

export const runtime = "nodejs";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const {
      videoUrl,
      brand,
      fps = 12,
      confidence_threshold = 0.5,
      batch_size = 30,
    } = body || {};

    if (!videoUrl || typeof videoUrl !== "string") {
      return NextResponse.json(
        { error: "videoUrl is required" },
        { status: 400 }
      );
    }
    if (!brand || typeof brand !== "string") {
      return NextResponse.json(
        { error: "brand is required" },
        { status: 400 }
      );
    }

    const jobId = await submitJob({
      videoUrl,
      brand,
      fps,
      confidence_threshold,
      batch_size,
      save_to_s3: false,
    });

    return NextResponse.json({ jobId });
  } catch (err: any) {
    console.error("Submit job error", err);
    return NextResponse.json(
      { error: err?.message || "Failed to submit job" },
      { status: 500 }
    );
  }
}
