export const dynamic = "force-dynamic";

import { NextRequest, NextResponse } from "next/server";
import { getJobStatus } from "@/lib/modal";

export const runtime = "nodejs";

export async function GET(
  _req: NextRequest,
  { params }: { params: { id: string } }
) {
  const jobId = params.id;
  if (!jobId) {
    return NextResponse.json({ error: "jobId is required" }, { status: 400 });
  }
  try {
    const status = await getJobStatus(jobId);
    return NextResponse.json({ jobId, status });
  } catch (err: any) {
    console.error("Status error", err);
    return NextResponse.json(
      { error: err?.message || "Failed to get status" },
      { status: 500 }
    );
  }
}
