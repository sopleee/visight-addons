import { NextRequest, NextResponse } from "next/server";
import { downloadResultZip } from "@/lib/modal";
import AdmZip from "adm-zip";

export const runtime = "nodejs";

export async function GET(
  req: NextRequest,
  { params }: { params: { id: string } }
) {
  const jobId = params.id;

  if (!jobId) {
    return NextResponse.json({ error: "jobId is required" }, { status: 400 });
  }

  try {
    // Download the zip file
    const zipBuffer = await downloadResultZip(jobId);
    const zip = new AdmZip(zipBuffer);
    
    // Find the annotated video file
    const videoEntry = zip.getEntries().find(
      e => e.entryName.endsWith('_annotated.mp4')
    );
    
    if (!videoEntry) {
      return NextResponse.json(
        { error: "Annotated video not found in results" },
        { status: 404 }
      );
    }
    
    // Extract video data as Buffer and convert to Uint8Array
    const videoBuffer = videoEntry.getData();
    const videoData = new Uint8Array(videoBuffer);
    
    return new NextResponse(videoData, {
      headers: {
        'Content-Type': 'video/mp4',
        'Content-Length': videoData.length.toString(),
        'Accept-Ranges': 'bytes',
      },
    });
  } catch (err: any) {
    console.error("Video extraction error", err);
    return NextResponse.json(
      { error: err?.message || "Failed to extract video" },
      { status: 500 }
    );
  }
}