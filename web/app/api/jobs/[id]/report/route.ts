import { NextRequest, NextResponse } from "next/server";
import PDFDocument from "pdfkit/js/pdfkit.standalone.js";
import path from "path";
import fs from "fs";
import { fetchInferencePayload } from "@/lib/modal";
import { computeBrandMetrics } from "@/lib/results";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

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
    const pdfBuffer = await createPdf(metrics, payload.summary_stats, jobId);

    return new NextResponse(pdfBuffer as any, {
      status: 200,
      headers: {
        "Content-Type": "application/pdf",
        "Content-Disposition": `attachment; filename=\"${metrics.brand.replace(
          /\s+/g,
          "_"
        )}_report_${jobId}.pdf\"`,
      },
    });
  } catch (err: any) {
    console.error("Report error", err);
    return NextResponse.json(
      { error: err?.message || "Failed to generate report" },
      { status: 500 }
    );
  }
}

async function createPdf(
  metrics: ReturnType<typeof computeBrandMetrics>,
  summaryStats: any,
  jobId: string
): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const doc = new PDFDocument({ margin: 50 });
    const chunks: Buffer[] = [];
    doc.on("data", (c) => chunks.push(Buffer.from(c)));
    doc.on("end", () => resolve(Buffer.concat(chunks)));
    doc.on("error", reject);

    try {
      const fontPath = path.join(process.cwd(), "public", "fonts", "Helvetica.ttf");
      if (fs.existsSync(fontPath)) {
        doc.registerFont("BrandFont", fontPath);
        doc.font("BrandFont");
      }
    } catch (e) {
      console.warn("Custom font registration failed:", e);
      // fall back to default
    }

    doc
      .fontSize(20)
      .fillColor("#0B6BFF")
      .text("Visight Brand Exposure Report", { align: "left" });
    doc.moveDown(0.5);
    doc
      .fontSize(12)
      .fillColor("#333")
      .text(`Brand: ${metrics.brand}`)
      .text(`Job ID: ${jobId}`)
      .text(
        `Video: ${metrics.videoResolution}, duration ${metrics.videoDurationSeconds.toFixed(
          1
        )}s @ ${metrics.fps.toFixed(2)} fps`
      );

    doc.moveDown();
    doc.fontSize(14).fillColor("#111").text("Key Metrics", { underline: true });
    doc.moveDown(0.5);

    const rows = [
      ["Frames processed", metrics.totalFrames],
      ["Frames with brand", metrics.framesWithBrand],
      ["Detections", metrics.detectionCount],
      [
        "On-screen time",
        `${metrics.totalOnScreenSeconds.toFixed(2)}s (${
          metrics.detectionRate * 100
        }% of frames)`,
      ],
      [
        "Avg coverage",
        `${metrics.avgCoveragePercent.toFixed(2)}% (peak ${metrics.peakCoveragePercent.toFixed(
          2
        )}%)`,
      ],
      [
        "First seen",
        metrics.firstSeenSeconds !== undefined
          ? `${metrics.firstSeenSeconds.toFixed(2)}s`
          : "—",
      ],
      [
        "Last seen",
        metrics.lastSeenSeconds !== undefined
          ? `${metrics.lastSeenSeconds.toFixed(2)}s`
          : "—",
      ],
    ];

    rows.forEach(([label, value]) => {
      doc.fontSize(12).fillColor("#444").text(`${label}: ${value}`);
    });

    doc.moveDown();
    doc.fontSize(14).fillColor("#111").text("Overall Summary", {
      underline: true,
    });
    doc.moveDown(0.5);
    if (summaryStats && typeof summaryStats === "object") {
      const detectedClasses =
        (summaryStats.class_counts &&
          Object.keys(summaryStats.class_counts as Record<string, number>)) ||
        [];
      doc
        .fontSize(12)
        .fillColor("#444")
        .text(
          `Classes detected: ${
            detectedClasses.length ? detectedClasses.join(", ") : "n/a"
          }`
        );
    } else {
      doc.fontSize(12).fillColor("#444").text("Summary unavailable");
    }

    doc.end();
  });
}
