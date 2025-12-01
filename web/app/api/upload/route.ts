import { NextRequest, NextResponse } from "next/server";
import { S3Client, PutObjectCommand, GetObjectCommand } from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";
import crypto from "crypto";

export const runtime = "nodejs";

const s3Client = new S3Client({
  region: process.env.AWS_REGION,
  credentials: process.env.AWS_ACCESS_KEY_ID
    ? {
        accessKeyId: process.env.AWS_ACCESS_KEY_ID!,
        secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY!,
      }
    : undefined,
});

function envRequired(key: string): string {
  const v = process.env[key];
  if (!v) throw new Error(`Missing env var ${key}`);
  return v;
}

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();
    const file = formData.get("file");
    if (!file || !(file instanceof File)) {
      return NextResponse.json({ error: "File is required" }, { status: 400 });
    }

    if (file.type !== "video/mp4") {
      return NextResponse.json(
        { error: "Only mp4 files are supported" },
        { status: 400 }
      );
    }

    const maxBytes =
      Number(process.env.MAX_UPLOAD_BYTES || 0) || 400 * 1024 * 1024; // ~400MB cap
    const arrayBuffer = await file.arrayBuffer();
    if (arrayBuffer.byteLength > maxBytes) {
      return NextResponse.json(
        { error: `File too large (max ${(maxBytes / 1024 / 1024).toFixed(0)}MB)` },
        { status: 413 }
      );
    }

    const bucket = envRequired("UPLOAD_S3_BUCKET");
    const key = `uploads/${Date.now()}-${crypto.randomUUID()}-${file.name.replace(
      /\s+/g,
      "_"
    )}`;

    await s3Client.send(
      new PutObjectCommand({
        Bucket: bucket,
        Key: key,
        Body: Buffer.from(arrayBuffer),
        ContentType: file.type,
      })
    );

    const signedUrl = await getSignedUrl(
      s3Client,
      new GetObjectCommand({ Bucket: bucket, Key: key }),
      { expiresIn: 60 * 60 } // 1 hour
    );

    return NextResponse.json({
      videoUrl: signedUrl,
      key,
      expiresInSeconds: 3600,
    });
  } catch (err: any) {
    console.error("Upload failed", err);
    return NextResponse.json(
      { error: err?.message || "Upload failed" },
      { status: 500 }
    );
  }
}
