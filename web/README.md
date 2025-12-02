# Visight Web UI

A lightweight Next.js app to submit inference jobs, poll status, view metrics, and download reports.

## Prerequisites
- Node.js 20+ (use `nvm install 20 && nvm use 20` if needed)
- Access to your deployed Modal endpoints (`submit_job`, `check_status`, `download_result`)
- (Optional) AWS creds + bucket if you want to upload videos via the UI instead of pasting URLs

## Setup
```bash
cd web
npm install
```
Create `.env.local`:
- `MODAL_SUBMIT_URL`, `MODAL_CHECK_STATUS_URL`, `MODAL_DOWNLOAD_URL`: your deployed Modal URL endpoints.
- `UPLOAD_S3_BUCKET`, `AWS_REGION`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`: required only if you use the upload button; paste video URLs to skip these.
- Optional: adjust `MAX_UPLOAD_BYTES`.

## Run dev server
```bash
npm run dev
```
Open http://localhost:3000.

## How it works
- Frontend page: `app/page.tsx`
  - Upload mp4 or paste URL
  - Select brand (from `lib/brands.ts`)
  - Start inference (POST to `/api/jobs`)
  - Poll status, show metrics, download PDF
- API routes:
  - `/api/jobs` → calls Modal `submit_job` (`lib/modal.ts`)
  - `/api/jobs/[id]` → polls Modal `check_status`
  - `/api/jobs/[id]/result` → downloads Modal `download_result`, parses `results.json`, computes brand metrics
  - `/api/jobs/[id]/report` → builds PDF from metrics
  - `/api/upload` → uploads video to S3 (optional path; requires creds)
- Modal client helpers: `lib/modal.ts`
- Metrics helpers: `lib/results.ts`

## Adjusting defaults
- Default inference params (fps, confidence, batch) are in `app/page.tsx` (`const defaults`).
- To change the brand list, edit `lib/brands.ts`.

## Common commands
- Start dev: `npm run dev`
- Type check/lint: `npm run lint`

## Notes
- Upload API loads files into memory; suitable for short clips (~4 minutes). Paste URLs for larger sources.
- `.env.local` is ignored by git; each teammate should set their own Modal/S3 values.
