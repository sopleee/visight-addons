import "./globals.css";
import type { ReactNode } from "react";

export const metadata = {
  title: "Visight Brand Exposure",
  description: "Upload a race clip, pick your brand, and get a report.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen">
        <div className="max-w-6xl mx-auto px-6 py-10">
          <header className="flex items-center justify-between mb-10">
            <div className="flex items-center gap-3">
              <div className="h-10 w-10 rounded-full bg-gradient-to-br from-brand-600 to-brand-700" />
              <div>
                <p className="text-sm uppercase tracking-[0.2em] text-slate-400">
                  Visight
                </p>
                <h1 className="text-2xl font-semibold text-white">
                  Brand Exposure Lab
                </h1>
              </div>
            </div>
          </header>
          {children}
        </div>
      </body>
    </html>
  );
}
