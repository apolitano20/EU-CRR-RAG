import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "EU CRR Legal Research",
  description:
    "AI-powered regulatory compliance Q&A for EU Capital Requirements Regulation (CRR – Regulation (EU) No 575/2013)",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={`${inter.className} h-screen overflow-hidden bg-slate-50`}>
        {children}
      </body>
    </html>
  );
}
