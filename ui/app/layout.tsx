import "./globals.css";
import type { Metadata } from "next";
import { Inter } from "next/font/google";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Medical Chatbot",
  description: "Chatbot for Medical Question Answering",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="h-full">
      <body className={`${inter.className} h-full`}>
        <div className="relative z-0 flex h-full w-full overflow-hidden">
          {children}
        </div>
      </body>
    </html>
  );
}
