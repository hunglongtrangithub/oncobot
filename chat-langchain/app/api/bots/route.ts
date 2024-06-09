import { NextResponse } from "next/server";
import fs from "fs/promises";
import path from "path";

let cachedBots: string[] | null = null;

const getBots = async () => {
  if (cachedBots) {
    console.log("Returning cached bots");
    return cachedBots;
  }

  const botsDir = path.resolve(process.cwd(), "public", "bots");
  try {
    const botFiles = await fs.readdir(botsDir);
    const botNames = botFiles
      .map((file) => path.parse(file).name)
      .filter((value, index, self) => self.indexOf(value) === index);

    const bots = botNames.filter((botName) => {
      const hasMp3 = botFiles.includes(`${botName}.mp3`);
      const hasJpg = botFiles.includes(`${botName}.jpg`);
      return hasMp3 && hasJpg;
    });

    cachedBots = bots;
    console.log("Cached bots.");
    return bots;
  } catch (error) {
    console.error("Error reading bots directory:", error);
    return [];
  }
};

export async function GET() {
  const bots = await getBots();
  return NextResponse.json({ bots });
}
