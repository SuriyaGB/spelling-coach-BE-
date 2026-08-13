import OpenAI from "openai";

let openAICache: OpenAI | null = null;

export function getOpenAIClient(): OpenAI {
  if (!process.env.OPENAI_API_KEY) {
    throw new Error("OPENAI_API_KEY is required.");
  }

  if (!openAICache) {
    openAICache = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY,
      maxRetries: 3,
    });
  }

  return openAICache;
}
