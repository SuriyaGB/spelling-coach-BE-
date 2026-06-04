import OpenAI from "openai";

const DEFAULT_TTS_MODEL = "gpt-4o-mini-tts";
const DEFAULT_TTS_VOICE = "alloy";

let openAICache: OpenAI | null = null;
const audioCache = new Map<string, Promise<Uint8Array>>();

function isPronunciationCacheEnabled(): boolean {
  return process.env.SPELLING_COACH_AUDIO_CACHE !== "off";
}

function isTtsInstructionEnabled(): boolean {
  return process.env.SPELLING_COACH_TTS_INSTRUCTIONS === "on";
}

function getOpenAIClient(): OpenAI {
  if (!process.env.OPENAI_API_KEY) {
    throw new Error("OPENAI_API_KEY is required for pronunciation audio.");
  }

  if (!openAICache) {
    openAICache = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY,
    });
  }

  return openAICache;
}

function buildCacheKey(word: string, voice: string, instructions?: string): string {
  return [word.toLowerCase(), voice, instructions ?? ""].join("|");
}

export async function generatePronunciationAudio(
  word: string,
  options: {
    voice?: string;
    instructions?: string;
  } = {},
): Promise<Uint8Array> {
  const voice = options.voice ?? DEFAULT_TTS_VOICE;
  const cacheKey = buildCacheKey(word, voice, options.instructions);
  const useCache = isPronunciationCacheEnabled();
  const cachedAudio = useCache ? audioCache.get(cacheKey) : undefined;

  if (cachedAudio) {
    return cachedAudio;
  }

  const audioPromise = (async () => {
    const client = getOpenAIClient();
    const instructions =
      options.instructions ??
      (isTtsInstructionEnabled()
        ? "Say 'Spell this word:' and then pronounce the spelling bee word clearly and naturally."
        : undefined);
    const response = await client.audio.speech.create({
      model: DEFAULT_TTS_MODEL,
      voice,
      input: `Spell this word: ${word}`,
      response_format: "mp3",
      instructions,
    });

    const arrayBuffer = await response.arrayBuffer();
    return new Uint8Array(arrayBuffer);
  })();

  if (useCache) {
    audioCache.set(cacheKey, audioPromise);
  }
  return audioPromise;
}
