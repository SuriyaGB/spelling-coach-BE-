import { getOpenAIClient } from "./openaiClient.js";
import { getWordByText } from "./wordCatalog.js";

const DEFAULT_TTS_MODEL = "gpt-4o-mini-tts";
const DEFAULT_TTS_VOICE = "alloy";

const audioCache = new Map<string, Promise<Uint8Array>>();

function isPronunciationCacheEnabled(): boolean {
  return process.env.SPELLING_COACH_AUDIO_CACHE !== "off";
}

function isTtsInstructionEnabled(): boolean {
  return process.env.SPELLING_COACH_TTS_INSTRUCTIONS === "on";
}

function extractTargetWord(text: string): string | null {
  const match = text.trim().match(/^spell this word:\s*(.+?)\.?$/i);
  return match?.[1]?.trim() ?? null;
}

function isRiskyPronunciationWord(word: string): boolean {
  const normalized = word.trim().toLowerCase();
  if (!normalized) {
    return false;
  }

  return (
    !/[aeiou]/.test(normalized) ||
    /[bcdfghjklmnpqrstvwxyz]{4,}/.test(normalized) ||
    normalized.startsWith("xyl") ||
    normalized.startsWith("syz") ||
    normalized.startsWith("pt") ||
    normalized.startsWith("ps") ||
    normalized.startsWith("mn") ||
    normalized.startsWith("pn") ||
    normalized.startsWith("cz") ||
    normalized.startsWith("yt") ||
    normalized.startsWith("phth") ||
    ((normalized.match(/y/g) ?? []).length >= 2 && !/[aeiou]/.test(normalized))
  );
}

function getFriendlyPronunciationForWord(word: string): string | null {
  const entry = getWordByText(word);
  const chunks = entry?.phoneme_metadata?.friendly_chunks ?? [];
  if (chunks.length === 0) {
    return null;
  }

  return chunks.join("-");
}

function buildCacheKey(text: string, voice: string, instructions?: string): string {
  return [text.toLowerCase(), voice, instructions ?? ""].join("|");
}

export function buildDefaultTtsInstructions(text: string): string | undefined {
  if (!isTtsInstructionEnabled()) {
    return undefined;
  }

  const targetWord = extractTargetWord(text);
  if (targetWord) {
    const baseInstruction =
      "Read the provided text exactly. Do not omit the target word. Pronounce the target word as a spoken word. Do not spell letters. Do not read it character by character. Say the word once clearly and naturally.";
    const friendlyPronunciation = getFriendlyPronunciationForWord(targetWord);

    if (friendlyPronunciation && isRiskyPronunciationWord(targetWord)) {
      return `${baseInstruction} Pronounce ${targetWord} as ${friendlyPronunciation}.`;
    }

    return baseInstruction;
  }

  return undefined;
}

export async function generateSpeechAudio(
  text: string,
  options: {
    voice?: string;
    instructions?: string;
  } = {},
): Promise<Uint8Array> {
  const voice = options.voice ?? DEFAULT_TTS_VOICE;
  const cacheKey = buildCacheKey(text, voice, options.instructions);
  const useCache = isPronunciationCacheEnabled();
  const cachedAudio = useCache ? audioCache.get(cacheKey) : undefined;

  if (cachedAudio) {
    console.log(`[TTS] Cache hit for ${cacheKey}`);
    return cachedAudio;
  }

  console.log(`[TTS] Generating audio for ${cacheKey}`);
  const audioPromise = (async () => {
    try {
      const client = getOpenAIClient();
      const instructions =
        options.instructions ?? buildDefaultTtsInstructions(text);
      console.log(`[TTS] Calling OpenAI with text: "${text}", model: ${DEFAULT_TTS_MODEL}, voice: ${voice}`);
      
      const startTime = Date.now();
      const response = await client.audio.speech.create({
        model: DEFAULT_TTS_MODEL,
        voice,
        input: text,
        response_format: "mp3",
        instructions,
      });
      console.log(`[TTS] OpenAI returned response in ${Date.now() - startTime}ms. Status: ${response.status}`);

      const arrayBuffer = await response.arrayBuffer();
      console.log(`[TTS] Consumed arrayBuffer. Byte length: ${arrayBuffer.byteLength}`);
      return new Uint8Array(arrayBuffer);
    } catch (err) {
      console.error(`[TTS] Error generating audio for text "${text}":`, err);
      throw err;
    }
  })();

  if (useCache) {
    audioCache.set(cacheKey, audioPromise);
  }
  return audioPromise;
}

export async function generatePronunciationAudio(
  word: string,
  options: {
    voice?: string;
    instructions?: string;
  } = {},
): Promise<Uint8Array> {
  return generateSpeechAudio(`Spell this word: ${word}.`, options);
}
