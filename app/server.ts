import "dotenv/config";
import "./instrument.js";
import * as Sentry from "@sentry/node";

import { createServer } from "node:http";
import { URL } from "node:url";
import { randomUUID } from "node:crypto";
import { authenticateRequest } from "./auth.js";
import {
  fetchCustomListsFromDB,
  fetchCustomListByIdFromDB,
  saveCustomListToDB,
  fetchUserProfileFromDB,
  updateUserProfileInDB,
  startPracticeSessionInDB,
  recordWordAttemptInDB,
  endPracticeSessionInDB,
  getUserStatisticsInDB,
  getSessionAttemptsFromDB,
  getPracticeSessionFromDB,
  getUserSubscriptionFromDB,
  updateUserSubscriptionInDB,
  addChallengeToSession,
  getChallengeFromSession,
  peekChallengeInSession,
  type DBCustomList,
} from "./supabase.js";
import {
  buildSpellingCoachInput,
  buildDetailedWordResponse,
  buildWordPrecomputeInputFromWordEntry,
  buildWordPrecomputeInput,
  buildWordResponse,
  CoachingRequestSchema,
  LevelQuerySchema,
  WordSearchQuerySchema,
} from "./inputBuilder.js";
import { CustomWordImportRequestSchema, importCustomWords } from "./customWordImport.js";
import {
  ForeignOriginImportRequestSchema,
  importForeignOriginWords,
} from "./foreignOriginImport.js";
import {
  FileImportRequestError,
  MAX_IMPORT_REQUEST_BYTES,
  handleImportFileRequest,
  getImportJob,
} from "./fileImportHandler.js";
import { getConfiguredModelName } from "./modelConfig.js";
import { isNewDeterministicPatternMatcherEnabled } from "./newPatternMatcher.js";
import { hasWordTeachingPrecompute, runSplitSpellingCoachAgent, warmWordTeachingPrecompute } from "./optimizedCoach.js";
import { generatePronunciationAudio } from "./pronunciation.js";
import { isNextStepEnabled, isRuntimeConceptTeachingEnabled } from "./prompt.js";
import {
  isSpellingRulePromptHintsEnabled,
  isSpellingRuleShortlistEnabled,
} from "./referenceData.js";
import { runSpellingCoachAgent } from "./runAgent.js";
import {
  readSpellingCoachStreamRequest,
  SpellingCoachStreamRequestError,
  streamSpellingCoach,
} from "./streamingCoach.js";
import { recordSpellingCoachTrace, recordImportListTrace } from "./langfuse.js";
import { MockBeeService } from "./mockBee.js";
import {
  getCustomWordListById,
  getForeignOriginWordListByOrigin,
  getWordByText,
  listCustomWordListsForUser,
  listForeignOrigins,
  pickNextWord,
  searchWords,
} from "./wordCatalog.js";
import { logError, logInfo } from "./logging.js";
import {
  buildVoiceResponse,
  interpretVoiceUtterance,
  transcribeAudio,
  VoiceInterpretRequestSchema,
  VoiceRespondRequestSchema,
} from "./voice.js";
import Stripe from "stripe";
let stripeInstance: Stripe | null = null;
function getStripe(): Stripe {
  if (!stripeInstance) {
    const key = process.env.STRIPE_SECRET_KEY;
    if (!key) {
      throw new Error("STRIPE_SECRET_KEY is not configured on the server.");
    }
    stripeInstance = new Stripe(key);
  }
  return stripeInstance;
}

function getFrontendReturnUrl(request: import("node:http").IncomingMessage): string {
  const fallbackBaseUrl = process.env.APP_BASE_URL || "http://localhost:8080";
  const referer = request.headers.referer || request.headers.origin || fallbackBaseUrl;
  return referer.split("?")[0].split("#")[0];
}

const PORT = Number(process.env.PORT ?? 3000);
const mockBeeService = new MockBeeService();
const STANDARD_FREE_WORD_LIMIT = Number(process.env.STANDARD_FREE_WORD_LIMIT ?? 30);

function isPremiumSubscriptionStatus(status?: string | null): boolean {
  return status === "active" || status === "trialing";
}

function subscriptionIsActive(
  subscription: Awaited<ReturnType<typeof getUserSubscriptionFromDB>>,
): boolean {
  if (!subscription || !isPremiumSubscriptionStatus(subscription.status)) {
    return false;
  }

  if (!subscription.current_period_end) return false;

  return new Date(subscription.current_period_end).getTime() > Date.now();
}

async function hasPremiumAccess(authToken: string, userId: string): Promise<boolean> {
  return subscriptionIsActive(await getUserSubscriptionFromDB(authToken, userId));
}

async function getStandardWordsUsed(authToken: string, userId: string): Promise<number> {
  const stats = await getUserStatisticsInDB(authToken, userId);
  return stats
    .filter((row) => row.mode === "standard" || row.mode?.startsWith("standard_level_"))
    .reduce((total, row) => total + (row.total_attempts || 0), 0);
}

function isStandardMode(mode: string): boolean {
  return mode === "standard" || mode.startsWith("standard_level_");
}

function sendUpgradeRequired(response: import("node:http").ServerResponse): void {
  sendJson(response, 402, {
    error: "Premium subscription required.",
    code: "SUBSCRIPTION_REQUIRED",
  });
}

function sendJson(response: import("node:http").ServerResponse, statusCode: number, body: unknown) {
  response.writeHead(statusCode, {
    "Content-Type": "application/json; charset=utf-8",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization, x-audio-filename",
  });
  response.end(`${JSON.stringify(body, null, 2)}\n`);
}

function sendAudio(
  response: import("node:http").ServerResponse,
  statusCode: number,
  audio: Uint8Array,
  options: {
    cacheControl?: string;
  } = {},
): void {
  response.writeHead(statusCode, {
    "Content-Type": "audio/mpeg",
    "Content-Length": audio.byteLength,
    "Cache-Control": options.cacheControl ?? "public, max-age=3600",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization, x-audio-filename",
  });
  response.end(Buffer.from(audio));
}

function collectBody(request: import("node:http").IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    let body = "";
    request.on("data", (chunk) => {
      body += chunk;
    });
    request.on("end", () => resolve(body));
    request.on("error", reject);
  });
}

function collectBinaryBody(
  request: import("node:http").IncomingMessage,
  maxBytes = Number.POSITIVE_INFINITY,
): Promise<Uint8Array> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let totalBytes = 0;
    let settled = false;
    request.on("data", (chunk) => {
      if (settled) return;
      const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
      totalBytes += buffer.length;
      if (totalBytes > maxBytes) {
        settled = true;
        reject(new FileImportRequestError("Uploaded file is too large.", 413));
        request.resume();
        return;
      }
      chunks.push(buffer);
    });
    request.on("end", () => {
      if (!settled) resolve(new Uint8Array(Buffer.concat(chunks)));
    });
    request.on("error", (error) => {
      if (!settled) reject(error);
    });
  });
}

function isAuthError(error: unknown): boolean {
  return (
    error instanceof Error &&
    (error.message.startsWith("Unauthorized:") ||
      error.message.startsWith("Supabase auth is not configured."))
  );
}

export default async function handler(
  request: import("node:http").IncomingMessage,
  response: import("node:http").ServerResponse,
) {
  if (!request.url || !request.method) {
    sendJson(response, 400, { error: "Invalid request." });
    return;
  }

  if (request.method === "OPTIONS") {
    sendJson(response, 204, {});
    return;
  }

  const url = new URL(request.url, `http://localhost:${PORT}`);
  const requestStart = performance.now();
  response.on("finish", () => {
    logInfo(
      `[spelling-coach api] ${request.method} ${url.pathname}${url.search} status=${response.statusCode} total=${(performance.now() - requestStart).toFixed(1)}ms`,
    );
  });

  try {
    if (request.method === "GET" && url.pathname === "/api/health") {
      const runtime =
        process.env.SPELLING_COACH_RUNTIME === "direct"
          ? "direct"
          : "deep_agent";
      const audioCaching =
        process.env.SPELLING_COACH_AUDIO_CACHE === "off" ? "off" : "on";
      const ttsInstructions =
        process.env.SPELLING_COACH_TTS_INSTRUCTIONS === "on" ? "on" : "off";
      const spellingRuleShortlist = isSpellingRuleShortlistEnabled()
        ? "on"
        : "off";
      const spellingRulePromptHints = isSpellingRulePromptHintsEnabled()
        ? "on"
        : "off";
      const newDeterministicPatternMatcher = isNewDeterministicPatternMatcherEnabled()
        ? "on"
        : "off";
      const nextStep = isNextStepEnabled() ? "on" : "off";
      const runtimeConceptTeaching = isRuntimeConceptTeachingEnabled()
        ? "on"
        : "off";

      sendJson(response, 200, {
        ok: true,
        runtime,
        model: getConfiguredModelName(),
        featureFlags: {
          audioCaching,
          ttsInstructions,
          spellingRuleShortlist,
          spellingRulePromptHints,
          newDeterministicPatternMatcher,
          nextStep,
          runtimeConceptTeaching,
        },
        envConfigured: {
          OPENAI_API_KEY: Boolean(process.env.OPENAI_API_KEY),
          SENTRY_DSN: Boolean(process.env.SENTRY_DSN),
          SUPABASE_URL: Boolean(process.env.SUPABASE_URL),
          SUPABASE_PUBLISHABLE_KEY: Boolean(process.env.SUPABASE_PUBLISHABLE_KEY),
          STRIPE_SECRET_KEY: Boolean(process.env.STRIPE_SECRET_KEY),
          STRIPE_PRICE_ID: Boolean(process.env.STRIPE_PRICE_ID),
          LANGFUSE_SECRET_KEY: Boolean(process.env.LANGFUSE_SECRET_KEY),
          LANGFUSE_PUBLIC_KEY: Boolean(process.env.LANGFUSE_PUBLIC_KEY),
          LANGFUSE_BASE_URL: Boolean(process.env.LANGFUSE_BASE_URL),
        },
      });
      return;
    }

    if (request.method === "GET" && url.pathname === "/api/words/next") {
      const query = LevelQuerySchema.parse({
        level: url.searchParams.get("level"),
        customListId: url.searchParams.get("customListId") ?? undefined,
        foreignOrigin: url.searchParams.get("foreignOrigin") ?? undefined,
        exclude: url.searchParams.get("exclude") ?? undefined,
      });
      const sessionId = url.searchParams.get("sessionId") ?? undefined;
      let customWordsFallback: any[] | undefined;
      // Always authenticate so we can store the challengeId in the session when sessionId is present.
      // For custom lists, authentication was already required.
      const user = (query.customListId || sessionId)
        ? await authenticateRequest(request)
        : undefined;

      if (query.customListId && user) {
        const authHeader = request.headers.authorization || "";
        const dbList = await fetchCustomListByIdFromDB(authHeader, query.customListId, user.id);
        if (dbList) {
          customWordsFallback = dbList.words;
        }
      }

      let excludeList = query.exclude || [];
      if (sessionId && user) {
        const authHeader = request.headers.authorization || "";
        try {
          const attempts = await getSessionAttemptsFromDB(authHeader, user.id, sessionId);
          if (attempts && attempts.length > 0) {
            const dbExclude = attempts.map((a: any) => a.target_word);
            excludeList = [...new Set([...excludeList, ...dbExclude])];
          }
        } catch (err) {
          logError("Failed to fetch session attempts for exclude logic:", err);
        }
      }

      const word = pickNextWord(
        query.level,
        excludeList,
        query.customListId,
        query.foreignOrigin,
        user?.id,
        customWordsFallback,
      );
      const precomputeInput = buildWordPrecomputeInputFromWordEntry(word);
      const precomputeStart = performance.now();
      void warmWordTeachingPrecompute(precomputeInput)
        .then(() => {
          logInfo(
            `[spelling-coach precompute timing] word="${word.word}" total=${(performance.now() - precomputeStart).toFixed(1)}ms`,
          );
        })
        .catch((error) => {
          logError("Word teaching precompute failed:", error);
        });

      // Generate a challengeId and store the word securely in the session.
      // A sessionId is required for all authenticated practice.
      if (!sessionId || !user) {
        sendJson(response, 400, { error: "sessionId is required to fetch a word." });
        return;
      }

      const authHeader = request.headers.authorization || "";
      const challengeId = randomUUID();
      try {
        await addChallengeToSession(authHeader, user.id, sessionId, challengeId, word.word);
      } catch (err) {
        logError("Failed to store challengeId in session:", err);
        sendJson(response, 500, { error: "Could not prepare word challenge. Please try again." });
        return;
      }

      const wordResponse = buildWordResponse(word);
      // The plain word field is intentionally omitted — challengeId is the only reference.
      const { word: _omitted, ...wordMetadata } = wordResponse;
      sendJson(response, 200, { ...wordMetadata, challengeId });
      return;
    }

    if (request.method === "GET" && url.pathname === "/api/words/search") {
      const query = WordSearchQuerySchema.parse({
        q: url.searchParams.get("q"),
        mode: url.searchParams.get("mode") ?? undefined,
        limit: url.searchParams.get("limit") ?? undefined,
      });

      const results = searchWords(query.q, query.mode, query.limit).map((word) => ({
        word: word.word,
        level: word.level,
        origin: word.origin,
        partOfSpeech: word.part_of_speech,
      }));

      sendJson(response, 200, {
        query: query.q,
        mode: query.mode,
        limit: query.limit,
        count: results.length,
        results,
      });
      return;
    }

    if (request.method === "GET" && url.pathname.startsWith("/api/words/")) {
      const parts = url.pathname.split("/");
      const encodedWord = parts[3];
      const tail = parts[4];

      if (tail === "pronunciation" || encodedWord === "import-jobs" || encodedWord === "pronunciation") {
        // handled below
      } else if (encodedWord) {
        const word = decodeURIComponent(encodedWord);
        const wordEntry = getWordByText(word);
        if (!wordEntry) {
          sendJson(response, 404, { error: `Unknown word: ${word}` });
          return;
        }

        sendJson(response, 200, buildDetailedWordResponse(wordEntry));
        return;
      }
    }

    //  Create/start a Mock Bee session
    if (request.method === "POST" && url.pathname === "/api/mock-bee/sessions") {
      const rawBody = await collectBody(request);
      const requestBody = JSON.parse(rawBody);

      const user = await authenticateRequest(request);
      const authHeader = request.headers.authorization || "";

      if (!(await hasPremiumAccess(authHeader, user.id))) {
        sendUpgradeRequired(response);
        return;
      }

      let customWordsFallback: DBCustomList["words"] | undefined;

      if (requestBody.wordSource === "custom_list") {
        const customListId = String(requestBody.customListId ?? "");
        const dbList = await fetchCustomListByIdFromDB(authHeader, customListId, user.id);
        if (!dbList) {
          sendJson(response, 404, { error: `Unknown custom list: ${customListId}` });
          return;
        }
        customWordsFallback = dbList.words;
      }

      const result = await mockBeeService.createSession(
        authHeader,
        user.id,
        requestBody,
        {
          ownerUserId: user.id,
          customWordsFallback,
        },
      );
      sendJson(response, 200, result);
      return;
    }

    if (
      url.pathname.startsWith("/api/mock-bee/sessions/") &&
      request.method === "GET"
    ) {
      const parts = url.pathname.split("/").filter(Boolean);
      const sessionId = decodeURIComponent(parts[3] ?? "");
      const tail = parts.slice(4);

      if (!sessionId) {
        sendJson(response, 400, { error: "Mock bee session id is required." });
        return;
      }

      const user = await authenticateRequest(request);
      const authHeader = request.headers.authorization || "";
      if (!(await hasPremiumAccess(authHeader, user.id))) {
        sendUpgradeRequired(response);
        return;
      }
      const session = await mockBeeService.getInternalSession(authHeader, user.id, sessionId);

      if (session.ownerUserId && user.id !== session.ownerUserId) {
        sendJson(response, 403, { error: "Forbidden." });
        return;
      }

      if (tail.length === 0) {
        sendJson(response, 200, {
          session: await mockBeeService.getSession(authHeader, user.id, sessionId),
        });
        return;
      }

      if (tail[0] === "review") {
        sendJson(response, 200, {
          review: await mockBeeService.getReview(authHeader, user.id, sessionId),
        });
        return;
      }

      if (tail[0] === "current-word" && tail[1] === "pronunciation") {
        if (session.status !== "active") {
          sendJson(response, 409, { error: "Mock bee session is already completed." });
          return;
        }

        const word = session.turns[session.currentTurnIndex]?.word;
        if (!word) {
          sendJson(response, 404, { error: "Current mock bee word not found." });
          return;
        }

        const audio = await generatePronunciationAudio(word.word);
        sendAudio(response, 200, audio, {
          cacheControl: "no-store, max-age=0",
        });
        return;
      }
    }

    // spelling test
    if (
      url.pathname.startsWith("/api/mock-bee/sessions/") &&
      request.method === "POST"
    ) {
      const parts = url.pathname.split("/").filter(Boolean);
      const sessionId = decodeURIComponent(parts[3] ?? "");
      const tail = parts.slice(4);

      if (!sessionId) {
        sendJson(response, 400, { error: "Mock bee session id is required." });
        return;
      }

      const user = await authenticateRequest(request);
      const authHeader = request.headers.authorization || "";
      if (!(await hasPremiumAccess(authHeader, user.id))) {
        sendUpgradeRequired(response);
        return;
      }
      const session = await mockBeeService.getInternalSession(authHeader, user.id, sessionId);

      if (session.ownerUserId && user.id !== session.ownerUserId) {
        sendJson(response, 403, { error: "Forbidden." });
        return;
      }

      if (tail[0] === "submit") {
        const rawBody = await collectBody(request);
        const result = await mockBeeService.submitAttempt(
          authHeader,
          user.id,
          sessionId,
          JSON.parse(rawBody),
        );
        sendJson(response, 200, result);
        return;
      }

      if (tail[0] === "timeout") {
        const result = await mockBeeService.timeoutCurrentWord(
          authHeader,
          user.id,
          sessionId,
        );
        sendJson(response, 200, result);
        return;
      }

      if (tail[0] === "end") {
        const sessionView = await mockBeeService.endSession(
          authHeader,
          user.id,
          sessionId,
        );
        sendJson(response, 200, { session: sessionView });
        return;
      }
    }

    if (request.method === "GET" && url.pathname === "/api/custom-lists") {
      const user = await authenticateRequest(request);
      const authHeader = request.headers.authorization || "";
      const lists = await fetchCustomListsFromDB(authHeader, user.id);
      sendJson(response, 200, { lists });
      return;
    }

    if (request.method === "GET" && url.pathname === "/api/auth/me") {
      const user = await authenticateRequest(request);
      sendJson(response, 200, { user });
      return;
    }

    if (request.method === "GET" && url.pathname === "/api/users/profile") {
      const user = await authenticateRequest(request);
      const authHeader = request.headers.authorization || "";
      const profile = await fetchUserProfileFromDB(authHeader, user.id, user.email);
      sendJson(response, 200, { profile });
      return;
    }

    if (request.method === "POST" && url.pathname === "/api/users/profile") {
      const user = await authenticateRequest(request);
      const authHeader = request.headers.authorization || "";
      const rawBody = await collectBody(request);
      const updates = JSON.parse(rawBody);
      const profile = await updateUserProfileInDB(authHeader, user.id, updates);
      sendJson(response, 200, { profile });
      return;
    }

    if (request.method === "GET" && url.pathname === "/api/users/stats") {
      const user = await authenticateRequest(request);
      const authHeader = request.headers.authorization || "";
      const stats = await getUserStatisticsInDB(authHeader, user.id);
      sendJson(response, 200, { stats });
      return;
    }

    if (request.method === "GET" && url.pathname === "/api/sessions/attempts") {
      const user = await authenticateRequest(request);
      const authHeader = request.headers.authorization || "";
      const sessionId = url.searchParams.get("sessionId");
      if (!sessionId) {
        sendJson(response, 400, { error: "Missing sessionId query parameter" });
        return;
      }
      const attempts = await getSessionAttemptsFromDB(authHeader, user.id, sessionId);
      const enrichedAttempts = attempts.map((att) => {
        const wordCatalogEntry = getWordByText(att.target_word);
        return {
          ...att,
          word_catalog_entry: wordCatalogEntry ? buildWordResponse(wordCatalogEntry) : null,
        };
      });
      sendJson(response, 200, { attempts: enrichedAttempts });
      return;
    }

    if (request.method === "GET" && url.pathname === "/api/sessions/current") {
      const user = await authenticateRequest(request);
      const authHeader = request.headers.authorization || "";
      const sessionId = url.searchParams.get("sessionId");
      if (!sessionId) {
        sendJson(response, 400, { error: "Missing sessionId query parameter" });
        return;
      }
      const session = await getPracticeSessionFromDB(authHeader, user.id, sessionId);
      sendJson(response, 200, { session });
      return;
    }

    if (request.method === "POST" && url.pathname === "/api/sessions/start") {
      const user = await authenticateRequest(request);
      const authHeader = request.headers.authorization || "";
      const rawBody = await collectBody(request);
      const {
        mode,
        level,
        forceCloseCurrent,
        originLanguage,
        customListId,
      } = JSON.parse(rawBody);

      const premium = await hasPremiumAccess(authHeader, user.id);
      if (!isStandardMode(mode) && !premium) {
        sendUpgradeRequired(response);
        return;
      }
      if (isStandardMode(mode) && !premium) {
        const standardWordsUsed = await getStandardWordsUsed(authHeader, user.id);
        if (standardWordsUsed >= STANDARD_FREE_WORD_LIMIT) {
          sendUpgradeRequired(response);
          return;
        }
      }

      const result = await startPracticeSessionInDB(
        authHeader,
        user.id,
        mode,
        level,
        forceCloseCurrent,
        {
          originLanguage,
          customListId,
        },
      );
      sendJson(response, 200, result);
      return;
    }

    if (request.method === "POST" && url.pathname === "/api/sessions/attempts") {
      const user = await authenticateRequest(request);
      const authHeader = request.headers.authorization || "";
      const rawBody = await collectBody(request);
      const {
        sessionId,
        targetWord,
        childAttempt,
        isCorrect,
        level,
        definitionViewed,
        exampleViewed,
        originViewed,
        partOfSpeechViewed,
        repeatWordCount,
        usedVoiceInput,
        mode,
        coachingResponse,
      } = JSON.parse(rawBody);

      if (!mode) {
        sendJson(response, 400, { error: "mode is required" });
        return;
      }

      const premium = await hasPremiumAccess(authHeader, user.id);
      if (!isStandardMode(mode) && !premium) {
        sendUpgradeRequired(response);
        return;
      }
      if (isStandardMode(mode) && !premium) {
        const standardWordsUsed = await getStandardWordsUsed(authHeader, user.id);
        if (standardWordsUsed >= STANDARD_FREE_WORD_LIMIT) {
          sendUpgradeRequired(response);
          return;
        }
      }

      const attemptId = await recordWordAttemptInDB(
        authHeader,
        user.id,
        sessionId,
        targetWord,
        childAttempt,
        isCorrect,
        mode,
        level,
        definitionViewed,
        exampleViewed,
        originViewed,
        partOfSpeechViewed,
        repeatWordCount,
        usedVoiceInput,
        coachingResponse,
      );

      sendJson(response, 200, { attemptId });
      return;
    }

    if (request.method === "POST" && url.pathname === "/api/sessions/end") {
      const user = await authenticateRequest(request);
      const authHeader = request.headers.authorization || "";
      const rawBody = await collectBody(request);
      const { sessionId, totalWordsAttempted, totalCorrect, durationSeconds } = JSON.parse(rawBody);

      await endPracticeSessionInDB(
        authHeader,
        user.id,
        sessionId,
        totalWordsAttempted || 0,
        totalCorrect || 0,
        durationSeconds || 0,
      );

      sendJson(response, 200, { success: true });
      return;
    }

    if (request.method === "GET" && url.pathname === "/api/foreign-origins") {
      sendJson(response, 200, {
        origins: listForeignOrigins(),
      });
      return;
    }

    if (
      request.method === "GET" &&
      url.pathname.startsWith("/api/foreign-origins/")
    ) {
      const parts = url.pathname.split("/");
      const origin = decodeURIComponent(parts[3] ?? "");

      if (!origin) {
        sendJson(response, 400, { error: "Foreign origin is required." });
        return;
      }

      const list = getForeignOriginWordListByOrigin(origin);
      if (!list) {
        sendJson(response, 404, {
          error: `Unknown foreign origin: ${origin}`,
        });
        return;
      }

      sendJson(response, 200, {
        origin: {
          origin: list.origin,
          wordCount: list.words.length,
          words: list.words.map((word) => buildWordResponse(word)),
        },
      });
      return;
    }

    if (
      request.method === "GET" &&
      url.pathname.startsWith("/api/custom-lists/")
    ) {
      const parts = url.pathname.split("/");
      const listId = decodeURIComponent(parts[3] ?? "");

      if (!listId) {
        sendJson(response, 400, { error: "Custom list id is required." });
        return;
      }

      const user = await authenticateRequest(request);
      const authHeader = request.headers.authorization || "";
      const list = await fetchCustomListByIdFromDB(authHeader, listId, user.id);
      if (!list) {
        sendJson(response, 404, {
          error: `Unknown custom list: ${listId}`,
        });
        return;
      }

      sendJson(response, 200, {
        list: {
          id: list.id,
          name: list.name,
          wordCount: list.words.length,
          words: list.words.map((word) => buildWordResponse(word)),
        },
      });
      return;
    }

    // Secure pronunciation route: resolves word via challengeId without exposing it in the URL.
    // This is the final route used after the frontend is fully migrated to challengeId.
    // The legacy /api/words/:word/pronunciation route below is a temporary backward-compatibility path.
    if (
      request.method === "GET" &&
      url.pathname === "/api/words/pronunciation"
    ) {
      const challengeId = url.searchParams.get("challengeId");
      const sessionId = url.searchParams.get("sessionId");

      if (!challengeId || !sessionId) {
        sendJson(response, 400, { error: "challengeId and sessionId are required." });
        return;
      }

      const user = await authenticateRequest(request);
      const authHeader = request.headers.authorization || "";
      const resolvedWord = await peekChallengeInSession(authHeader, user.id, sessionId, challengeId);

      if (!resolvedWord) {
        sendJson(response, 400, { error: "Invalid or expired challengeId." });
        return;
      }

      const audio = await generatePronunciationAudio(resolvedWord);
      sendAudio(response, 200, audio);
      return;
    }

    if (request.method === "GET" && url.pathname === "/api/voice/capabilities") {
      sendJson(response, 200, {
        intents: [
          "repeat_word",
          "example_sentence",
          "definition",
          "origin",
          "part_of_speech",
          "spelling_attempt",
        ],
        spellingBehavior: {
          shouldAutoSubmit: false,
          micDuringPlayback: "disable",
        },
      });
      return;
    }

    if (request.method === "POST" && url.pathname === "/api/audio/transcribe") {
      const contentType = request.headers["content-type"] ?? "audio/webm";
      const fileName = request.headers["x-audio-filename"];
      const audio = await collectBinaryBody(request);
      const text = await transcribeAudio(
        audio,
        Array.isArray(contentType) ? contentType[0] : contentType,
        Array.isArray(fileName) ? fileName[0] : fileName ?? "voice.webm",
      );
      sendJson(response, 200, { text });
      return;
    }

    if (request.method === "POST" && url.pathname === "/api/voice/interpret") {
      const rawBody = await collectBody(request);
      const requestBody = VoiceInterpretRequestSchema.parse(JSON.parse(rawBody));
      const result = interpretVoiceUtterance(
        requestBody.targetWord,
        requestBody.utterance,
      );
      sendJson(response, 200, result);
      return;
    }

    if (request.method === "POST" && url.pathname === "/api/voice/respond") {
      const rawBody = await collectBody(request);
      const parsed = JSON.parse(rawBody);

      let targetWord: string;
      if (parsed.challengeId && parsed.sessionId) {
        // Secure path: resolve the word from the active session.
        const user = await authenticateRequest(request);
        const authHeader = request.headers.authorization || "";
        const resolved = await peekChallengeInSession(authHeader, user.id, parsed.sessionId, parsed.challengeId);
        if (!resolved) {
          sendJson(response, 400, { error: "Invalid or expired challengeId." });
          return;
        }
        targetWord = resolved;
      } else {
        // Legacy path: targetWord sent directly. Validate via existing schema.
        const requestBody = VoiceRespondRequestSchema.parse(parsed);
        targetWord = requestBody.targetWord;
      }

      const includeAudio = typeof parsed.includeAudio === "boolean" ? parsed.includeAudio : true;
      const utterance = String(parsed.utterance ?? "");
      const result = await buildVoiceResponse(targetWord, utterance, includeAudio);
      sendJson(response, 200, result);
      return;
    }

    if (
      request.method === "POST" &&
      url.pathname === "/api/spelling-coach/preview-input"
    ) {
      const rawBody = await collectBody(request);
      const requestBody = CoachingRequestSchema.parse(JSON.parse(rawBody));
      sendJson(response, 200, buildSpellingCoachInput(requestBody));
      return;
    }

    if (request.method === "POST" && url.pathname === "/api/spelling-coach") {
      const startTime = Date.now();
      const rawBody = await collectBody(request);
      const requestBody = CoachingRequestSchema.parse(JSON.parse(rawBody));
      const coachInput = buildSpellingCoachInput(requestBody);
      const result = hasWordTeachingPrecompute(coachInput)
        ? await runSplitSpellingCoachAgent(coachInput)
        : await runSpellingCoachAgent(coachInput);

      try {
        await recordSpellingCoachTrace({
          input: coachInput,
          output: result,
          latencyMs: Date.now() - startTime,
        });
      } catch (err) {
        console.error("[LANGFUSE] Error in recordSpellingCoachTrace:", err);
      }
      sendJson(response, 200, result);
      return;
    }

    if (
      request.method === "POST" &&
      url.pathname === "/api/spelling-coach/stream"
    ) {
      let requestBody: unknown;
      try {
        requestBody = await readSpellingCoachStreamRequest(request);
      } catch (error) {
        if (error instanceof SpellingCoachStreamRequestError) {
          logInfo(
            `[spelling-coach stream] rejected invalid request: ${error.message}`,
          );
          if (!response.destroyed) {
            sendJson(response, error.statusCode, { error: error.message });
          }
          return;
        }
        throw error;
      }

      // If a challengeId was provided, resolve it to the targetWord via the session.
      const parsedBody = requestBody as { challengeId?: string; sessionId?: string; targetWord?: string };
      if (parsedBody.challengeId) {
        if (!parsedBody.sessionId) {
          sendJson(response, 400, { error: "sessionId is required when challengeId is provided." });
          return;
        }
        const authHeader = request.headers.authorization || "";
        const user = await authenticateRequest(request);
        const resolvedWord = await getChallengeFromSession(
          authHeader,
          user.id,
          parsedBody.sessionId,
          parsedBody.challengeId,
        );
        if (!resolvedWord) {
          sendJson(response, 400, { error: "Invalid or expired challengeId." });
          return;
        }
        // Inject the resolved targetWord and strip challengeId before passing downstream.
        requestBody = { ...parsedBody, targetWord: resolvedWord, challengeId: undefined };
      }

      try {
        await streamSpellingCoach(
          request,
          response,
          requestBody,
          undefined,
          requestStart,
        );
      } catch (error) {
        logError("Spelling coach streaming API error:", error);
        Sentry.captureException(error);
        if (!response.destroyed && !response.headersSent) {
          sendJson(
            response,
            500,
            {
              error: "Streaming request failed.",
            },
          );
        } else if (!response.destroyed && !response.writableEnded) {
          response.end();
        }
      }
      return;
    }

    if (request.method === "POST" && url.pathname === "/api/words/import-custom") {
      const startTime = Date.now();
      const rawBody = await collectBody(request);
      const requestBody = CustomWordImportRequestSchema.parse(JSON.parse(rawBody));
      const user = await authenticateRequest(request);
      const authHeader = request.headers.authorization || "";

      let existingList: any = undefined;
      if (requestBody.listId) {
        const dbList = await fetchCustomListByIdFromDB(authHeader, requestBody.listId, user.id);
        if (dbList) {
          existingList = dbList;
        }
      }

      const result = await importCustomWords(requestBody, {
        ownerUserId: user.id,
        existingList,
        skipFileSave: true,
      });

      const savedList = await saveCustomListToDB(
        authHeader,
        user.id,
        requestBody.listName,
        result.listWords,
        requestBody.listId || result.list.id,
      );

      try {
        await recordImportListTrace({
          user,
          listName: requestBody.listName,
          wordCount: requestBody.words.length,
          latencyMs: Date.now() - startTime,
          inputWords: Array.isArray(requestBody.words) ? requestBody.words : [requestBody.words],
          outputWords: savedList.words.map((word) => buildWordResponse(word)),
        });
      } catch (err) {
        console.error("[LANGFUSE] Error in recordImportListTrace:", err);
      }
      sendJson(response, 200, {
        list: {
          id: savedList.id,
          name: savedList.name,
          wordCount: savedList.words.length,
        },
        importedCount: result.importedCount,
        skippedExistingCount: result.skippedExistingCount,
      });
      return;
    }

    if (request.method === "POST" && url.pathname === "/api/words/import-file") {
      try {
        const user = await authenticateRequest(request);
        const rawBytes = await collectBinaryBody(request, MAX_IMPORT_REQUEST_BYTES);
        const result = await handleImportFileRequest(request, Buffer.from(rawBytes), user);

        sendJson(response, 202, result);
      } catch (error) {
        if (isAuthError(error)) throw error;
        const statusCode = error instanceof FileImportRequestError ? error.statusCode : 500;
        if (statusCode >= 500) {
          logError("File import request failed:", error);
          Sentry.captureException(error);
        }
        sendJson(response, statusCode, {
          error: error instanceof Error ? error.message : "File import failed.",
        });
      }
      return;
    }

    if (
      request.method === "GET" &&
      /^\/api\/words\/import-jobs\/[^/]+$/.test(url.pathname)
    ) {
      const parts = url.pathname.split("/").filter(Boolean);
      const jobId = decodeURIComponent(parts[3] ?? "");
      if (!jobId) {
        sendJson(response, 400, { error: "Job ID is required." });
        return;
      }
      const user = await authenticateRequest(request);
      const job = getImportJob(jobId, user.id);
      if (!job) {
        sendJson(response, 404, { error: "Import job not found or expired." });
        return;
      }
      sendJson(response, 200, job);
      return;
    }

    if (
      request.method === "POST" &&
      url.pathname === "/api/words/import-foreign-origins"
    ) {
      const rawBody = await collectBody(request);
      const requestBody = ForeignOriginImportRequestSchema.parse(
        JSON.parse(rawBody),
      );
      const result = await importForeignOriginWords(requestBody);
      sendJson(response, 200, {
        origins: result.origins,
        importedCount: result.importedCount,
        skippedExistingCount: result.skippedExistingCount,
        words: result.words.map((word) => buildWordResponse(word)),
      });
      return;
    }

    if (
      request.method === "POST" &&
      url.pathname === "/api/stripe/create-checkout-session"
    ) {
      const user = await authenticateRequest(request);
      if (!user.email) {
        sendJson(response, 400, { error: "User email is required for checkout." });
        return;
      }

      const cleanReferer = getFrontendReturnUrl(request);
      const successUrl = `${cleanReferer}?payment_success=true`;
      const cancelUrl = `${cleanReferer}?payment_cancelled=true`;

      if (!process.env.STRIPE_SECRET_KEY) {
        sendJson(response, 500, { error: "STRIPE_SECRET_KEY is not configured on the server." });
        return;
      }
      if (!process.env.STRIPE_PRICE_ID) {
        sendJson(response, 500, { error: "STRIPE_PRICE_ID is not configured on the server." });
        return;
      }

      const stripe = getStripe();
      const session = await stripe.checkout.sessions.create({
        payment_method_types: ["card"],
        customer_email: user.email,
        line_items: [
          {
            price: process.env.STRIPE_PRICE_ID,
            quantity: 1,
          },
        ],
        mode: "subscription",
        success_url: successUrl,
        cancel_url: cancelUrl,
        client_reference_id: user.id,
      });

      sendJson(response, 200, { url: session.url });
      return;
    }

    if (
      request.method === "GET" &&
      url.pathname === "/api/stripe/subscription-status"
    ) {
      const user = await authenticateRequest(request);
      const authHeader = request.headers.authorization || "";

      if (!user.email) {
        sendJson(response, 200, { subscribed: false });
        return;
      }

      // 1. Try to read from the database first
      try {
        const cachedSub = await getUserSubscriptionFromDB(authHeader, user.id);
        if (cachedSub) {
          const status = cachedSub.status;
          const currentPeriodEnd = cachedSub.current_period_end;

          if ((status === "active" || status === "trialing") && currentPeriodEnd) {
            const expiryTime = new Date(currentPeriodEnd).getTime();
            // If the subscription is active and has more than 10 minutes left
            if (expiryTime > Date.now() + 10 * 60 * 1000) {
              let stripePriceId: string | null = cachedSub.stripe_price_id ?? null;
              let priceAmount: number | null = cachedSub.price_unit_amount ?? null;
              let priceCurrency: string | null = cachedSub.price_currency ?? null;
              let billingInterval: string | null = cachedSub.billing_interval ?? null;
              if (process.env.STRIPE_SECRET_KEY && cachedSub.stripe_subscription_id) {
                try {
                  const stripeSubscription = await getStripe().subscriptions.retrieve(
                    cachedSub.stripe_subscription_id,
                  );
                  const price = stripeSubscription.items.data[0]?.price;
                  stripePriceId = price?.id ?? null;
                  priceAmount = price?.unit_amount ?? null;
                  priceCurrency = price?.currency ?? null;
                  billingInterval = price?.recurring?.interval ?? null;
                  await updateUserSubscriptionInDB(authHeader, user.id, {
                    stripeCustomerId: cachedSub.stripe_customer_id,
                    stripeSubscriptionId: cachedSub.stripe_subscription_id,
                    status,
                    currentPeriodEnd,
                    cancelAtPeriodEnd: cachedSub.cancel_at_period_end || false,
                    stripePriceId,
                    priceUnitAmount: priceAmount,
                    priceCurrency,
                    billingInterval,
                  });
                } catch (err) {
                  logError("Failed to fetch cached subscription price from Stripe:", err);
                }
              }
              sendJson(response, 200, {
                subscribed: true,
                currentPeriodEnd: Math.floor(expiryTime / 1000),
                cancelAtPeriodEnd: cachedSub.cancel_at_period_end || false,
                priceAmount,
                priceCurrency,
                billingInterval,
              });
              return;
            }
          }
        }
      } catch (err) {
        logError("Failed to fetch subscription from DB:", err);
      }

      // 2. Fall back to live Stripe query if not in DB or expired/inactive
      if (!process.env.STRIPE_SECRET_KEY) {
        sendJson(response, 500, { error: "STRIPE_SECRET_KEY is not configured on the server." });
        return;
      }

      const stripe = getStripe();
      const customers = await stripe.customers.list({
        email: user.email,
        limit: 1,
      });

      if (customers.data.length === 0) {
        try {
          await updateUserSubscriptionInDB(authHeader, user.id, {
            stripeCustomerId: null,
            stripeSubscriptionId: null,
            status: "inactive",
            currentPeriodEnd: null,
            cancelAtPeriodEnd: false,
            stripePriceId: null,
            priceUnitAmount: null,
            priceCurrency: null,
            billingInterval: null,
          });
        } catch (err) {
          logError("Failed to update subscription in DB:", err);
        }
        sendJson(response, 200, { subscribed: false });
        return;
      }
      const stripeCustomerId = customers.data[0].id;
      const subscriptions = await stripe.subscriptions.list({
        customer: stripeCustomerId,
        status: "all",
        limit: 10,
      });

      const sub = subscriptions.data.find((item) =>
        isPremiumSubscriptionStatus(item.status),
      ) as any;

      if (sub) {
        const periodEnd = sub.current_period_end || sub.items?.data?.[0]?.current_period_end || sub.billing_cycle_anchor;
        const cancelAtPeriodEnd = sub.cancel_at_period_end || false;
        const stripeSubscriptionId = sub.id;
        const status = sub.status || "active";
        const price = sub.items?.data?.[0]?.price;

        try {
          await updateUserSubscriptionInDB(authHeader, user.id, {
            stripeCustomerId,
            stripeSubscriptionId,
            status,
            currentPeriodEnd: new Date(periodEnd * 1000).toISOString(),
            cancelAtPeriodEnd,
            stripePriceId: price?.id ?? null,
            priceUnitAmount: price?.unit_amount ?? null,
            priceCurrency: price?.currency ?? null,
            billingInterval: price?.recurring?.interval ?? null,
          });
        } catch (err) {
          logError("Failed to update subscription in DB:", err);
        }

        sendJson(response, 200, {
          subscribed: true,
          currentPeriodEnd: periodEnd,
          cancelAtPeriodEnd,
          priceAmount: price?.unit_amount ?? null,
          priceCurrency: price?.currency ?? null,
          billingInterval: price?.recurring?.interval ?? null,
        });
      } else {
        try {
          await updateUserSubscriptionInDB(authHeader, user.id, {
            stripeCustomerId,
            stripeSubscriptionId: null,
            status: "inactive",
            currentPeriodEnd: null,
            cancelAtPeriodEnd: false,
            stripePriceId: null,
            priceUnitAmount: null,
            priceCurrency: null,
            billingInterval: null,
          });
        } catch (err) {
          logError("Failed to update subscription in DB:", err);
        }
        sendJson(response, 200, { subscribed: false });
      }
      return;
    }

    if (
      request.method === "POST" &&
      url.pathname === "/api/stripe/create-portal-session"
    ) {
      const user = await authenticateRequest(request);
      if (!user.email) {
        sendJson(response, 400, { error: "User email is required." });
        return;
      }

      if (!process.env.STRIPE_SECRET_KEY) {
        sendJson(response, 500, { error: "STRIPE_SECRET_KEY is not configured on the server." });
        return;
      }

      const stripe = getStripe();
      const customers = await stripe.customers.list({
        email: user.email,
        limit: 1,
      });

      if (customers.data.length === 0) {
        sendJson(response, 400, { error: "No active Stripe customer found." });
        return;
      }

      const cleanReferer = getFrontendReturnUrl(request);

      const portalSession = await stripe.billingPortal.sessions.create({
        customer: customers.data[0].id,
        return_url: cleanReferer,
      });

      sendJson(response, 200, { url: portalSession.url });
      return;
    }
    sendJson(response, 404, { error: "Not found." });
  } catch (error) {
    logError("Spelling coach API error:", error);
    Sentry.captureException(error);
    if (response.destroyed) {
      return;
    }
    if (response.headersSent) {
      if (!response.writableEnded) {
        response.end();
      }
      return;
    }
    if (
      error instanceof Error &&
      error.message.startsWith("Unknown mock bee session:")
    ) {
      sendJson(response, 404, { error: error.message });
      return;
    }
    if (
      error instanceof Error &&
      error.message.includes("is already completed.")
    ) {
      sendJson(response, 409, { error: error.message });
      return;
    }
    if (isAuthError(error)) {
      const statusCode =
        error instanceof Error &&
          error.message.startsWith("Supabase auth is not configured.")
          ? 500
          : 401;
      sendJson(response, statusCode, {
        error: error instanceof Error ? error.message : "Unauthorized.",
      });
      return;
    }

    sendJson(response, 400, {
      error: error instanceof Error ? error.message : String(error),
    });
  }
}

const server = createServer(handler);

if (!process.env.VERCEL) {
  server.listen(PORT, () => {
    logInfo(`Spelling coach API listening on http://localhost:${PORT}`);
  });
}
