import "dotenv/config";
import "./instrument.js";
import * as Sentry from "@sentry/node";

import { createServer } from "node:http";
import { URL } from "node:url";
import { authenticateRequest } from "./auth.js";
import {
  buildSpellingCoachInput,
  buildWordPrecomputeInput,
  buildWordResponse,
  CoachingRequestSchema,
  LevelQuerySchema,
} from "./inputBuilder.js";
import { CustomWordImportRequestSchema, importCustomWords } from "./customWordImport.js";
import {
  ForeignOriginImportRequestSchema,
  importForeignOriginWords,
} from "./foreignOriginImport.js";
import { getConfiguredModelName } from "./modelConfig.js";
import { hasWordTeachingPrecompute, runSplitSpellingCoachAgent, warmWordTeachingPrecompute } from "./optimizedCoach.js";
import { generatePronunciationAudio } from "./pronunciation.js";
import {
  isSpellingRulePromptHintsEnabled,
  isSpellingRuleShortlistEnabled,
} from "./referenceData.js";
import { runSpellingCoachAgent } from "./runAgent.js";
import { recordSpellingCoachTrace, recordImportListTrace } from "./langfuse.js";
import {
  getCustomWordListById,
  getForeignOriginWordListByOrigin,
  getWordByText,
  listCustomWordListsForUser,
  listForeignOrigins,
  pickNextWord,
} from "./wordCatalog.js";
import { logError, logInfo } from "./logging.js";
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

const PORT = Number(process.env.PORT ?? 3000);

function sendJson(response: import("node:http").ServerResponse, statusCode: number, body: unknown) {
  response.writeHead(statusCode, {
    "Content-Type": "application/json; charset=utf-8",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization",
  });
  response.end(`${JSON.stringify(body, null, 2)}\n`);
}

function sendAudio(
  response: import("node:http").ServerResponse,
  statusCode: number,
  audio: Uint8Array,
): void {
  response.writeHead(statusCode, {
    "Content-Type": "audio/mpeg",
    "Content-Length": audio.byteLength,
    "Cache-Control": "public, max-age=3600",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization",
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

      sendJson(response, 200, {
        ok: true,
        runtime,
        model: getConfiguredModelName(),
        featureFlags: {
          audioCaching,
          ttsInstructions,
          spellingRuleShortlist,
          spellingRulePromptHints,
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
      const user = query.customListId
        ? await authenticateRequest(request)
        : undefined;
      const word = pickNextWord(
        query.level,
        query.exclude,
        query.customListId,
        query.foreignOrigin,
        user?.id,
      );
      if (word.level !== "1") {
        const precomputeInput = buildWordPrecomputeInput(word.word);
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
      }
      sendJson(response, 200, buildWordResponse(word));
      return;
    }

    if (request.method === "GET" && url.pathname === "/api/custom-lists") {
      const user = await authenticateRequest(request);
      sendJson(response, 200, {
        lists: listCustomWordListsForUser(user.id),
      });
      return;
    }

    if (request.method === "GET" && url.pathname === "/api/auth/me") {
      const user = await authenticateRequest(request);
      sendJson(response, 200, { user });
      return;
    }

    if (request.method === "GET" && url.pathname === "/api/subscription/status") {
      const user = await authenticateRequest(request);
      sendJson(response, 200, {
        subscribed: true,
        currentPeriodEnd: Math.floor(Date.now() / 1000) + 30 * 24 * 60 * 60, // 30 days from now
        cancelAtPeriodEnd: false,
      });
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
      const list = getCustomWordListById(listId, user.id);
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

    if (
      request.method === "GET" &&
      url.pathname.startsWith("/api/words/") &&
      url.pathname.endsWith("/pronunciation")
    ) {
      const parts = url.pathname.split("/");
      const encodedWord = parts[3];
      const word = decodeURIComponent(encodedWord ?? "");

      if (!word) {
        sendJson(response, 400, { error: "Word is required." });
        return;
      }

      const wordEntry = getWordByText(word);
      if (!wordEntry) {
        sendJson(response, 404, { error: `Unknown word: ${word}` });
        return;
      }

      const audio = await generatePronunciationAudio(wordEntry.word);
      sendAudio(response, 200, audio);
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

    if (request.method === "POST" && url.pathname === "/api/words/import-custom") {
      const startTime = Date.now();
      const rawBody = await collectBody(request);
      const requestBody = CustomWordImportRequestSchema.parse(JSON.parse(rawBody));
      const user = await authenticateRequest(request);
      const result = await importCustomWords(requestBody, {
        ownerUserId: user.id,
      });
      try {
        await recordImportListTrace({
          user,
          listName: requestBody.listName,
          wordCount: requestBody.words.length,
          latencyMs: Date.now() - startTime,
        });
      } catch (err) {
        console.error("[LANGFUSE] Error in recordImportListTrace:", err);
      }
      sendJson(response, 200, {
        list: result.list,
        importedCount: result.importedCount,
        skippedExistingCount: result.skippedExistingCount,
        words: result.words.map((word) => buildWordResponse(word)),
      });
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

      const referer = request.headers.referer || request.headers.origin || "http://localhost:5173";
      const cleanReferer = referer.split("?")[0].split("#")[0];
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
      if (!user.email) {
        sendJson(response, 200, { subscribed: false });
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
        sendJson(response, 200, { subscribed: false });
        return;
      }

      const subscriptions = await stripe.subscriptions.list({
        customer: customers.data[0].id,
        status: "active",
        limit: 1,
      });

      if (subscriptions.data.length > 0) {
        const sub = subscriptions.data[0] as any;
        const periodEnd = sub.current_period_end || sub.items?.data?.[0]?.current_period_end || sub.billing_cycle_anchor;
        sendJson(response, 200, {
          subscribed: true,
          currentPeriodEnd: periodEnd,
          cancelAtPeriodEnd: sub.cancel_at_period_end,
        });
      } else {
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

      const referer = request.headers.referer || request.headers.origin || "http://localhost:5173";
      const cleanReferer = referer.split("?")[0].split("#")[0];

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
