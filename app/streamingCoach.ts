import type { IncomingMessage, ServerResponse } from "node:http";
import { randomUUID } from "node:crypto";
import { z } from "zod";
import {
  buildSpellingCoachInput,
  type CoachingRequest,
} from "./inputBuilder.js";
import {
  warmWordTeachingPrecompute,
} from "./optimizedCoach.js";
import {
  createDirectSpellingCoachModel,
} from "./directModel.js";
import {
  buildStreamingRuntimePrompt,
  SPELLING_COACH_STREAMING_RUNTIME_SYSTEM_PROMPT,
} from "./prompt.js";
import type {
  SpellingCoachInput,
  SpellingCoachOutput,
  WordTeachingPrecompute,
} from "./schemas.js";
import { logError, logInfo } from "./logging.js";
import { getFriendlyPronunciationCue } from "./friendlyPronunciation.js";
export const SpellingCoachStreamRequestSchema = z
  .object({
    // Either targetWord (legacy) or challengeId+sessionId must be provided.
    // server.ts resolves challengeId to targetWord before this schema is used
    // for the actual coaching request.
    targetWord: z.string().min(1).optional(),
    challengeId: z.string().min(1).optional(),
    childAttempt: z.string(),
    level: z.number().finite().optional(),
    mode: z.string().min(1),
    definitionViewed: z.boolean(),
    exampleViewed: z.boolean(),
    originViewed: z.boolean(),
    partOfSpeechViewed: z.boolean(),
    repeatWordCount: z.number().int().nonnegative(),
    usedVoiceInput: z.boolean(),
    sessionId: z.string().min(1).optional(),
  })
  .strict();

export type SpellingCoachStreamRequest = z.infer<
  typeof SpellingCoachStreamRequestSchema
>;

export const SPELLING_COACH_STREAM_MAX_BODY_BYTES = 64 * 1024;

export class SpellingCoachStreamRequestError extends Error {
  constructor(
    message: string,
    readonly statusCode: 400 | 408 | 413,
    options?: ErrorOptions,
  ) {
    super(message, options);
    this.name = "SpellingCoachStreamRequestError";
  }
}

function collectStreamRequestBody(
  request: IncomingMessage,
  maxBytes: number,
  timeoutMs: number,
): Promise<string> {
  return new Promise((resolve, reject) => {
    let body = "";
    let bodyBytes = 0;
    let settled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const cleanup = () => {
      if (timer) clearTimeout(timer);
      request.removeListener("data", onData);
      request.removeListener("end", onEnd);
      request.removeListener("error", onError);
      request.removeListener("aborted", onAborted);
      request.removeListener("close", onClose);
    };
    const drainAfterReject = () => {
      const finishDraining = () => {
        request.removeListener("error", finishDraining);
        request.removeListener("end", finishDraining);
        request.removeListener("close", finishDraining);
      };
      request.once("error", finishDraining);
      request.once("end", finishDraining);
      request.once("close", finishDraining);
      request.resume();
    };
    const onData = (chunk: Buffer | string) => {
      if (settled) return;
      const chunkBytes = Buffer.isBuffer(chunk)
        ? chunk.byteLength
        : Buffer.byteLength(chunk);
      bodyBytes += chunkBytes;
      if (bodyBytes > maxBytes) {
        settled = true;
        body = "";
        cleanup();
        // Keep draining the request so the server can return a clean 413.
        drainAfterReject();
        reject(
          new SpellingCoachStreamRequestError(
            `Request body exceeds the ${maxBytes}-byte limit.`,
            413,
          ),
        );
        return;
      }
      body += chunk.toString();
    };
    const onEnd = () => {
      if (settled) return;
      settled = true;
      cleanup();
      resolve(body);
    };
    const onError = (error: Error) => {
      if (settled) return;
      settled = true;
      cleanup();
      reject(error);
    };
    const onAborted = () => {
      if (settled) return;
      settled = true;
      cleanup();
      reject(new Error("Request was aborted before its body was received."));
    };
    const onClose = () => {
      if (settled) return;
      settled = true;
      cleanup();
      reject(new Error("Request closed before its body was received."));
    };
    const onTimeout = () => {
      if (settled) return;
      settled = true;
      cleanup();
      drainAfterReject();
      reject(
        new SpellingCoachStreamRequestError(
          `Request body was not received within ${timeoutMs}ms.`,
          408,
        ),
      );
    };

    timer = setTimeout(onTimeout, timeoutMs);
    request.on("data", onData);
    request.once("end", onEnd);
    request.once("error", onError);
    request.once("aborted", onAborted);
    request.once("close", onClose);
  });
}

function defaultRequestBodyTimeoutMs(): number {
  const configured = Number(process.env.SPELLING_COACH_REQUEST_BODY_TIMEOUT_MS);
  return Number.isFinite(configured) && configured > 0 ? configured : 10_000;
}

export async function readSpellingCoachStreamRequest(
  request: IncomingMessage,
  maxBytes = SPELLING_COACH_STREAM_MAX_BODY_BYTES,
  timeoutMs = defaultRequestBodyTimeoutMs(),
): Promise<SpellingCoachStreamRequest> {
  const rawBody = await collectStreamRequestBody(request, maxBytes, timeoutMs);
  try {
    return SpellingCoachStreamRequestSchema.parse(JSON.parse(rawBody));
  } catch (error) {
    throw new SpellingCoachStreamRequestError(
      error instanceof Error ? error.message : "Invalid request body.",
      400,
      { cause: error },
    );
  }
}

export type SpellingCoachStreamSection =
  | "short_feedback"
  | "miss_analysis"
  | "explanation"
  | "memory_tip";

export type SpellingCoachStreamDependencies = {
  precompute?: (
    input: SpellingCoachInput,
    signal: AbortSignal,
  ) => Promise<WordTeachingPrecompute>;
  runRuntimeStream?: (
    input: SpellingCoachInput,
    precomputed: WordTeachingPrecompute,
    signal: AbortSignal,
  ) => AsyncIterable<string> | Promise<AsyncIterable<string>>;
  runRuntime?: (
    input: SpellingCoachInput,
    signal: AbortSignal,
  ) => Promise<SpellingCoachOutput>;
  requestId?: () => string;
  firstMarkerTimeoutMs?: number;
  sectionTimeoutMs?: number;
  runtimeTimeoutMs?: number;
  precomputeTimeoutMs?: number;
  now?: () => number;
};

type FlushableResponse = ServerResponse & { flush?: () => void };

type SectionError = Error & { code: string };

const SECTIONS: readonly SpellingCoachStreamSection[] = [
  "short_feedback",
  "miss_analysis",
  "explanation",
  "memory_tip",
];

function defaultSectionTimeoutMs(): number {
  const configured = Number(process.env.SPELLING_COACH_SECTION_TIMEOUT_MS);
  return Number.isFinite(configured) && configured > 0 ? configured : 30_000;
}

function defaultPrecomputeTimeoutMs(): number {
  const configured = Number(process.env.SPELLING_COACH_PRECOMPUTE_TIMEOUT_MS);
  return Number.isFinite(configured) && configured > 0 ? configured : 10_000;
}

function withPrecomputeTimeout<T>(
  promise: Promise<T>,
  timeoutMs: number,
  onTimeout: () => void,
): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    let settled = false;
    const finish = (callback: (value: T) => void, value: T) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      callback(value);
    };
    const fail = (error: unknown) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      reject(error);
    };
    const timer = setTimeout(() => {
      onTimeout();
      fail(new Error(`Precompute timed out after ${timeoutMs}ms.`));
    }, timeoutMs);
    promise.then((value) => finish(resolve, value), fail);
  });
}

function buildLegacyCoachingRequest(
  request: SpellingCoachStreamRequest,
): CoachingRequest {
  if (!request.targetWord) {
    throw new Error("targetWord must be resolved before building coaching request.");
  }
  return {
    targetWord: request.targetWord,
    childAttempt: request.childAttempt,
    childProfile: {
      childId: request.sessionId ?? "streaming-client",
      age: 0,
      grade: String(request.level),
      spellingLevel: String(request.level),
    },
    supportsUsed: {
      definitionViewed: request.definitionViewed,
      exampleViewed: request.exampleViewed,
      originViewed: request.originViewed,
    },
    sessionContext: {
      mode: request.mode,
      previousAttemptsOnThisWord: request.repeatWordCount,
      previousMissPatterns: [],
      recentlyPracticedWords: [],
    },
    level: request.level,
  };
}

export function buildStreamingSpellingCoachInput(
  request: SpellingCoachStreamRequest,
): SpellingCoachInput {
  return buildSpellingCoachInput(buildLegacyCoachingRequest(request));
}

function timeoutError(
  section: SpellingCoachStreamSection,
  code = "SECTION_TIMEOUT",
): SectionError {
  const error = new Error(
    `Runtime coaching section \"${section}\" timed out.`,
  ) as SectionError;
  error.code = code;
  return error;
}

function sectionFailure(error: unknown): { code: string; message: string } {
  if (error && typeof error === "object" && "code" in error) {
    const candidate = error as { code?: unknown; message?: unknown };
    const code =
      typeof candidate.code === "string"
        ? candidate.code
        : "SECTION_GENERATION_FAILED";
    return {
      code,
      message:
        code === "SECTION_TIMEOUT" && typeof candidate.message === "string"
          ? candidate.message
          : "Runtime coaching section failed.",
    };
  }

  return {
    code: "SECTION_GENERATION_FAILED",
    message: "Runtime coaching section failed.",
  };
}

function writeSseEvent(
  response: FlushableResponse,
  event: string,
  data: unknown,
): void {
  if (response.destroyed || response.writableEnded) {
    return;
  }

  response.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
  response.flush?.();
}

function defaultFirstMarkerTimeoutMs(): number {
  const configured = Number(process.env.SPELLING_COACH_FIRST_MARKER_TIMEOUT_MS);
  return Number.isFinite(configured) && configured > 0 ? configured : 10_000;
}

function defaultRuntimeTimeoutMs(): number {
  const configured = Number(process.env.SPELLING_COACH_RUNTIME_TIMEOUT_MS);
  return Number.isFinite(configured) && configured > 0 ? configured : 45_000;
}

function extractTextContent(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (typeof part === "string") return part;
        if (
          part &&
          typeof part === "object" &&
          "type" in part &&
          (part as { type?: unknown }).type === "text" &&
          "text" in part
        ) {
          return String((part as { text: unknown }).text);
        }
        return "";
      })
      .join("");
  }
  return "";
}

function extractRuntimeToken(chunk: unknown): string {
  if (typeof chunk === "string") return chunk;
  if (chunk && typeof chunk === "object") {
    const content = (chunk as { content?: unknown }).content;
    if (content !== undefined) return extractTextContent(content);
    const delta = (chunk as { delta?: unknown }).delta;
    if (delta !== undefined) return extractTextContent(delta);
  }
  return "";
}

function deterministicMissAnalysis(
  input: SpellingCoachInput,
): SpellingCoachOutput["missAnalysis"] {
  const signals = input.missSignals;
  const types: NonNullable<SpellingCoachOutput["missAnalysis"]["primaryErrorType"]>[] = [];
  if (signals.missingLetters.length > 0) {
    types.push("missing_letter");
  }
  if (signals.extraLetters.length > 0) {
    types.push("extra_letter");
  }
  if (signals.substitutedLetters.length > 0) {
    types.push("letter_substitution");
  }
  if (signals.transposedLetters.length > 0) {
    types.push("letter_transposition");
  }
  if (signals.repeatedLetterIssue) {
    types.push("double_letter_error");
  }
  if (signals.likelyRushed) {
    types.push("likely_rushed");
  }

  const primaryErrorFocus = [
    signals.missingLetters.length > 0
      ? `Missing: ${signals.missingLetters.join(", ")}`
      : "",
    signals.extraLetters.length > 0
      ? `Extra: ${signals.extraLetters.join(", ")}`
      : "",
    signals.substitutedLetters.length > 0
      ? `Substitution: ${signals.substitutedLetters.join(", ")}`
      : "",
    signals.transposedLetters.length > 0
      ? `Transposed: ${signals.transposedLetters.join(", ")}`
      : "",
  ].find(Boolean) ?? "";

  return {
    summary: "",
    primaryErrorType: types.length > 0 ? types[0] : null,
    secondaryErrorTypes: types.length > 1 ? types.slice(1) : [],
    errorTypeEvidence: {},
    primaryErrorFocus,
    likelyWrongWordInterpretation: false,
    usedMeaningDisambiguationWell: false,
  };
}

async function* singleChunkStream(value: string): AsyncIterable<string> {
  yield value;
}

async function defaultRuntimeStream(
  input: SpellingCoachInput,
  precomputed: WordTeachingPrecompute,
  signal: AbortSignal,
): Promise<AsyncIterable<string>> {
  const model = await createDirectSpellingCoachModel();
  const messages = [
    {
      role: "system" as const,
      content: SPELLING_COACH_STREAMING_RUNTIME_SYSTEM_PROMPT,
    },
    {
      role: "user" as const,
      content: buildStreamingRuntimePrompt(
        input,
        JSON.stringify(precomputed, null, 2),
      ),
    },
  ];

  if (model.stream) {
    const stream = await model.stream(messages, { signal });
    return (async function* () {
      for await (const chunk of stream) {
        const text = extractRuntimeToken(chunk);
        if (text) yield text;
      }
    })();
  }

  const response = await model.invoke(messages, { signal });
  return singleChunkStream(extractRuntimeToken(response));
}

type ParserEvent =
  | { type: "section-start"; section: SpellingCoachStreamSection }
  | { type: "section-chunk"; section: SpellingCoachStreamSection; text: string }
  | { type: "section-complete"; section: SpellingCoachStreamSection }
  | {
      type: "section-error";
      section: SpellingCoachStreamSection;
      code: string;
      message: string;
    };

const MARKERS: Record<string, SpellingCoachStreamSection | "end"> = {
  "[[SHORT_FEEDBACK]]": "short_feedback",
  "[[MISS_ANALYSIS]]": "miss_analysis",
  "[[EXPLANATION]]": "explanation",
  "[[MEMORY_TIP]]": "memory_tip",
  "[[END_SECTION]]": "end",
};

const SECTION_MARKER_BY_SECTION: Record<SpellingCoachStreamSection, string> = {
  short_feedback: "[[SHORT_FEEDBACK]]",
  miss_analysis: "[[MISS_ANALYSIS]]",
  explanation: "[[EXPLANATION]]",
  memory_tip: "[[MEMORY_TIP]]",
};

const MALFORMED_MARKER_PATTERN = /\[\[[A-Z_]+\]\]/;

function markerPrefixSuffixLength(value: string): number {
  const markerValues = Object.keys(MARKERS);
  const max = Math.min(
    value.length,
    Math.max(...markerValues.map((marker) => marker.length - 1)),
  );
  for (let length = max; length > 0; length -= 1) {
    const suffix = value.slice(-length);
    if (markerValues.some((marker) => marker.startsWith(suffix))) {
      return length;
    }
  }
  return 0;
}

class RuntimeSectionParser {
  private buffer = "";
  private activeSection: SpellingCoachStreamSection | null = null;
  private nextSectionIndex = 0;
  private sawValidMarker = false;
  private completed = new Set<SpellingCoachStreamSection>();

  feed(token: string): ParserEvent[] {
    this.buffer += token;
    return this.drain(false);
  }

  finish(): ParserEvent[] {
    return this.drain(true).concat(this.finishOpenAndMissing());
  }

  currentSection(): SpellingCoachStreamSection | null {
    return this.activeSection;
  }

  expectedSection(): SpellingCoachStreamSection | null {
    return SECTIONS[this.nextSectionIndex] ?? null;
  }

  private drain(final: boolean): ParserEvent[] {
    const events: ParserEvent[] = [];

    while (this.buffer.length > 0) {
      const markerIndex = this.findNextMarkerIndex();
      if (markerIndex === -1) {
        const keep = final ? 0 : markerPrefixSuffixLength(this.buffer);
        const text = this.buffer.slice(0, this.buffer.length - keep);
        this.buffer = this.buffer.slice(this.buffer.length - keep);
        if (!text) break;
        this.handleText(text, events);
        if (!final) break;
        continue;
      }

      if (markerIndex > 0) {
        const text = this.buffer.slice(0, markerIndex);
        this.buffer = this.buffer.slice(markerIndex);
        this.handleText(text, events);
      }

      const validMarker = Object.keys(MARKERS).find((marker) =>
        this.buffer.startsWith(marker),
      );
      if (validMarker) {
        this.buffer = this.buffer.slice(validMarker.length);
        this.handleMarker(validMarker, events);
        continue;
      }

      const malformed = this.buffer.match(MALFORMED_MARKER_PATTERN)?.[0];
      if (malformed) {
        this.buffer = this.buffer.slice(malformed.length);
        this.handleMalformedMarker(malformed, events);
        continue;
      }

      if (!final && this.buffer.startsWith("[[")) break;
      this.handleText(this.buffer[0], events);
      this.buffer = this.buffer.slice(1);
    }

    return events;
  }

  private findNextMarkerIndex(): number {
    const indexes = [
      ...Object.keys(MARKERS).map((marker) => this.buffer.indexOf(marker)),
      this.buffer.search(MALFORMED_MARKER_PATTERN),
    ].filter((index) => index >= 0);
    return indexes.length > 0 ? Math.min(...indexes) : -1;
  }

  private handleText(text: string, events: ParserEvent[]): void {
    if (!text) return;
    if (!this.sawValidMarker) {
      if (text.trim()) {
        events.push({
          type: "section-error",
          section: this.expectedSection() ?? "miss_analysis",
          code: "UNMARKED_TEXT_BEFORE_FIRST_MARKER",
          message: "Runtime output included text before the first section marker.",
        });
      }
      return;
    }
    if (!this.activeSection) {
      if (text.trim()) {
        events.push({
          type: "section-error",
          section: this.expectedSection() ?? "memory_tip",
          code: "UNMARKED_TEXT_OUTSIDE_SECTION",
          message: "Runtime output included text outside a marked section.",
        });
      }
      return;
    }
    events.push({
      type: "section-chunk",
      section: this.activeSection,
      text,
    });
  }

  private handleMarker(marker: string, events: ParserEvent[]): void {
    const markerTarget = MARKERS[marker];
    if (markerTarget === "end") {
      if (!this.activeSection) {
        events.push({
          type: "section-error",
          section: this.expectedSection() ?? "memory_tip",
          code: "END_SECTION_WITHOUT_SECTION",
          message: "Runtime output closed a section before opening one.",
        });
        return;
      }
      this.completeActive(events);
      return;
    }

    const markerSection = markerTarget;
    const markerIndex = SECTIONS.indexOf(markerSection);
    const expected = this.expectedSection();

    if (this.activeSection) {
      this.completeActive(events);
    }

    if (expected && markerSection !== expected) {
      if (markerIndex > this.nextSectionIndex) {
        for (
          let missingIndex = this.nextSectionIndex;
          missingIndex < markerIndex;
          missingIndex += 1
        ) {
          const missing = SECTIONS[missingIndex];
          events.push({
            type: "section-error",
            section: missing,
            code: "MISSING_SECTION_MARKER",
            message: `Runtime output skipped ${SECTION_MARKER_BY_SECTION[missing]}.`,
          });
        }
        this.nextSectionIndex = markerIndex;
      } else {
        events.push({
          type: "section-error",
          section: markerSection,
          code: "OUT_OF_ORDER_MARKER",
          message: `Runtime output repeated or reordered ${marker}.`,
        });
        return;
      }
    }

    this.sawValidMarker = true;
    this.activeSection = markerSection;
    this.nextSectionIndex = markerIndex + 1;
    events.push({ type: "section-start", section: markerSection });
  }

  private handleMalformedMarker(
    marker: string,
    events: ParserEvent[],
  ): void {
    events.push({
      type: "section-error",
      section: this.activeSection ?? this.expectedSection() ?? "miss_analysis",
      code: "MALFORMED_MARKER",
      message: `Runtime output included malformed section marker ${marker}.`,
    });
  }

  private completeActive(events: ParserEvent[]): void {
    if (!this.activeSection) return;
    const section = this.activeSection;
    this.completed.add(section);
    this.activeSection = null;
    events.push({ type: "section-complete", section });
  }

  private finishOpenAndMissing(): ParserEvent[] {
    const events: ParserEvent[] = [];
    if (this.activeSection) {
      this.completeActive(events);
    }
    for (const section of SECTIONS) {
      if (!this.completed.has(section)) {
        events.push({
          type: "section-error",
          section,
          code: "MISSING_SECTION",
          message: `Runtime output did not complete ${SECTION_MARKER_BY_SECTION[section]}.`,
        });
      }
    }
    return events;
  }
}

async function* runtimeOutputToMarkedText(
  promise: Promise<SpellingCoachOutput>,
  signal: AbortSignal,
): AsyncIterable<string> {
  const output = await Promise.race([
    promise,
    new Promise<never>((_resolve, reject) => {
      if (signal.aborted) {
        reject(signal.reason ?? new Error("Runtime aborted."));
        return;
      }
      signal.addEventListener(
        "abort",
        () => reject(signal.reason ?? new Error("Runtime aborted.")),
        { once: true },
      );
    }),
  ]);
  yield `[[SHORT_FEEDBACK]]${output.coachingText.shortFeedback}[[END_SECTION]]`;
  yield `[[MISS_ANALYSIS]]${output.missAnalysis.summary}[[END_SECTION]]`;
  yield `[[EXPLANATION]]${output.coachingText.fullExplanation}[[END_SECTION]]`;
  yield `[[MEMORY_TIP]]${output.coachingText.memoryTip}[[END_SECTION]]`;
}

async function* abortableRuntimeStream(
  stream: AsyncIterable<string>,
  signal: AbortSignal,
): AsyncIterable<string> {
  const iterator = stream[Symbol.asyncIterator]();
  const abortPromise = new Promise<never>((_resolve, reject) => {
    if (signal.aborted) {
      reject(signal.reason ?? new Error("Runtime aborted."));
      return;
    }
    signal.addEventListener(
      "abort",
      () => reject(signal.reason ?? new Error("Runtime aborted.")),
      { once: true },
    );
  });

  try {
    while (true) {
      const result = await Promise.race([iterator.next(), abortPromise]);
      if (result.done) break;
      yield result.value;
    }
  } finally {
    void iterator.return?.();
  }
}

/**
 * Streams one v1 spelling-coach response. Validation and precompute happen
 * before SSE headers are committed so request/fatal setup errors can use the
 * API's normal JSON error response.
 */
export async function streamSpellingCoach(
  request: IncomingMessage,
  response: ServerResponse,
  body: unknown,
  dependencies: SpellingCoachStreamDependencies = {},
  requestStart = performance.now(),
): Promise<void> {
  const now = dependencies.now ?? (() => performance.now());
  const parsedRequest = SpellingCoachStreamRequestSchema.parse(body);
  const coachInput = buildStreamingSpellingCoachInput(parsedRequest);
  const metaMs = now() - requestStart;
  const requestId = (dependencies.requestId ?? randomUUID)();
  const precomputeAbortController = new AbortController();
  const runtimeAbortController = new AbortController();
  let connected =
    !request.aborted &&
    !response.destroyed;

  const cancelWork = () => {
    if (response.writableEnded) {
      return;
    }
    connected = false;
    precomputeAbortController.abort(new Error("Client disconnected."));
    runtimeAbortController.abort(new Error("Client disconnected."));
  };

  request.once("aborted", cancelWork);
  response.once("close", cancelWork);
  response.once("error", cancelWork);

  try {
    if (!connected) {
      return;
    }

    const precomputeStart = now();
    const precomputeTimeoutMs = dependencies.precomputeTimeoutMs ?? defaultPrecomputeTimeoutMs();
    let precomputed: WordTeachingPrecompute;
    
    // 1. Start the precompute database lookup (but do not await it yet)
    const precompute =
      dependencies.precompute ??
      ((input: SpellingCoachInput, signal: AbortSignal) =>
        warmWordTeachingPrecompute(input, { signal, requestId }));
        
    const precomputePromise = precompute(coachInput, precomputeAbortController.signal);

    // 2. Instantly flush headers and send the meta event
    response.writeHead(200, {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
      "X-Accel-Buffering": "no",
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
      "Access-Control-Allow-Headers":
        "Content-Type, Authorization, x-audio-filename",
    });
    response.flushHeaders();
    response.socket?.setNoDelay(true);

    const flushableResponse = response as FlushableResponse;
    const isCorrect = coachInput.missSignals.isCorrect;
    const shortFeedback = isCorrect 
      ? `Correct. You spelled '${coachInput.targetWord}' exactly right.`
      : undefined;

    writeSseEvent(flushableResponse, "meta", {
      requestId,
      isCorrect,
      timingMs: metaMs,
      targetWordMasked: true,
      targetWord: parsedRequest.targetWord,
      sayAloudTip: parsedRequest.targetWord ? getFriendlyPronunciationCue(parsedRequest.targetWord) ?? undefined : undefined,
      missAnalysis: deterministicMissAnalysis(coachInput),
      shortFeedback,
    });

    // 3. Now safely await the precompute before triggering the LLM
    try {
      precomputed = await withPrecomputeTimeout(
        precomputePromise,
        precomputeTimeoutMs,
        () =>
          precomputeAbortController.abort(
            new Error("Spelling coach precompute timed out."),
          ),
      );
    } catch (error) {
      logError(
        `[spelling-coach stream] requestId=${requestId} precompute failed after SSE started`,
        error,
      );
      // Throwing here will abort the stream (since headers are flushed, it drops the connection).
      // This preserves the original strict failure behavior.
      throw error;
    }

    const precomputedMs = now() - precomputeStart;
    logInfo(
      `[spelling-coach precompute timing] requestId=${requestId} word="${coachInput.targetWord}" total=${precomputedMs.toFixed(1)}ms`,
    );

    writeSseEvent(flushableResponse, "precomputed", {
      payload: precomputed,
      timingMs: precomputedMs,
    });

    if (isCorrect && coachInput.level === 1) {
      writeSseEvent(flushableResponse, "done", {
        requestId,
        targetWord: coachInput.targetWord,
      });
      response.end();
      return;
    }


    let runtimeCoachingMs = 0;

    if ((!coachInput.missSignals.isCorrect || (coachInput.level ?? 0) >= 2) && connected) {
      const runtimeStart = now();
      const parser = new RuntimeSectionParser();
      const sectionStarts = new Map<SpellingCoachStreamSection, number>();
      const sectionDurations: Partial<Record<SpellingCoachStreamSection, number>> = {};
      const firstMarkerTimeoutMs =
        dependencies.firstMarkerTimeoutMs ?? defaultFirstMarkerTimeoutMs();
      const sectionTimeoutMs =
        dependencies.sectionTimeoutMs ?? defaultSectionTimeoutMs();
      const runtimeTimeoutMs =
        dependencies.runtimeTimeoutMs ?? defaultRuntimeTimeoutMs();
      let firstMarkerTimer: ReturnType<typeof setTimeout> | undefined;
      let sectionTimer: ReturnType<typeof setTimeout> | undefined;
      let runtimeTimer: ReturnType<typeof setTimeout> | undefined;
      let timedOut = false;
      let firstMarkerLatencyMs: number | undefined;

      const clearSectionTimer = () => {
        if (sectionTimer) clearTimeout(sectionTimer);
        sectionTimer = undefined;
      };
      const clearRuntimeTimers = () => {
        if (firstMarkerTimer) clearTimeout(firstMarkerTimer);
        if (runtimeTimer) clearTimeout(runtimeTimer);
        clearSectionTimer();
      };
      const emitSectionError = (
        section: SpellingCoachStreamSection,
        code: string,
        message: string,
      ) => {
        const timingMs = now() - runtimeStart;
        logError(
          `[spelling-coach stream] requestId=${requestId} section=${section} parser failure code=${code}`,
          new Error(message),
        );
        writeSseEvent(flushableResponse, "section-error", {
          section,
          error: { code, message },
          timingMs,
        });
      };
      const startSectionTimer = (section: SpellingCoachStreamSection) => {
        clearSectionTimer();
        sectionTimer = setTimeout(() => {
          if (!connected || runtimeAbortController.signal.aborted) return;
          timedOut = true;
          emitSectionError(
            section,
            "SECTION_TIMEOUT",
            `Runtime coaching section \"${section}\" timed out.`,
          );
          runtimeAbortController.abort(timeoutError(section));
        }, sectionTimeoutMs);
      };
      const handleParserEvent = (event: ParserEvent) => {
        if (!connected) return;
        const timingMs = now() - runtimeStart;
        switch (event.type) {
          case "section-start":
            if (firstMarkerLatencyMs === undefined) {
              firstMarkerLatencyMs = timingMs;
              if (firstMarkerTimer) clearTimeout(firstMarkerTimer);
            }
            sectionStarts.set(event.section, now());
            startSectionTimer(event.section);
            writeSseEvent(flushableResponse, "section-start", {
              section: event.section,
              timingMs,
            });
            break;
          case "section-chunk":
            writeSseEvent(flushableResponse, "section-chunk", {
              section: event.section,
              text: event.text,
              timingMs,
            });
            break;
          case "section-complete": {
            clearSectionTimer();
            const started = sectionStarts.get(event.section) ?? runtimeStart;
            sectionDurations[event.section] = now() - started;
            writeSseEvent(flushableResponse, "section-complete", {
              section: event.section,
              timingMs,
            });
            break;
          }
          case "section-error":
            emitSectionError(
              event.section,
              event.code,
              event.message,
            );
            break;
        }
      };

      firstMarkerTimer = setTimeout(() => {
        if (!connected || runtimeAbortController.signal.aborted) return;
        timedOut = true;
        const section = parser.expectedSection() ?? "miss_analysis";
        emitSectionError(
          section,
          "FIRST_MARKER_TIMEOUT",
          `Runtime output did not emit the first section marker within ${firstMarkerTimeoutMs}ms.`,
        );
        runtimeAbortController.abort(timeoutError(section, "FIRST_MARKER_TIMEOUT"));
      }, firstMarkerTimeoutMs);
      runtimeTimer = setTimeout(() => {
        if (!connected || runtimeAbortController.signal.aborted) return;
        timedOut = true;
        const section =
          parser.currentSection() ?? parser.expectedSection() ?? "memory_tip";
        emitSectionError(
          section,
          "RUNTIME_TIMEOUT",
          `Runtime coaching did not finish within ${runtimeTimeoutMs}ms.`,
        );
        runtimeAbortController.abort(timeoutError(section, "RUNTIME_TIMEOUT"));
      }, runtimeTimeoutMs);

      try {
        const runtimeStream = dependencies.runRuntimeStream
          ? await dependencies.runRuntimeStream(
              coachInput,
              precomputed,
              runtimeAbortController.signal,
            )
          : dependencies.runRuntime
            ? runtimeOutputToMarkedText(
                dependencies.runRuntime(
                  coachInput,
                  runtimeAbortController.signal,
                ),
                runtimeAbortController.signal,
              )
            : await defaultRuntimeStream(
                coachInput,
                precomputed,
                runtimeAbortController.signal,
              );

        for await (const token of abortableRuntimeStream(
          runtimeStream,
          runtimeAbortController.signal,
        )) {
          if (!connected || runtimeAbortController.signal.aborted) break;
          for (const event of parser.feed(token)) {
            handleParserEvent(event);
          }
        }

        if (!timedOut && connected && !runtimeAbortController.signal.aborted) {
          for (const event of parser.finish()) {
            handleParserEvent(event);
          }
        }
      } catch (error) {
        if (connected && !runtimeAbortController.signal.aborted) {
          const section =
            parser.currentSection() ?? parser.expectedSection() ?? "miss_analysis";
          const failure = sectionFailure(error);
          emitSectionError(section, failure.code, failure.message);
        }
      } finally {
        clearRuntimeTimers();
      }

      runtimeCoachingMs = now() - runtimeStart;
      logInfo(
        `[spelling-coach stream metrics] requestId=${requestId} runtime=${runtimeCoachingMs.toFixed(1)}ms firstMarker=${firstMarkerLatencyMs === undefined ? "missing" : `${firstMarkerLatencyMs.toFixed(1)}ms`} sectionDurations=${JSON.stringify(sectionDurations)}`,
      );
    }

    if (connected) {
      writeSseEvent(flushableResponse, "done", {
        complete: true,
        targetWord: parsedRequest.targetWord,
        timings: {
          metaMs,
          precomputedMs,
          runtimeMs: runtimeCoachingMs,
          runtimeCoachingMs,
          totalMs: now() - requestStart,
        },
      });
      response.end();
      logInfo(
        `[spelling-coach stream] requestId=${requestId} complete requestDuration=${(now() - requestStart).toFixed(1)}ms precompute=${precomputedMs.toFixed(1)}ms runtime=${runtimeCoachingMs.toFixed(1)}ms`,
      );
    }
  } finally {
    if (!precomputeAbortController.signal.aborted) {
      precomputeAbortController.abort(new Error("SSE request completed."));
    }
    if (!response.writableEnded && !runtimeAbortController.signal.aborted) {
      runtimeAbortController.abort(
        new Error("SSE connection closed before completion."),
      );
    }
    request.removeListener("aborted", cancelWork);
    response.removeListener("close", cancelWork);
    response.removeListener("error", cancelWork);
  }
}
