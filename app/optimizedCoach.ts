import { createSpellingCoachAgent, type DeepAgentLike } from "./agent.js";
import {
  buildDirectRuntimeSystemPrompt,
  createDirectSpellingCoachModel,
  type DirectModelLike,
} from "./directModel.js";
import { applyBlendPatternsToPrecompute } from "./blendPatterns.js";
import {
  buildLevelOnePrecomputePrompt,
  buildMissOnlyPrompt,
  buildWordTeachingPrecomputePrompt,
} from "./prompt.js";
import { getConfiguredModelName } from "./modelConfig.js";
import {
  parseMissOnlyOutput,
  parseSpellingCoachInput,
  parseSpellingCoachOutput,
  parseWordTeachingPrecompute,
  type MissOnlyOutput,
  type SpellingCoachInput,
  type SpellingCoachOutput,
  type WordTeachingPrecompute,
} from "./schemas.js";
import type { ZodError } from "zod";
import { logInfo } from "./logging.js";
import { getWordByText } from "./wordCatalog.js";

type RuntimeMode = "deep_agent" | "direct";

type SharedOptions = {
  agent?: DeepAgentLike;
  directModel?: DirectModelLike;
  model?: string | object;
  runtime?: RuntimeMode;
};

type TimingEntry = {
  stage: string;
  durationMs: number;
};

const wordTeachingCache = new Map<string, Promise<WordTeachingPrecompute>>();

function resolveRuntime(runtime?: RuntimeMode): RuntimeMode {
  return runtime ??
    (process.env.SPELLING_COACH_RUNTIME === "direct" ? "direct" : "deep_agent");
}

function buildCacheKey(
  input: SpellingCoachInput,
  runtime: RuntimeMode,
  model?: string | object,
): string {
  const modelKey =
    typeof model === "string"
      ? model
      : model === undefined
        ? getConfiguredModelName()
        : "custom-model";
  return [runtime, modelKey, input.targetWord.toLowerCase()].join("|");
}

function nowMs(): number {
  return performance.now();
}

function formatDuration(durationMs: number): string {
  return `${durationMs.toFixed(1)}ms`;
}

function logTimings(
  prefix: string,
  word: string,
  timings: TimingEntry[],
  totalDurationMs: number,
): void {
  const details = timings
    .map((timing) => `${timing.stage}=${formatDuration(timing.durationMs)}`)
    .join(" | ");

  logInfo(
    `[${prefix}] word="${word}" total=${formatDuration(totalDurationMs)} | ${details}`,
  );
}

function extractTextContent(content: unknown): string {
  if (typeof content === "string") {
    return content;
  }

  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (typeof part === "string") {
          return part;
        }

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

function extractAssistantPayload(result: unknown): string {
  if (typeof result === "string") {
    return result.trim();
  }

  if (result && typeof result === "object") {
    const maybeContent = (result as { content?: unknown }).content;
    if (maybeContent !== undefined) {
      return extractTextContent(maybeContent).trim();
    }

    const messages = (result as { messages?: unknown }).messages;
    if (Array.isArray(messages) && messages.length > 0) {
      const lastMessage = messages[messages.length - 1] as { content?: unknown };
      return extractTextContent(lastMessage?.content).trim();
    }
  }

  throw new Error("Agent response did not contain assistant text content.");
}

function parseStrictJson(payload: string): unknown {
  const trimmed = payload.trim();
  if (!trimmed.startsWith("{") || !trimmed.endsWith("}")) {
    throw new Error("Agent output must be a single JSON object with no wrapper text.");
  }

  return JSON.parse(trimmed);
}

function formatValidationError(error: unknown): string {
  if (error && typeof error === "object" && "issues" in error) {
    return JSON.stringify((error as ZodError).issues, null, 2);
  }

  if (error instanceof Error) {
    return error.message;
  }

  return String(error);
}

function isLevelOnePractice(input: SpellingCoachInput): boolean {
  return getWordByText(input.targetWord)?.level === "1";
}

async function getRuntimeInvoker(
  options: SharedOptions = {},
): Promise<{ runtime: RuntimeMode; invoker: DeepAgentLike | DirectModelLike }> {
  const runtime = resolveRuntime(options.runtime);
  const invoker =
    runtime === "deep_agent"
      ? options.agent ?? (await createSpellingCoachAgent({ model: options.model }))
      : options.directModel ??
        (await createDirectSpellingCoachModel({ model: options.model }));

  return { runtime, invoker };
}

async function invokeValidatedJson<T>(
  prompt: string,
  parser: (output: unknown) => T,
  options: SharedOptions = {},
  timings?: TimingEntry[],
  stagePrefix = "model",
): Promise<T> {
  const { runtime, invoker } = await getRuntimeInvoker(options);
  const messages: Array<{ role: "user" | "assistant"; content: string }> = [{ role: "user", content: prompt }];
  let lastError: unknown;

  for (let attempt = 0; attempt <= 1; attempt += 1) {
    const invokeStart = nowMs();
    const response =
      runtime === "deep_agent"
        ? await (invoker as DeepAgentLike).invoke({ messages })
        : await (invoker as DirectModelLike).invoke([
            {
              role: "system",
              content: buildDirectRuntimeSystemPrompt(),
            },
            ...messages,
          ]);
    timings?.push({
      stage: `${stagePrefix}_invoke_${attempt + 1}`,
      durationMs: nowMs() - invokeStart,
    });

    const extractStart = nowMs();
    const payload = extractAssistantPayload(response);
    timings?.push({
      stage: `${stagePrefix}_extract_${attempt + 1}`,
      durationMs: nowMs() - extractStart,
    });

    const validationStart = nowMs();
    try {
      const parsed = parser(parseStrictJson(payload));
      timings?.push({
        stage: `${stagePrefix}_validation_${attempt + 1}`,
        durationMs: nowMs() - validationStart,
      });
      return parsed;
    } catch (error) {
      lastError = error;
      timings?.push({
        stage: `${stagePrefix}_validation_${attempt + 1}`,
        durationMs: nowMs() - validationStart,
      });
      if (attempt === 1) {
        throw error;
      }

      const repairStart = nowMs();
      messages.push({
        role: "assistant" as const,
        content: payload,
      });
      messages.push({
        role: "user" as const,
        content: [
          "Your previous response did not match the required JSON schema.",
          "Fix it and return one corrected JSON object only.",
          "Do not add markdown or explanation.",
          "Validation errors:",
          formatValidationError(error),
        ].join("\n\n"),
      });
      timings?.push({
        stage: `${stagePrefix}_repair_${attempt + 1}`,
        durationMs: nowMs() - repairStart,
      });
    }
  }

  throw lastError instanceof Error
    ? lastError
    : new Error("Agent output failed validation.");
}

export function hasWordTeachingPrecompute(
  input: SpellingCoachInput,
  options: SharedOptions = {},
): boolean {
  const validatedInput = parseSpellingCoachInput(input);
  const runtime = resolveRuntime(options.runtime);
  return wordTeachingCache.has(buildCacheKey(validatedInput, runtime, options.model));
}

export function warmWordTeachingPrecompute(
  input: SpellingCoachInput,
  options: SharedOptions = {},
): Promise<WordTeachingPrecompute> {
  const validatedInput = parseSpellingCoachInput(input);
  const runtime = resolveRuntime(options.runtime);
  const cacheKey = buildCacheKey(validatedInput, runtime, options.model);
  const cached = wordTeachingCache.get(cacheKey);

  if (cached) {
    return cached;
  }

  const promise = invokeValidatedJson(
    isLevelOnePractice(validatedInput)
      ? buildLevelOnePrecomputePrompt(validatedInput)
      : buildWordTeachingPrecomputePrompt(validatedInput),
    parseWordTeachingPrecompute,
    options,
    undefined,
    "precompute_model",
  )
    .then((precompute) =>
      applyBlendPatternsToPrecompute(
        validatedInput.targetWord,
        precompute,
        validatedInput.wordMetadata?.origin,
      ))
    .catch((error) => {
    wordTeachingCache.delete(cacheKey);
    throw error;
  });

  wordTeachingCache.set(cacheKey, promise);
  return promise;
}

export async function runSplitSpellingCoachAgent(
  input: SpellingCoachInput,
  options: SharedOptions = {},
): Promise<SpellingCoachOutput> {
  const totalStart = nowMs();
  const timings: TimingEntry[] = [];
  const validatedInput = parseSpellingCoachInput(input);
  const cacheLookupStart = nowMs();
  const precomputed = await warmWordTeachingPrecompute(validatedInput, options);
  timings.push({
    stage: "word_teaching_lookup",
    durationMs: nowMs() - cacheLookupStart,
  });
  const missOnly = await invokeValidatedJson<MissOnlyOutput>(
    buildMissOnlyPrompt(validatedInput, JSON.stringify(precomputed, null, 2)),
    parseMissOnlyOutput,
    options,
    timings,
    "miss_model",
  );

  const mergeStart = nowMs();
  const result = parseSpellingCoachOutput({
    ...missOnly,
    ...precomputed,
    });
  timings.push({
    stage: "merge_validation",
    durationMs: nowMs() - mergeStart,
  });

  logTimings(
    "spelling-coach split timing",
    validatedInput.targetWord,
    timings,
    nowMs() - totalStart,
  );

  return result;
}
