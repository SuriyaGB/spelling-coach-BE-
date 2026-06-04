import { createSpellingCoachAgent, type DeepAgentLike } from "./agent.js";
import {
  buildDirectRuntimeSystemPrompt,
  createDirectSpellingCoachModel,
  type DirectModelLike,
} from "./directModel.js";
import { applyBlendPatternsToOutput, getLevelOnePatternNote } from "./blendPatterns.js";
import { buildLevelOneCoachingPrompt, buildSpellingCoachPrompt } from "./prompt.js";
import {
  parseLevelOneCoachingOutput,
  parseSpellingCoachInput,
  parseSpellingCoachOutput,
  type LevelOneCoachingOutput,
  type SpellingCoachInput,
  type SpellingCoachOutput,
} from "./schemas.js";
import type { ZodError } from "zod";
import { logInfo } from "./logging.js";
import { getWordByText } from "./wordCatalog.js";
import { warmWordTeachingPrecompute } from "./optimizedCoach.js";

export type RunSpellingCoachAgentOptions = {
  agent?: DeepAgentLike;
  directModel?: DirectModelLike;
  model?: string | object;
  maxValidationRetries?: number;
  enableTimingLogs?: boolean;
  runtime?: "deep_agent" | "direct";
};

type TimingEntry = {
  stage: string;
  durationMs: number;
};

function nowMs(): number {
  return performance.now();
}

function formatDuration(durationMs: number): string {
  return `${durationMs.toFixed(1)}ms`;
}

function logTimings(
  word: string,
  timings: TimingEntry[],
  totalDurationMs: number,
): void {
  const details = timings
    .map((timing) => `${timing.stage}=${formatDuration(timing.durationMs)}`)
    .join(" | ");

  logInfo(
    `[spelling-coach timing] word="${word}" total=${formatDuration(totalDurationMs)} | ${details}`,
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

function buildLevelOneOutput(
  input: SpellingCoachInput,
  coaching: LevelOneCoachingOutput,
  precomputedChunks: string[],
): SpellingCoachOutput {
  const chunks = precomputedChunks.filter(Boolean);
  const isCorrect = input.missSignals.isCorrect;
  const patternNote = getLevelOnePatternNote(
    input.targetWord,
    input.wordMetadata?.origin,
  );

  return parseSpellingCoachOutput({
    correctness: {
      isCorrect,
      reinforceSuccess: isCorrect,
    },
    missAnalysis: {
      summary: "",
      errorTypes: [],
      primaryErrorFocus: "",
      likelyWrongWordInterpretation: false,
      usedMeaningDisambiguationWell: false,
    },
    wordTeaching: {
      formTeaching: {
        summary: "",
        patterns: [],
        chunks: [],
        chunkReason: "",
        sayAloudFocus: "",
      },
      conceptTeaching: {
        summary: "",
        meaningFocus: "",
        originFocus: "",
        morphologyFocus: "",
        originLabels: [],
        morphologyLabels: [],
      },
    },
    errorRelevance: {
      mostRelevantToError: "unclear",
      confidence: 0,
      reason: "",
    },
    teachingDecision: {
      strategy: "chunking",
      primaryFocus: "",
      secondaryFocuses: [],
      confidence: 0,
      rationale: "",
    },
    coachingText: {
      shortFeedback: coaching.shortFeedback,
      fullExplanation: "",
      memoryTip: "",
      sayAloudTip: coaching.sayAloudTip,
    },
    wordBreakdown: {
      displayChunks: chunks,
      chunkReason: patternNote,
    },
    conceptLabels: {
      originLabels: [],
      patternLabels: [],
      morphologyLabels: [],
    },
    nextStep: {
      practiceFocus: "",
      shouldReviewSoon: !isCorrect,
      suggestedSimilarWordTypes: [],
    },
  });
}

export async function runSpellingCoachAgent(
  input: SpellingCoachInput,
  options: RunSpellingCoachAgentOptions = {},
): Promise<SpellingCoachOutput> {
  const enableTimingLogs = options.enableTimingLogs ?? true;
  const totalStart = nowMs();
  const timings: TimingEntry[] = [];

  const validateInputStart = nowMs();
  const validatedInput = parseSpellingCoachInput(input);
  timings.push({
    stage: "input_validation",
    durationMs: nowMs() - validateInputStart,
  });

  const runtime =
    options.runtime ??
    (process.env.SPELLING_COACH_RUNTIME === "direct" ? "direct" : "deep_agent");

  const agentStart = nowMs();
  const agent =
    runtime === "deep_agent"
      ? options.agent ?? (await createSpellingCoachAgent({ model: options.model }))
      : options.directModel ??
        (await createDirectSpellingCoachModel({ model: options.model }));
  timings.push({
    stage: runtime === "deep_agent" ? "agent_setup" : "direct_model_setup",
    durationMs: nowMs() - agentStart,
  });

  if (isLevelOnePractice(validatedInput)) {
    const precomputeLookupStart = nowMs();
    const precomputed = await warmWordTeachingPrecompute(validatedInput, {
      agent: options.agent,
      directModel: options.directModel,
      model: options.model,
      runtime,
    });
    timings.push({
      stage: "level1_word_breakdown_lookup",
      durationMs: nowMs() - precomputeLookupStart,
    });

    const minimalPromptStart = nowMs();
    const minimalPrompt = buildLevelOneCoachingPrompt(validatedInput);
    timings.push({
      stage: "level1_prompt_build",
      durationMs: nowMs() - minimalPromptStart,
    });

    const minimalOutput = await invokeLevelOneCoaching(
      validatedInput,
      minimalPrompt,
      precomputed.wordBreakdown.displayChunks,
      agent,
      runtime,
      timings,
      options.maxValidationRetries ?? 1,
    );

    if (enableTimingLogs) {
      logTimings(validatedInput.targetWord, timings, nowMs() - totalStart);
    }

    return minimalOutput;
  }

  const maxValidationRetries = options.maxValidationRetries ?? 1;
  const promptStart = nowMs();
  const messages: Array<{ role: "user" | "assistant"; content: string }> = [
    {
      role: "user",
      content: buildSpellingCoachPrompt(validatedInput),
    },
  ];
  timings.push({
    stage: "prompt_build",
    durationMs: nowMs() - promptStart,
  });

  let lastError: unknown;

  for (let attempt = 0; attempt <= maxValidationRetries; attempt += 1) {
    const invokeStart = nowMs();
    const response =
      runtime === "deep_agent"
        ? await agent.invoke({ messages })
        : await agent.invoke([
            {
              role: "system",
              content: buildDirectRuntimeSystemPrompt(),
            },
            ...messages,
          ]);
    timings.push({
      stage: `model_invoke_${attempt + 1}`,
      durationMs: nowMs() - invokeStart,
    });

    const extractStart = nowMs();
    const payload = extractAssistantPayload(response);
    timings.push({
      stage: `response_extract_${attempt + 1}`,
      durationMs: nowMs() - extractStart,
    });

    const outputValidationStart = nowMs();
    try {
      const parsedJson = parseStrictJson(payload);
      const validatedOutput = applyBlendPatternsToOutput(
        validatedInput.targetWord,
        parseSpellingCoachOutput(parsedJson),
        validatedInput.wordMetadata?.origin,
      );
      timings.push({
        stage: `output_validation_${attempt + 1}`,
        durationMs: nowMs() - outputValidationStart,
      });

      if (enableTimingLogs) {
        logTimings(
          validatedInput.targetWord,
          timings,
          nowMs() - totalStart,
        );
      }

      return validatedOutput;
    } catch (error) {
      lastError = error;
      timings.push({
        stage: `output_validation_${attempt + 1}`,
        durationMs: nowMs() - outputValidationStart,
      });

      if (attempt === maxValidationRetries) {
        if (enableTimingLogs) {
          logTimings(
            validatedInput.targetWord,
            timings,
            nowMs() - totalStart,
          );
        }
        throw error;
      }

      const repairPromptStart = nowMs();
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
      timings.push({
        stage: `repair_prompt_${attempt + 1}`,
        durationMs: nowMs() - repairPromptStart,
      });
    }
  }

  throw lastError instanceof Error
    ? lastError
    : new Error("Agent output failed validation.");
}

async function invokeLevelOneCoaching(
  input: SpellingCoachInput,
  prompt: string,
  precomputedChunks: string[],
  agent: DeepAgentLike | DirectModelLike,
  runtime: "deep_agent" | "direct",
  timings: TimingEntry[],
  maxValidationRetries: number,
): Promise<SpellingCoachOutput> {
  const messages: Array<{ role: "user" | "assistant"; content: string }> = [
    {
      role: "user",
      content: prompt,
    },
  ];
  let lastError: unknown;

  for (let attempt = 0; attempt <= maxValidationRetries; attempt += 1) {
    const invokeStart = nowMs();
    const response =
      runtime === "deep_agent"
        ? await (agent as DeepAgentLike).invoke({ messages })
        : await (agent as DirectModelLike).invoke([
            {
              role: "system",
              content: buildDirectRuntimeSystemPrompt(),
            },
            ...messages,
          ]);
    timings.push({
      stage: `level1_model_invoke_${attempt + 1}`,
      durationMs: nowMs() - invokeStart,
    });

    const extractStart = nowMs();
    const payload = extractAssistantPayload(response);
    timings.push({
      stage: `level1_response_extract_${attempt + 1}`,
      durationMs: nowMs() - extractStart,
    });

    const outputValidationStart = nowMs();
    try {
      const parsedJson = parseStrictJson(payload);
      const validatedOutput = buildLevelOneOutput(
        input,
        parseLevelOneCoachingOutput(parsedJson),
        precomputedChunks,
      );
      timings.push({
        stage: `level1_output_validation_${attempt + 1}`,
        durationMs: nowMs() - outputValidationStart,
      });
      return validatedOutput;
    } catch (error) {
      lastError = error;
      timings.push({
        stage: `level1_output_validation_${attempt + 1}`,
        durationMs: nowMs() - outputValidationStart,
      });

      if (attempt === maxValidationRetries) {
        throw error;
      }

      const repairPromptStart = nowMs();
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
      timings.push({
        stage: `level1_repair_prompt_${attempt + 1}`,
        durationMs: nowMs() - repairPromptStart,
      });
    }
  }

  throw lastError instanceof Error
    ? lastError
    : new Error("Level 1 coaching output failed validation.");
}
