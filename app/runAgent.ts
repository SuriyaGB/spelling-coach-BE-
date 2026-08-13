import { createSpellingCoachAgent, type DeepAgentLike } from "./agent.js";
import {
  buildDirectRuntimeSystemPrompt,
  createDirectSpellingCoachModel,
  type DirectModelLike,
} from "./directModel.js";
import { normalizeSpellingCoachOutputChunkReason } from "./chunkReason.js";
import { getFriendlyPronunciationCue } from "./friendlyPronunciation.js";
import {
  normalizeMissAnalysisErrorTypes,
  sanitizeMissAnalysis,
} from "./missAnalysis.js";
import { applyNewPatternsToOutput } from "./newPatternMatcher.js";
import {
  buildLevelOneCoachingPrompt,
  buildSpellingCoachPrompt,
  isNextStepEnabled,
  isRuntimeConceptTeachingEnabled,
} from "./prompt.js";
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
import { getStoredWordTeachingOnlyPrecompute, getWordByText } from "./wordCatalog.js";
import { warmWordTeachingPrecompute } from "./optimizedCoach.js";

export type RunSpellingCoachAgentOptions = {
  agent?: DeepAgentLike;
  directModel?: DirectModelLike;
  model?: string | object;
  maxValidationRetries?: number;
  enableTimingLogs?: boolean;
  runtime?: "deep_agent" | "direct";
  signal?: AbortSignal;
  requestId?: string;
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
  requestId?: string,
): void {
  const details = timings
    .map((timing) => `${timing.stage}=${formatDuration(timing.durationMs)}`)
    .join(" | ");

  logInfo(
    `[spelling-coach timing]${requestId ? ` requestId=${requestId}` : ""} word="${word}" total=${formatDuration(totalDurationMs)} | ${details}`,
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

function clearExplanationForCorrectSpelling(
  output: SpellingCoachOutput,
): SpellingCoachOutput {
  if (!output.correctness.isCorrect) {
    return output;
  }

  output.coachingText.fullExplanation = "";
  return output;
}

function normalizeNextStepFeature(
  output: SpellingCoachOutput,
): SpellingCoachOutput {
  if (isNextStepEnabled()) {
    return output;
  }

  output.nextStep = {
    practiceFocus: "",
    shouldReviewSoon: false,
    suggestedSimilarWordTypes: [],
  };
  return output;
}

function normalizeRuntimeConceptTeachingFeature(
  targetWord: string,
  output: SpellingCoachOutput,
): SpellingCoachOutput {
  if (isRuntimeConceptTeachingEnabled()) {
    return output;
  }

  const stored = getStoredWordTeachingOnlyPrecompute(targetWord);
  if (stored) {
    output.wordTeaching = stored.wordTeaching;
    output.conceptLabels = stored.conceptLabels;
    return output;
  }

  output.wordTeaching = {
    conceptTeaching: {
      summary: "",
      meaningFocus: "",
      originFocus: "",
      morphologyFocus: "",
      originLabels: [],
      morphologyLabels: [],
      relatedForms: [],
    },
  };
  output.conceptLabels = {
    originLabels: [],
    patternLabels: [],
    morphologyLabels: [],
  };
  return output;
}

function buildLevelOneOutput(
  input: SpellingCoachInput,
  coaching: LevelOneCoachingOutput,
  precomputedWordBreakdown: SpellingCoachOutput["wordBreakdown"],
): SpellingCoachOutput {
  const chunks = precomputedWordBreakdown.displayChunks.filter(Boolean);
  const isCorrect = input.missSignals.isCorrect;
  const friendlyPronunciationCue =
    getFriendlyPronunciationCue(input.targetWord) ?? coaching.sayAloudTip;

  return parseSpellingCoachOutput({
    correctness: {
      isCorrect,
      reinforceSuccess: isCorrect,
    },
    missAnalysis: {
      summary: "",
      primaryErrorType: null,
      secondaryErrorTypes: [],
      errorTypeEvidence: {},
      primaryErrorFocus: "",
      likelyWrongWordInterpretation: false,
      usedMeaningDisambiguationWell: false,
    },
    wordTeaching: {
      conceptTeaching: {
        summary: "",
        meaningFocus: "",
        originFocus: "",
        morphologyFocus: "",
        originLabels: [],
        morphologyLabels: [],
        relatedForms: [],
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
      sayAloudTip: friendlyPronunciationCue,
    },
    wordBreakdown: {
      displayChunks: chunks,
      alternateDisplayChunks: precomputedWordBreakdown.alternateDisplayChunks,
      chunkReason: precomputedWordBreakdown.chunkReason,
      matchedPatterns: precomputedWordBreakdown.matchedPatterns,
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
      signal: options.signal,
      requestId: options.requestId,
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
      precomputed.wordBreakdown,
      agent,
      runtime,
      timings,
      options.maxValidationRetries ?? 1,
      options.signal,
    );

    if (enableTimingLogs) {
      logTimings(
        validatedInput.targetWord,
        timings,
        nowMs() - totalStart,
        options.requestId,
      );
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
        ? await agent.invoke(
            { messages },
            options.signal ? { signal: options.signal } : undefined,
          )
        : await agent.invoke([
            {
              role: "system",
              content: buildDirectRuntimeSystemPrompt(),
            },
            ...messages,
          ], options.signal ? { signal: options.signal } : undefined);
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
      const schemaValidationStart = nowMs();
      const parsedOutput = parseSpellingCoachOutput(parsedJson);
      const validatedOutput = applyNewPatternsToOutput(
        validatedInput.targetWord,
        normalizeSpellingCoachOutputChunkReason(parsedOutput),
      );
      timings.push({
        stage: `schema_and_pattern_validation_${attempt + 1}`,
        durationMs: nowMs() - schemaValidationStart,
      });

      const missNormalizationStart = nowMs();
      normalizeMissAnalysisErrorTypes(validatedInput, validatedOutput);
      timings.push({
        stage: `miss_analysis_normalization_${attempt + 1}`,
        durationMs: nowMs() - missNormalizationStart,
      });

      const sanitizationStart = nowMs();
      sanitizeMissAnalysis(validatedOutput);
      timings.push({
        stage: `miss_text_sanitization_${attempt + 1}`,
        durationMs: nowMs() - sanitizationStart,
      });

      const conceptTeachingNormalizationStart = nowMs();
      normalizeRuntimeConceptTeachingFeature(
        validatedInput.targetWord,
        validatedOutput,
      );
      timings.push({
        stage: `concept_teaching_normalization_${attempt + 1}`,
        durationMs: nowMs() - conceptTeachingNormalizationStart,
      });

      const responseCleanupStart = nowMs();
      clearExplanationForCorrectSpelling(validatedOutput);
      normalizeNextStepFeature(validatedOutput);
      timings.push({
        stage: `response_cleanup_${attempt + 1}`,
        durationMs: nowMs() - responseCleanupStart,
      });

      const pronunciationCueStart = nowMs();
      const friendlyPronunciationCue = getFriendlyPronunciationCue(
        validatedInput.targetWord,
      );
      if (friendlyPronunciationCue) {
        validatedOutput.coachingText.sayAloudTip = friendlyPronunciationCue;
      }
      timings.push({
        stage: `pronunciation_cue_override_${attempt + 1}`,
        durationMs: nowMs() - pronunciationCueStart,
      });
      timings.push({
        stage: `output_validation_${attempt + 1}`,
        durationMs: nowMs() - outputValidationStart,
      });

      if (enableTimingLogs) {
        logTimings(
          validatedInput.targetWord,
          timings,
          nowMs() - totalStart,
          options.requestId,
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
            options.requestId,
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
  precomputedWordBreakdown: SpellingCoachOutput["wordBreakdown"],
  agent: DeepAgentLike | DirectModelLike,
  runtime: "deep_agent" | "direct",
  timings: TimingEntry[],
  maxValidationRetries: number,
  signal?: AbortSignal,
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
        ? await (agent as DeepAgentLike).invoke(
            { messages },
            signal ? { signal } : undefined,
          )
        : await (agent as DirectModelLike).invoke([
            {
              role: "system",
              content: buildDirectRuntimeSystemPrompt(),
            },
            ...messages,
          ], signal ? { signal } : undefined);
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
        precomputedWordBreakdown,
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
