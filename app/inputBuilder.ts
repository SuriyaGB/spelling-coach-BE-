import { z } from "zod";
import {
  type SpellingCoachInput,
  type SpellingCoachOutput,
} from "./schemas.js";
import { getReferenceHints } from "./referenceData.js";
import {
  getStoredWordBreakdown,
  getWordByText,
  type SupportedLevel,
  type WordEntry,
} from "./wordCatalog.js";
import { buildDeterministicMissSignalFacts } from "./deterministicMissSignals.js";

export const CoachingRequestSchema = z
  .object({
    targetWord: z.string(),
    childAttempt: z.string(),
    childProfile: z.object({
      childId: z.string(),
      age: z.number().int().nonnegative(),
      grade: z.string(),
      spellingLevel: z.string(),
    }),
    supportsUsed: z
      .object({
        definitionViewed: z.boolean().optional(),
        exampleViewed: z.boolean().optional(),
        originViewed: z.boolean().optional(),
      })
      .optional(),
    sessionContext: z
      .object({
        mode: z.string().default("practice"),
        previousAttemptsOnThisWord: z.number().int().nonnegative().default(0),
        previousMissPatterns: z.array(z.string()).default([]),
        recentlyPracticedWords: z.array(z.string()).default([]),
      })
      .default({
        mode: "practice",
        previousAttemptsOnThisWord: 0,
        previousMissPatterns: [],
        recentlyPracticedWords: [],
      }),
    definition: z.string().optional(),
    exampleSentence: z.string().optional(),
    origin: z.string().optional(),
    partOfSpeech: z.string().optional(),
    level: z.number().optional(),
  });

export type CoachingRequest = z.infer<typeof CoachingRequestSchema>;

export const WordSearchQuerySchema = z
  .object({
    q: z.string().trim().min(2),
    mode: z.enum(["startsWith", "contains"]).default("startsWith"),
    limit: z.coerce.number().int().min(1).max(25).default(20),
  })
  .strict();

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export function maskWordInExampleSentence(
  exampleSentence: string,
  targetWord: string,
): string {
  if (!exampleSentence || !targetWord) {
    return exampleSentence;
  }

  const pattern = new RegExp(escapeRegExp(targetWord), "gi");
  return exampleSentence.replace(pattern, "*****");
}

export function maskWordInPublicText(
  text: string,
  targetWord: string,
): string {
  if (!text || !targetWord) {
    return text;
  }

  const pattern = new RegExp(escapeRegExp(targetWord), "gi");
  return text.replace(pattern, "*****");
}

export function buildWordResponse(word: WordEntry) {
  return {
    word: word.word,
    level: word.level,
    gradeBand: word.grade_band,
    difficulty: word.difficulty,
    origin: word.origin,
    definition: maskWordInPublicText(word.definition, word.word),
    exampleSentence: maskWordInPublicText(
      word.example_sentence,
      word.word,
    ),
    partOfSpeech: word.part_of_speech,
    pronunciation: "",
    patterns: word.patterns,
  };
}

function mapPatternMatch(
  pattern:
    | {
        label: string;
        matchedText?: string;
        matchedParts?: string[];
        alternateMatchedParts?: string[][];
      }
    | undefined,
) {
  if (!pattern) {
    return pattern;
  }

  return {
    label: pattern.label,
    ...(pattern.matchedText ? { matchedText: pattern.matchedText } : {}),
    ...(pattern.matchedParts ? { matchedParts: pattern.matchedParts } : {}),
    ...(pattern.alternateMatchedParts
      ? { alternateMatchedParts: pattern.alternateMatchedParts }
      : {}),
  };
}

function mapTeachingFact(
  fact:
    | {
        text: string;
        label: string;
        reason: string;
        source: "phoneme-validated" | "derived-rule";
        sounds_like?: string;
      }
    | undefined,
) {
  if (!fact) {
    return fact;
  }

  return {
    text: fact.text,
    label: fact.label,
    reason: fact.reason,
    source: fact.source,
    ...(fact.sounds_like ? { soundsLike: fact.sounds_like } : {}),
  };
}

export function buildDetailedWordResponse(word: WordEntry) {
  return {
    word: word.word,
    level: word.level,
    gradeBand: word.grade_band,
    difficulty: word.difficulty,
    origin: word.origin,
    definition: word.definition,
    exampleSentence: word.example_sentence,
    partOfSpeech: word.part_of_speech,
    patterns: word.patterns,
    commonMistakes: word.common_mistakes,
    coachTip: word.coach_tip,
    ...(word.word_breakdown
      ? {
          wordBreakdown: {
            displayChunks: word.word_breakdown.display_chunks,
            alternateDisplayChunks:
              word.word_breakdown.alternate_display_chunks,
            chunkReason: word.word_breakdown.chunk_reason,
            matchedPatterns: word.word_breakdown.matched_patterns.map(
              (pattern) => mapPatternMatch(pattern)!,
            ),
          },
        }
      : {}),
    ...(word.concept_labels
      ? {
          conceptLabels: {
            originLabels: word.concept_labels.origin_labels,
            patternLabels: word.concept_labels.pattern_labels,
            morphologyLabels: word.concept_labels.morphology_labels,
          },
        }
      : {}),
    ...(word.word_teaching
      ? {
          wordTeaching: {
            conceptTeaching: {
              summary: word.word_teaching.concept_teaching.summary,
              meaningFocus: word.word_teaching.concept_teaching.meaning_focus,
              originFocus: word.word_teaching.concept_teaching.origin_focus,
              morphologyFocus:
                word.word_teaching.concept_teaching.morphology_focus,
              originLabels: word.word_teaching.concept_teaching.origin_labels,
              morphologyLabels:
                word.word_teaching.concept_teaching.morphology_labels,
              relatedForms: word.word_teaching.concept_teaching.related_forms,
            },
          },
        }
      : {}),
    ...(word.phoneme_metadata
      ? {
          phonemeMetadata: {
            source: word.phoneme_metadata.source,
            phonemes: word.phoneme_metadata.phonemes,
            soundAwarePatterns: word.phoneme_metadata.sound_aware_patterns.map(
              (pattern) => mapPatternMatch(pattern)!,
            ),
            silentLetters: word.phoneme_metadata.silent_letters.map(
              (fact) => mapTeachingFact(fact)!,
            ),
            trickyParts: word.phoneme_metadata.tricky_parts.map(
              (fact) => mapTeachingFact(fact)!,
            ),
            friendlyChunks: word.phoneme_metadata.friendly_chunks,
            sayAloudTip: word.phoneme_metadata.say_aloud_tip ?? "",
            ...(word.phoneme_metadata.pronunciation_confidence
              ? {
                  pronunciationConfidence:
                    word.phoneme_metadata.pronunciation_confidence,
                }
              : {}),
          },
        }
      : {}),
  };
}

function levenshteinMatrix(left: string, right: string): number[][] {
  const rows = left.length + 1;
  const cols = right.length + 1;
  const matrix = Array.from({ length: rows }, () => Array<number>(cols).fill(0));

  for (let row = 0; row < rows; row += 1) {
    matrix[row][0] = row;
  }

  for (let col = 0; col < cols; col += 1) {
    matrix[0][col] = col;
  }

  for (let row = 1; row < rows; row += 1) {
    for (let col = 1; col < cols; col += 1) {
      const cost = left[row - 1] === right[col - 1] ? 0 : 1;
      matrix[row][col] = Math.min(
        matrix[row - 1][col] + 1,
        matrix[row][col - 1] + 1,
        matrix[row - 1][col - 1] + cost,
      );
    }
  }

  return matrix;
}

function diffWords(targetWord: string, childAttempt: string) {
  const target = targetWord.toLowerCase();
  const attempt = childAttempt.toLowerCase();
  const matrix = levenshteinMatrix(target, attempt);
  const missingLetters: string[] = [];
  const extraLetters: string[] = [];
  const substitutedLetters: string[] = [];
  const transposedLetters: string[] = [];

  let row = target.length;
  let col = attempt.length;

  while (row > 0 || col > 0) {
    if (
      row > 0 &&
      col > 0 &&
      target[row - 1] === attempt[col - 1]
    ) {
      row -= 1;
      col -= 1;
      continue;
    }

    if (
      row > 1 &&
      col > 1 &&
      target[row - 1] === attempt[col - 2] &&
      target[row - 2] === attempt[col - 1]
    ) {
      transposedLetters.push(`${attempt[col - 2]}${attempt[col - 1]}`);
      row -= 2;
      col -= 2;
      continue;
    }

    const current = matrix[row][col];
    if (
      row > 0 &&
      col > 0 &&
      matrix[row - 1][col - 1] + 1 === current
    ) {
      substitutedLetters.push(`${attempt[col - 1]} for ${target[row - 1]}`);
      row -= 1;
      col -= 1;
      continue;
    }

    if (row > 0 && matrix[row - 1][col] + 1 === current) {
      missingLetters.push(target[row - 1]);
      row -= 1;
      continue;
    }

    if (col > 0 && matrix[row][col - 1] + 1 === current) {
      extraLetters.push(attempt[col - 1]);
      col -= 1;
      continue;
    }

    break;
  }

  return {
    editDistance: matrix[target.length][attempt.length],
    missingLetters: missingLetters.reverse(),
    extraLetters: extraLetters.reverse(),
    substitutedLetters: substitutedLetters.reverse(),
    transposedLetters: transposedLetters.reverse(),
  };
}

function detectLikelyChunks(word: WordEntry): string[] {
  const storedBreakdown = getStoredWordBreakdown(word.word);
  if (storedBreakdown?.displayChunks.length) {
    return storedBreakdown.displayChunks;
  }

  const hints = getReferenceHints({
    targetWord: word.word,
    childAttempt: word.word,
    childProfile: {
      childId: "system",
      age: 0,
      grade: "system",
      spellingLevel: word.level,
    },
    wordMetadata: {
      definition: word.definition,
      origin: word.origin,
      partOfSpeech: word.part_of_speech,
      exampleSentence: word.example_sentence,
    },
    missSignals: {
      isCorrect: true,
      nearMiss: false,
      missingLetters: [],
      extraLetters: [],
      substitutedLetters: [],
      transposedLetters: [],
      repeatedLetterIssue: false,
      likelyRushed: false,
      editDistance: 0,
    },
    structuralHints: {
      syllables: [],
      likelyChunks: [],
      detectedPatterns: [],
    },
    sessionContext: {
      mode: "practice",
      previousAttemptsOnThisWord: 0,
      previousMissPatterns: [],
      recentlyPracticedWords: [],
    },
  });

  const prefix = hints.find((hint) => hint.role === "prefix")?.matchedForm ?? "";
  const suffix =
    hints.find((hint) => hint.role === "suffix_family")?.matchedForm ?? "";
  const chunks: string[] = [];

  if (prefix && word.word.toLowerCase().startsWith(prefix)) {
    chunks.push(word.word.slice(0, prefix.length));
  }

  const remainingAfterPrefix = chunks.length > 0 ? word.word.slice(chunks[0].length) : word.word;
  if (
    suffix &&
    remainingAfterPrefix.toLowerCase().endsWith(suffix) &&
    remainingAfterPrefix.length > suffix.length
  ) {
    const middle = remainingAfterPrefix.slice(
      0,
      remainingAfterPrefix.length - suffix.length,
    );
    if (middle) {
      chunks.push(middle);
    }
    chunks.push(
      remainingAfterPrefix.slice(remainingAfterPrefix.length - suffix.length),
    );
  } else if (chunks.length === 0 && word.patterns.length > 0) {
    chunks.push(word.patterns[0], word.word.slice(word.patterns[0].length));
  } else if (chunks.length > 0 && remainingAfterPrefix) {
    chunks.push(remainingAfterPrefix);
  }

  return chunks.filter(Boolean);
}

function detectLikelyPrefix(word: WordEntry): string | undefined {
  return getReferenceHints({
    targetWord: word.word,
    childAttempt: word.word,
    childProfile: {
      childId: "system",
      age: 0,
      grade: "system",
      spellingLevel: "system",
    },
    wordMetadata: {
      definition: word.definition,
      origin: word.origin,
      partOfSpeech: word.part_of_speech,
      exampleSentence: word.example_sentence,
    },
    missSignals: {
      isCorrect: true,
      nearMiss: false,
      missingLetters: [],
      extraLetters: [],
      substitutedLetters: [],
      transposedLetters: [],
      repeatedLetterIssue: false,
      likelyRushed: false,
      editDistance: 0,
    },
    structuralHints: {
      syllables: [],
      likelyChunks: [],
      detectedPatterns: [],
    },
    sessionContext: {
      mode: "practice",
      previousAttemptsOnThisWord: 0,
      previousMissPatterns: [],
      recentlyPracticedWords: [],
    },
  }).find((hint) => hint.role === "prefix")?.matchedForm;
}

function detectLikelySuffix(word: WordEntry): string | undefined {
  return getReferenceHints({
    targetWord: word.word,
    childAttempt: word.word,
    childProfile: {
      childId: "system",
      age: 0,
      grade: "system",
      spellingLevel: "system",
    },
    wordMetadata: {
      definition: word.definition,
      origin: word.origin,
      partOfSpeech: word.part_of_speech,
      exampleSentence: word.example_sentence,
    },
    missSignals: {
      isCorrect: true,
      nearMiss: false,
      missingLetters: [],
      extraLetters: [],
      substitutedLetters: [],
      transposedLetters: [],
      repeatedLetterIssue: false,
      likelyRushed: false,
      editDistance: 0,
    },
    structuralHints: {
      syllables: [],
      likelyChunks: [],
      detectedPatterns: [],
    },
    sessionContext: {
      mode: "practice",
      previousAttemptsOnThisWord: 0,
      previousMissPatterns: [],
      recentlyPracticedWords: [],
    },
  }).find((hint) => hint.role === "suffix_family")?.matchedForm;
}

export function buildSpellingCoachInput(
  request: CoachingRequest,
): SpellingCoachInput {
  const parsedRequest = CoachingRequestSchema.parse(request);
  let word = getWordByText(parsedRequest.targetWord);

  if (!word) {
    // If the word isn't in catalog, build a dynamic custom WordEntry
    word = {
      word: parsedRequest.targetWord,
      level: parsedRequest.level === 0 ? "custom" : (parsedRequest.level ? String(parsedRequest.level) : "custom") as any,
      grade_band: "custom",
      difficulty: "custom",
      origin: parsedRequest.origin || "",
      definition: parsedRequest.definition || "",
      example_sentence: parsedRequest.exampleSentence || "",
      patterns: [],
      common_mistakes: [],
      coach_tip: "",
      part_of_speech: parsedRequest.partOfSpeech || "noun",
    };
  }

  return buildSpellingCoachInputFromWordEntry(word, parsedRequest);
}

export function buildSpellingCoachInputFromWordEntry(
  word: WordEntry,
  request: CoachingRequest,
): SpellingCoachInput {
  const parsedRequest = CoachingRequestSchema.parse(request);

  const diff = diffWords(word.word, parsedRequest.childAttempt);
  const deterministicFacts = buildDeterministicMissSignalFacts(
    word,
    parsedRequest.childAttempt,
    diff,
  );
  const isCorrect =
    word.word.toLowerCase() === parsedRequest.childAttempt.toLowerCase();
  const storedBreakdown = getStoredWordBreakdown(word.word);

  return {
    targetWord: word.word,
    childAttempt: parsedRequest.childAttempt,
    childProfile: parsedRequest.childProfile,
    level: !isNaN(Number(word.level)) ? Number(word.level) : undefined,
    wordMetadata: {
      definition: word.definition,
      origin: word.origin,
      partOfSpeech: word.part_of_speech,
      exampleSentence: word.example_sentence,
    },
    missSignals: {
      isCorrect,
      nearMiss: !isCorrect && diff.editDistance <= 2,
      missingLetters: diff.missingLetters,
      extraLetters: diff.extraLetters,
      substitutedLetters: diff.substitutedLetters,
      transposedLetters: diff.transposedLetters,
      repeatedLetterIssue: deterministicFacts.repeatedLetterIssue,
      vowelSubstitutionPairs: deterministicFacts.vowelSubstitutionPairs,
      doubleLetterMismatch: deterministicFacts.doubleLetterMismatch,
      chunkMismatchFacts: deterministicFacts.chunkMismatchFacts,
      endingConfusionFacts: deterministicFacts.endingConfusionFacts,
      silentLetterFactsTouched: deterministicFacts.silentLetterFactsTouched,
      trickyPartFactsTouched: deterministicFacts.trickyPartFactsTouched,
      wrongWordInterpretationHints:
        deterministicFacts.wrongWordInterpretationHints,
      deterministicLikelyWrongWordInterpretation:
        deterministicFacts.deterministicLikelyWrongWordInterpretation,
      likelyRushed:
        !isCorrect &&
        diff.editDistance <= 2 &&
        (diff.missingLetters.length > 0 || diff.transposedLetters.length > 0),
      editDistance: diff.editDistance,
    },
    structuralHints: {
      syllables: [],
      likelyChunks: storedBreakdown?.displayChunks ?? detectLikelyChunks(word),
      detectedPatterns: [...word.patterns],
      likelyPrefix: detectLikelyPrefix(word),
      likelySuffix: detectLikelySuffix(word),
    },
    sessionContext: parsedRequest.sessionContext,
  };
}

export function buildWordPrecomputeInput(targetWord: string | WordEntry): SpellingCoachInput {
  const word = typeof targetWord === "string" ? getWordByText(targetWord) : targetWord;

  if (!word) {
    throw new Error(`Unknown target word: ${typeof targetWord === "string" ? targetWord : targetWord.word}`);
  }

  return buildWordPrecomputeInputFromWordEntry(word);
}

export function buildWordPrecomputeInputFromWordEntry(
  word: WordEntry,
): SpellingCoachInput {

  const storedBreakdown = getStoredWordBreakdown(word.word);

  return {
    targetWord: word.word,
    childAttempt: word.word,
    childProfile: {
      childId: "system",
      age: 0,
      grade: "system",
      spellingLevel: "system",
    },
    wordMetadata: {
      definition: word.definition,
      origin: word.origin,
      partOfSpeech: word.part_of_speech,
      exampleSentence: word.example_sentence,
    },
    missSignals: {
      isCorrect: true,
      nearMiss: false,
      missingLetters: [],
      extraLetters: [],
      substitutedLetters: [],
      transposedLetters: [],
      repeatedLetterIssue: false,
      likelyRushed: false,
      editDistance: 0,
    },
    structuralHints: {
      syllables: [],
      likelyChunks: storedBreakdown?.displayChunks ?? detectLikelyChunks(word),
      detectedPatterns: [...word.patterns],
      likelyPrefix: detectLikelyPrefix(word),
      likelySuffix: detectLikelySuffix(word),
    },
    sessionContext: {
      mode: "practice",
      previousAttemptsOnThisWord: 0,
      previousMissPatterns: [],
      recentlyPracticedWords: [],
    },
  };
}

export const LevelQuerySchema = z.object({
  level: z.preprocess(
    (value) => {
      if (value === null || value === undefined) {
        return undefined;
      }

      const normalized = String(value).trim();
      if (!normalized || normalized === "NaN" || normalized === "undefined" || normalized === "null") {
        return undefined;
      }

      return normalized;
    },
    z.enum(["1", "2", "3"]).optional(),
  ),
  customListId: z.string().optional(),
  foreignOrigin: z.preprocess(
    (value) => {
      if (value === null || value === undefined) {
        return undefined;
      }

      const normalized = String(value).trim();
      if (!normalized || normalized === "undefined" || normalized === "null") {
        return undefined;
      }

      return normalized;
    },
    z.string().optional(),
  ),
  exclude: z
    .string()
    .optional()
    .transform((value) =>
      value ? value.split(",").map((item) => item.trim()).filter(Boolean) : [],
    ),
}).superRefine((value, context) => {
  if (!value.level && !value.customListId && !value.foreignOrigin) {
    context.addIssue({
      code: z.ZodIssueCode.custom,
      message: "Either level, customListId, or foreignOrigin is required.",
      path: ["level"],
    });
  }
});

export type LevelQuery = {
  level?: SupportedLevel;
  customListId?: string;
  foreignOrigin?: string;
  exclude?: string[];
};
