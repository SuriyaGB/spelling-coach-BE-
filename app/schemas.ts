import { z } from "zod";

export const ChildProfileSchema = z.object({
  childId: z.string(),
  age: z.number().int().nonnegative(),
  grade: z.string(),
  spellingLevel: z.string(),
});

export const WordMetadataSchema = z
  .object({
    definition: z.string().optional(),
    origin: z.string().optional(),
    partOfSpeech: z.string().optional(),
    exampleSentence: z.string().optional(),
    pronunciation: z.string().optional(),
  })
  .strict()
  .optional();

export const MissSignalsSchema = z
  .object({
    isCorrect: z.boolean(),
    nearMiss: z.boolean(),
    missingLetters: z.array(z.string()),
    extraLetters: z.array(z.string()),
    substitutedLetters: z.array(z.string()),
    transposedLetters: z.array(z.string()),
    repeatedLetterIssue: z.boolean(),
    vowelSubstitutionPairs: z
      .array(
        z
          .object({
            actual: z.string(),
            expected: z.string(),
          })
          .strict(),
      )
      .optional(),
    doubleLetterMismatch: z
      .object({
        detected: z.boolean(),
        missingFromDouble: z.array(z.string()),
        extraDouble: z.array(z.string()),
        affectedLetters: z.array(z.string()),
      })
      .strict()
      .optional(),
    chunkMismatchFacts: z
      .array(
        z
          .object({
            expectedChunk: z.string(),
            observedFragment: z.string(),
            phoneticRewrite: z.boolean(),
          })
          .strict(),
      )
      .optional(),
    endingConfusionFacts: z
      .array(
        z
          .object({
            suffix: z.string(),
            attemptedEnding: z.string(),
            phoneticRewrite: z.boolean(),
          })
          .strict(),
      )
      .optional(),
    silentLetterFactsTouched: z
      .array(
        z
          .object({
            text: z.string(),
            label: z.string(),
            reason: z.string(),
            sounds_like: z.string().optional(),
            source: z.string().optional(),
          })
          .strict(),
      )
      .optional(),
    trickyPartFactsTouched: z
      .array(
        z
          .object({
            text: z.string(),
            label: z.string(),
            reason: z.string(),
            sounds_like: z.string().optional(),
            phoneticRewrite: z.boolean(),
            source: z.string().optional(),
          })
          .strict(),
      )
      .optional(),
    wrongWordInterpretationHints: z
      .object({
        targetNormalized: z.string(),
        attemptNormalized: z.string(),
        sharedPrefixLength: z.number().int().nonnegative(),
        sharedSuffixLength: z.number().int().nonnegative(),
        longestCommonSubsequenceLength: z.number().int().nonnegative(),
        longestCommonSubsequenceRatio: z.number().nonnegative(),
        bigramOverlapRatio: z.number().nonnegative(),
        trigramOverlapRatio: z.number().nonnegative(),
        substantialStructuralOverlap: z.boolean(),
      })
      .strict()
      .optional(),
    deterministicLikelyWrongWordInterpretation: z.boolean().optional(),
    likelyRushed: z.boolean(),
    editDistance: z.number().nonnegative(),
  })
  .strict();

export const StructuralHintsSchema = z
  .object({
    syllables: z.array(z.string()).default([]),
    likelyChunks: z.array(z.string()).default([]),
    detectedPatterns: z.array(z.string()).default([]),
    likelyPrefix: z.string().optional(),
    likelySuffix: z.string().optional(),
  })
  .strict();

export const SessionContextSchema = z
  .object({
    mode: z.string(),
    previousAttemptsOnThisWord: z.number().int().nonnegative(),
    previousMissPatterns: z.array(z.string()),
    recentlyPracticedWords: z.array(z.string()),
  })
  .strict();

export const SpellingCoachInputSchema = z
  .object({
    targetWord: z.string().min(1),
    childAttempt: z.string(),
    level: z.number().optional(),
    childProfile: ChildProfileSchema,
    wordMetadata: WordMetadataSchema,
    missSignals: MissSignalsSchema,
    structuralHints: StructuralHintsSchema,
    sessionContext: SessionContextSchema,
  })
  .strict();

export const TeachingStrategySchema = z.enum([
  "concept",
  "pattern",
  "chunking",
  "memory",
  "mixed",
]);

export const ERROR_TYPE_VALUES = [
  "far_from_target",
  "missing_letter",
  "extra_letter",
  "letter_substitution",
  "letter_transposition",
  "double_letter_error",
  "phonetic_spelling",
  "vowel_confusion",
  "ending_confusion",
  "chunk_mismatch",
  "consonant_cluster_error",
  "silent_letter_error",
  "pattern_rule_mismatch",
  "morphology_error",
  "likely_rushed",
  "wrong_word_interpretation",
] as const;

export const ErrorTypeSchema = z.enum(ERROR_TYPE_VALUES);

export const ErrorRelevanceSchema = z.enum([
  "form",
  "concept",
  "mixed",
  "unclear",
]);

export const WordTeachingSchema = z
  .object({
    conceptTeaching: z
      .object({
        summary: z.string(),
        meaningFocus: z.string(),
        originFocus: z.string(),
        morphologyFocus: z.string(),
        originLabels: z.array(z.string()),
        morphologyLabels: z.array(z.string()),
        relatedForms: z.array(z.string()).default([]),
      })
      .strict(),
  })
  .strict();

export const PatternMatchSchema = z
  .object({
    label: z.string(),
    matchedText: z.string().optional(),
    matchedParts: z.array(z.string()).optional(),
    alternateMatchedParts: z.array(z.array(z.string())).optional(),
  })
  .strict();

export const WordBreakdownSchema = z
  .object({
    displayChunks: z.array(z.string()),
    alternateDisplayChunks: z.array(z.array(z.string())).default([]),
    chunkReason: z.string(),
    matchedPatterns: z.array(PatternMatchSchema).default([]),
  })
  .strict();

export const ConceptLabelsSchema = z
  .object({
    originLabels: z.array(z.string()),
    patternLabels: z.array(z.string()),
    morphologyLabels: z.array(z.string()),
  })
  .strict();

export const CorrectnessSchema = z
  .object({
    isCorrect: z.boolean(),
    reinforceSuccess: z.boolean(),
  })
  .strict();

export const MissAnalysisSchema = z
  .object({
    summary: z.string(),
    primaryErrorType: ErrorTypeSchema.nullable(),
    secondaryErrorTypes: z.array(ErrorTypeSchema),
    errorTypeEvidence: z.record(z.string(), z.string()),
    primaryErrorFocus: z.string(),
    likelyWrongWordInterpretation: z.boolean(),
    usedMeaningDisambiguationWell: z.boolean(),
  })
  .strict();

export const ErrorRelevanceDetailSchema = z
  .object({
    mostRelevantToError: ErrorRelevanceSchema,
    confidence: z.number().min(0).max(1),
    reason: z.string(),
  })
  .strict();

export const TeachingDecisionDetailSchema = z
  .object({
    strategy: TeachingStrategySchema,
    primaryFocus: z.string(),
    secondaryFocuses: z.array(z.string()),
    confidence: z.number().min(0).max(1),
    rationale: z.string(),
  })
  .strict();

export const CoachingTextSchema = z
  .object({
    shortFeedback: z.string(),
    fullExplanation: z.string(),
    memoryTip: z.string(),
    sayAloudTip: z.string(),
  })
  .strict();

export const NextStepSchema = z
  .object({
    practiceFocus: z.string(),
    shouldReviewSoon: z.boolean(),
    suggestedSimilarWordTypes: z.array(z.string()),
  })
  .strict();

export const WordTeachingPrecomputeSchema = z
  .object({
    wordTeaching: WordTeachingSchema,
    wordBreakdown: WordBreakdownSchema,
    conceptLabels: ConceptLabelsSchema,
  })
  .strict();

export const WordTeachingOnlyPrecomputeSchema = z
  .object({
    wordTeaching: z
      .object({
        conceptTeaching: WordTeachingSchema.shape.conceptTeaching,
      })
      .strict(),
    conceptLabels: ConceptLabelsSchema,
  })
  .strict();

export const RelatedFormsOnlyPrecomputeSchema = z
  .object({
    relatedForms: z.array(z.string()).default([]),
  })
  .strict();

export const MissOnlyOutputSchema = z
  .object({
    correctness: CorrectnessSchema,
    missAnalysis: MissAnalysisSchema,
    errorRelevance: ErrorRelevanceDetailSchema,
    teachingDecision: TeachingDecisionDetailSchema,
    coachingText: CoachingTextSchema,
    nextStep: NextStepSchema,
  })
  .strict();

export const DeterministicPatternFilterOutputSchema = z
  .object({
    keptDescriptions: z.array(z.string()),
  })
  .strict();

export const LevelOneCoachingOutputSchema = z
  .object({
    shortFeedback: z.string(),
    sayAloudTip: z.string(),
  })
  .strict();

export const SpellingCoachOutputSchema = z
  .object({
    correctness: CorrectnessSchema,
    missAnalysis: MissAnalysisSchema,
    wordTeaching: WordTeachingSchema,
    errorRelevance: ErrorRelevanceDetailSchema,
    teachingDecision: TeachingDecisionDetailSchema,
    coachingText: CoachingTextSchema,
    wordBreakdown: WordBreakdownSchema,
    conceptLabels: ConceptLabelsSchema,
    nextStep: NextStepSchema,
  })
  .strict();

export type SpellingCoachInput = z.infer<typeof SpellingCoachInputSchema>;
export type WordBreakdown = z.input<typeof WordBreakdownSchema>;
export type ParsedWordBreakdown = z.infer<typeof WordBreakdownSchema>;
export type PatternMatch = z.infer<typeof PatternMatchSchema>;
export type SpellingCoachOutput = Omit<
  z.infer<typeof SpellingCoachOutputSchema>,
  "wordBreakdown"
> & {
  wordBreakdown: WordBreakdown;
};
export type WordTeachingPrecompute = Omit<
  z.infer<typeof WordTeachingPrecomputeSchema>,
  "wordBreakdown"
> & {
  wordBreakdown: WordBreakdown;
};
export type WordTeachingOnlyPrecompute = z.infer<
  typeof WordTeachingOnlyPrecomputeSchema
>;
export type RelatedFormsOnlyPrecompute = z.infer<
  typeof RelatedFormsOnlyPrecomputeSchema
>;
export type MissOnlyOutput = z.infer<typeof MissOnlyOutputSchema>;
export type DeterministicPatternFilterOutput = z.infer<
  typeof DeterministicPatternFilterOutputSchema
>;
export type LevelOneCoachingOutput = z.infer<typeof LevelOneCoachingOutputSchema>;

export function parseSpellingCoachInput(input: unknown): SpellingCoachInput {
  return SpellingCoachInputSchema.parse(input);
}

export function parseSpellingCoachOutput(output: unknown): SpellingCoachOutput {
  return SpellingCoachOutputSchema.parse(output);
}

export function parseRelatedFormsOnlyPrecompute(
  output: unknown,
): RelatedFormsOnlyPrecompute {
  return RelatedFormsOnlyPrecomputeSchema.parse(output);
}

export function parseWordTeachingPrecompute(
  output: unknown,
): WordTeachingPrecompute {
  return WordTeachingPrecomputeSchema.parse(output);
}

export function parseWordTeachingOnlyPrecompute(
  output: unknown,
): WordTeachingOnlyPrecompute {
  return WordTeachingOnlyPrecomputeSchema.parse(output);
}

export function parseWordBreakdown(
  output: unknown,
): ParsedWordBreakdown {
  return WordBreakdownSchema.parse(output);
}

export function parseMissOnlyOutput(output: unknown): MissOnlyOutput {
  return MissOnlyOutputSchema.parse(output);
}

export function parseDeterministicPatternFilterOutput(
  output: unknown,
): DeterministicPatternFilterOutput {
  return DeterministicPatternFilterOutputSchema.parse(output);
}

export function parseLevelOneCoachingOutput(
  output: unknown,
): LevelOneCoachingOutput {
  return LevelOneCoachingOutputSchema.parse(output);
}
