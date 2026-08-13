import test from "node:test";
import assert from "node:assert/strict";
import {
  buildSpellingCoachInput,
  buildDetailedWordResponse,
  buildWordPrecomputeInput,
  buildWordResponse,
  LevelQuerySchema,
  maskWordInExampleSentence,
  WordSearchQuerySchema,
} from "./inputBuilder.js";
import {
  normalizeSpellingCoachOutputChunkReason,
  normalizeWordTeachingPrecomputeChunkReason,
} from "./chunkReason.js";
import {
  buildSoundAwareTipNote,
  buildStoredSayAloudTip,
  deriveFriendlyPronunciationChunks,
  deriveFriendlyPronunciation,
  getFriendlyPronunciationCue,
} from "./friendlyPronunciation.js";
import { derivePhonemeTeachingFacts } from "./phonemeTeachingFacts.js";
import { buildDefaultTtsInstructions } from "./pronunciation.js";
import {
  auditFriendlyPronunciation,
  buildPronunciationReviewBuckets,
} from "./pronunciationConfidence.js";
import {
  applyNewPatternsToOutput,
  getNewMatchedPatterns,
} from "./newPatternMatcher.js";
import { normalizeMissAnalysisErrorTypes } from "./missAnalysis.js";
import { getSoundAwareMatchedPatterns } from "./soundAwarePatterns.js";
import { importCustomWords } from "./customWordImport.js";
import { importForeignOriginWords } from "./foreignOriginImport.js";
import {
  hasWordTeachingPrecompute,
  runSplitSpellingCoachAgent,
  warmWordTeachingPrecompute,
} from "./optimizedCoach.js";
import {
  buildMissOnlyPrompt,
  buildRelatedFormsOnlyPrecomputePrompt,
  buildSpellingCoachPrompt,
  buildStreamingRuntimePrompt,
  buildWordTeachingPrecomputePrompt,
  SPELLING_COACH_SYSTEM_PROMPT,
} from "./prompt.js";
import { runSpellingCoachAgent } from "./runAgent.js";
import { interpretVoiceUtterance, normalizeSpokenSpelling } from "./voice.js";
import {
  buildReferenceHintsText,
  buildSpellingRuleHintsText,
  getReferenceHints,
} from "./referenceData.js";
import type { DeepAgentLike } from "./agent.js";
import type { DirectModelLike } from "./directModel.js";
import {
  parseRelatedFormsOnlyPrecompute,
  parseWordTeachingOnlyPrecompute,
  type SpellingCoachInput,
  type SpellingCoachOutput,
} from "./schemas.js";
import {
  getCustomWordListById,
  getForeignOriginWordListByOrigin,
  getStoredWordBreakdown,
  getStoredWordTeachingOnlyPrecompute,
  getWordByText,
  listCustomWordListsForUser,
  loadCustomWordLists,
  loadForeignOriginWordLists,
  searchWords,
  saveForeignOriginWordLists,
  pickNextWord,
  saveCustomWordLists,
} from "./wordCatalog.js";

type OutputOverrides = Omit<
  Partial<SpellingCoachOutput>,
  "missAnalysis" | "wordTeaching" | "errorRelevance"
> & {
  missAnalysis?: Partial<SpellingCoachOutput["missAnalysis"]>;
  wordTeaching?: {
    conceptTeaching?: Partial<
      SpellingCoachOutput["wordTeaching"]["conceptTeaching"]
    >;
  };
  errorRelevance?: Partial<SpellingCoachOutput["errorRelevance"]>;
};

function makeOutput(overrides: OutputOverrides): SpellingCoachOutput {
  return {
    correctness: {
      isCorrect: false,
      reinforceSuccess: false,
      ...overrides.correctness,
    },
    missAnalysis: {
      summary: "",
      primaryErrorType: null,
      secondaryErrorTypes: [],
      errorTypeEvidence: {},
      primaryErrorFocus: "",
      likelyWrongWordInterpretation: false,
      usedMeaningDisambiguationWell: false,
      ...overrides.missAnalysis,
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
        ...overrides.wordTeaching?.conceptTeaching,
      },
    },
    errorRelevance: {
      mostRelevantToError: "unclear",
      confidence: 0,
      reason: "",
      ...overrides.errorRelevance,
    },
    teachingDecision: {
      strategy: "pattern",
      primaryFocus: "",
      secondaryFocuses: [],
      confidence: 0,
      rationale: "",
      ...overrides.teachingDecision,
    },
    coachingText: {
      shortFeedback: "",
      fullExplanation: "",
      memoryTip: "",
      sayAloudTip: "",
      ...overrides.coachingText,
    },
    wordBreakdown: {
      displayChunks: [],
      alternateDisplayChunks: [],
      chunkReason: "",
      matchedPatterns: [],
      ...overrides.wordBreakdown,
    },
    conceptLabels: {
      originLabels: [],
      patternLabels: [],
      morphologyLabels: [],
      ...overrides.conceptLabels,
    },
    nextStep: {
      practiceFocus: "",
      shouldReviewSoon: false,
      suggestedSimilarWordTypes: [],
      ...overrides.nextStep,
    },
  };
}

function isPatternFilterPrompt(content: unknown): boolean {
  return (
    typeof content === "string" &&
    content.includes("Review the candidate spelling patterns detected by code")
  );
}

function isLevelOnePrompt(content: unknown): boolean {
  return (
    typeof content === "string" &&
    content.includes("Analyze this Level 1 spelling attempt for a child around ages 6 to 8.")
  );
}

function isLevelOnePrecomputePrompt(content: unknown): boolean {
  return (
    typeof content === "string" &&
    content.includes("This precompute is only for teaching-friendly chunking.")
  );
}

function createMockAgent(output: SpellingCoachOutput): DeepAgentLike {
  return {
    async invoke(input?: { messages?: Array<{ content?: unknown }> }) {
      const lastContent = input?.messages?.at(-1)?.content;
      if (isPatternFilterPrompt(lastContent)) {
        return {
          messages: [
            {
              role: "assistant",
              content: JSON.stringify({ keptDescriptions: [] }),
            },
          ],
        };
      }

      if (isLevelOnePrecomputePrompt(lastContent)) {
        return {
          messages: [
            {
              role: "assistant",
              content: JSON.stringify({
                wordTeaching: {
                  conceptTeaching: {
                    summary: "",
                    meaningFocus: "",
                    originFocus: "",
                    morphologyFocus: "",
                    originLabels: [],
                    morphologyLabels: [],
                  },
                },
                wordBreakdown: {
                  displayChunks: ["sun", "set"],
                  chunkReason: "",
                },
                conceptLabels: {
                  originLabels: [],
                  patternLabels: [],
                  morphologyLabels: [],
                },
              }),
            },
          ],
        };
      }

      if (isLevelOnePrompt(lastContent)) {
        return {
          messages: [
            {
              role: "assistant",
              content: JSON.stringify({
                shortFeedback: "Nice try.",
                sayAloudTip: "Say sun-set.",
              }),
            },
          ],
        };
      }

      return {
        messages: [
          {
            role: "assistant",
            content: JSON.stringify(output),
          },
        ],
      };
    },
  };
}

function createSequenceMockAgent(outputs: string[]): DeepAgentLike {
  let index = 0;

  return {
    async invoke(input?: { messages?: Array<{ content?: unknown }> }) {
      const lastContent = input?.messages?.at(-1)?.content;
      if (isPatternFilterPrompt(lastContent)) {
        return {
          messages: [
            {
              role: "assistant",
              content: JSON.stringify({ keptDescriptions: [] }),
            },
          ],
        };
      }

      if (isLevelOnePrecomputePrompt(lastContent)) {
        const content = outputs[Math.min(index, outputs.length - 1)];
        index += 1;
        return {
          messages: [
            {
              role: "assistant",
              content,
            },
          ],
        };
      }

      if (isLevelOnePrompt(lastContent)) {
        const content = outputs[Math.min(index, outputs.length - 1)];
        index += 1;
        return {
          messages: [
            {
              role: "assistant",
              content,
            },
          ],
        };
      }

      const content = outputs[Math.min(index, outputs.length - 1)];
      index += 1;
      return {
        messages: [
          {
            role: "assistant",
            content,
          },
        ],
      };
    },
  };
}

function createSequenceMockModel(outputs: string[]): DirectModelLike {
  let index = 0;

  return {
    async invoke(messages?: Array<{ content?: unknown }>) {
      const lastContent = messages?.at(-1)?.content;
      if (isPatternFilterPrompt(lastContent)) {
        return JSON.stringify({ keptDescriptions: [] });
      }

      if (isLevelOnePrecomputePrompt(lastContent)) {
        const content = outputs[Math.min(index, outputs.length - 1)];
        index += 1;
        return content;
      }

      if (isLevelOnePrompt(lastContent)) {
        const content = outputs[Math.min(index, outputs.length - 1)];
        index += 1;
        return content;
      }

      const content = outputs[Math.min(index, outputs.length - 1)];
      index += 1;
      return content;
    },
  };
}

function createDirectMockModel(output: unknown): DirectModelLike {
  return {
    async invoke(messages?: Array<{ content?: unknown }>) {
      const lastContent = messages?.at(-1)?.content;
      if (isPatternFilterPrompt(lastContent)) {
        return JSON.stringify({ keptDescriptions: [] });
      }

      if (isLevelOnePrecomputePrompt(lastContent)) {
        return JSON.stringify({
          wordTeaching: {
            conceptTeaching: {
              summary: "",
              meaningFocus: "",
              originFocus: "",
              morphologyFocus: "",
              originLabels: [],
              morphologyLabels: [],
            },
          },
          wordBreakdown: {
            displayChunks: ["a", "bout"],
            chunkReason: "",
          },
          conceptLabels: {
            originLabels: [],
            patternLabels: [],
            morphologyLabels: [],
          },
        });
      }

      if (isLevelOnePrompt(lastContent)) {
        return JSON.stringify(output);
      }

      return JSON.stringify(output);
    },
  };
}

const baseProfile = {
  childId: "c1",
  age: 11,
  grade: "6",
  spellingLevel: "competition",
} as const;

function flushMicrotasks(): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, 0);
  });
}

test("handles adscititious missing-letter near miss", async () => {
  const input: SpellingCoachInput = {
    targetWord: "adscititious",
    childAttempt: "adcititious",
    childProfile: baseProfile,
    wordMetadata: {
      definition: "added from outside, not originally part of something",
      origin: "Latin",
      partOfSpeech: "adjective",
      exampleSentence: "The story had adscititious details added later.",
      pronunciation: "ad-si-TISH-us",
    },
    missSignals: {
      isCorrect: false,
      nearMiss: true,
      missingLetters: ["s"],
      extraLetters: [],
      substitutedLetters: [],
      transposedLetters: [],
      repeatedLetterIssue: false,
      likelyRushed: true,
      editDistance: 1,
    },
    structuralHints: {
      syllables: ["ad", "sci", "ti", "tious"],
      likelyChunks: ["ad", "scit", "itious"],
      detectedPatterns: ["sc-cluster", "-itious"],
      likelyPrefix: "ad",
      likelySuffix: "itious",
    },
    sessionContext: {
      mode: "practice",
      previousAttemptsOnThisWord: 0,
      previousMissPatterns: ["missing-letter deletion"],
      recentlyPracticedWords: ["fictitious", "ambitious"],
    },
  };

  const expected = makeOutput({
    missAnalysis: {
      summary: "Very close. The attempt drops the s in the sc cluster near the start of the word.",
      primaryErrorType: "missing_letter",
      secondaryErrorTypes: ["consonant_cluster_error"],
      errorTypeEvidence: {
        missing_letter: "A required letter is missing from the attempt.",
        consonant_cluster_error: "The early consonant cluster is incomplete in the attempt.",
      },
      primaryErrorFocus: "Remember the sc cluster in ad + scititious.",
    },
    errorRelevance: {
      mostRelevantToError: "form",
      confidence: 0.9,
      reason: "The miss is a one-letter deletion in the sc cluster, so the form-based spelling pattern is the direct fix.",
    },
    teachingDecision: {
      strategy: "pattern",
      primaryFocus: "Hold onto the sc cluster before the -itious ending.",
      secondaryFocuses: ["Notice the familiar -itious ending", "Slow down enough to check the early cluster"],
      confidence: 0.9,
      rationale: "The miss is a one-letter deletion, and the most reusable teaching point is keeping the sc cluster intact.",
    },
    coachingText: {
      shortFeedback: "That was very close. You only missed the s.",
      fullExplanation: "The tricky part is the sc in adscititious. Keep that cluster together first, then finish with the familiar -itious ending.",
      memoryTip: "Think: ad + scit + itious, and do not let the s disappear.",
      sayAloudTip: "Say the start slowly: ad-sci-ti-tious.",
    },
    wordBreakdown: {
      displayChunks: ["ad", "scit", "itious"],
      chunkReason: "The split highlights the missing sc cluster and the stable -itious ending.",
    },
    conceptLabels: {
      originLabels: [],
      patternLabels: ["sc-cluster", "-itious"],
      morphologyLabels: [],
    },
    nextStep: {
      practiceFocus: "Practice words with an early consonant cluster plus -itious.",
      shouldReviewSoon: true,
      suggestedSimilarWordTypes: ["-itious words", "cluster-heavy academic words"],
    },
  });

  const result = await runSpellingCoachAgent(input, {
    agent: createMockAgent(expected),
  });

  assert.equal(result.teachingDecision.strategy, "pattern");
  assert.deepEqual(result.wordBreakdown.displayChunks, ["ad", "scit", "itious"]);
  assert.equal(result.correctness.isCorrect, false);
});

test("handles arachnophagous with heavy phonetic simplification", async () => {
  const input: SpellingCoachInput = {
    targetWord: "arachnophagous",
    childAttempt: "araknophagus",
    childProfile: baseProfile,
    wordMetadata: {
      definition: "feeding on spiders",
      origin: "Greek",
      partOfSpeech: "adjective",
      pronunciation: "uh-RAK-nof-uh-gus",
    },
    missSignals: {
      isCorrect: false,
      nearMiss: false,
      missingLetters: ["c", "h"],
      extraLetters: ["k"],
      substitutedLetters: ["k for ch", "u for ou"],
      transposedLetters: [],
      repeatedLetterIssue: false,
      likelyRushed: false,
      editDistance: 4,
    },
    structuralHints: {
      syllables: ["a", "rach", "no", "pha", "gous"],
      likelyChunks: ["arachno", "phagous"],
      detectedPatterns: ["ch says k", "ph says f", "-gous"],
      likelySuffix: "gous",
    },
    sessionContext: {
      mode: "practice",
      previousAttemptsOnThisWord: 1,
      previousMissPatterns: ["phonetic spelling"],
      recentlyPracticedWords: ["arachnid", "sarcophagus"],
    },
  };

  const expected = makeOutput({
    missAnalysis: {
      summary: "The attempt is phonetic and swaps several learned patterns for simpler sounds.",
      primaryErrorType: "phonetic_spelling",
      secondaryErrorTypes: ["pattern_rule_mismatch"],
      errorTypeEvidence: {
        phonetic_spelling: "The child wrote a sound-based approximation instead of the conventional spelling.",
        pattern_rule_mismatch: "Learned spelling patterns were simplified in the attempt.",
      },
      primaryErrorFocus: "Use the stored patterns ch and ph instead of writing only the sounds you hear.",
    },
    wordTeaching: {
      conceptTeaching: {
        summary: "The concept side links arachno to spider and phagous to eating.",
        meaningFocus: "feeding on spiders",
        originFocus: "Greek-derived",
        morphologyFocus: "arachno + phagous",
        originLabels: ["greek-derived"],
        morphologyLabels: ["arachno", "phagous"],
      },
    },
    errorRelevance: {
      mostRelevantToError: "mixed",
      confidence: 0.93,
      reason: "The child simplified multiple spelling patterns, so chunking and pattern coaching both matter most to the miss.",
    },
    teachingDecision: {
      strategy: "mixed",
      primaryFocus: "Chunk the word into arachno + phagous while locking in ch and ph.",
      secondaryFocuses: ["ph says f", "Keep the ou in -gous"],
      confidence: 0.93,
      rationale: "This word needs both chunking and pattern coaching because multiple sound-based simplifications happened across the word.",
    },
    coachingText: {
      shortFeedback: "You heard the sounds well, but this word keeps some book-spelling patterns.",
      fullExplanation: "Try it as arachno + phagous. In the first chunk, ch stays ch even though it sounds like k. In the second chunk, ph spells the f sound, and -gous keeps ou.",
      memoryTip: "A spider word starts like arachnid and then adds phagous.",
      sayAloudTip: "Say: a-rach-no-phag-ous, and listen for the chunk change.",
    },
    wordBreakdown: {
      displayChunks: ["arachno", "phagous"],
      chunkReason: "The chunks match the strongest reusable teaching pieces and reduce overload.",
    },
    conceptLabels: {
      originLabels: ["greek-derived"],
      patternLabels: ["ch says k", "ph says f", "-gous"],
      morphologyLabels: ["arachno", "phagous"],
    },
    nextStep: {
      practiceFocus: "Review long words that keep classical spelling patterns instead of pure sound spelling.",
      shouldReviewSoon: true,
      suggestedSimilarWordTypes: ["ph words", "Greek-pattern science words"],
    },
  });

  const result = await runSpellingCoachAgent(input, {
    agent: createMockAgent(expected),
  });

  assert.equal(result.teachingDecision.strategy, "mixed");
  assert.equal(result.coachingText.memoryTip.includes("arachnid"), true);
  assert.deepEqual(result.nextStep, {
    practiceFocus: "",
    shouldReviewSoon: false,
    suggestedSimilarWordTypes: [],
  });
});

test("reinforces a correct spelling without over-teaching", async () => {
  const input: SpellingCoachInput = {
    targetWord: "pulpit",
    childAttempt: "pulpit",
    childProfile: {
      childId: "c2",
      age: 9,
      grade: "4",
      spellingLevel: "on-grade",
    },
    wordMetadata: {
      definition: "a raised platform in a church",
      partOfSpeech: "noun",
      pronunciation: "PUHL-pit",
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
      syllables: ["pul", "pit"],
      likelyChunks: ["pul", "pit"],
      detectedPatterns: ["closed syllables"],
      likelyPrefix: undefined,
      likelySuffix: undefined,
    },
    sessionContext: {
      mode: "practice",
      previousAttemptsOnThisWord: 0,
      previousMissPatterns: [],
      recentlyPracticedWords: ["puppet", "pulp"],
    },
  };

  const expected = makeOutput({
    correctness: {
      isCorrect: true,
      reinforceSuccess: true,
    },
    missAnalysis: {
      summary: "The word was spelled correctly.",
      primaryErrorType: null,
      secondaryErrorTypes: [],
      errorTypeEvidence: {},
      primaryErrorFocus: "Accurate spelling",
      usedMeaningDisambiguationWell: true,
    },
    errorRelevance: {
      mostRelevantToError: "unclear",
      confidence: 0.4,
      reason: "There is no miss to diagnose, so no single teaching layer is clearly tied to an error.",
    },
    teachingDecision: {
      strategy: "pattern",
      primaryFocus: "Notice the two short-vowel chunks pul + pit.",
      secondaryFocuses: [],
      confidence: 0.84,
      rationale: "The child got it right, so a light reusable pattern reminder is enough.",
    },
    coachingText: {
      shortFeedback: "Correct. You spelled pulpit exactly right.",
      fullExplanation: "Nice job. One quick reminder: pulpit has two short chunks, pul + pit.",
      memoryTip: "",
      sayAloudTip: "Say pul-pit.",
    },
    wordBreakdown: {
      displayChunks: ["pul", "pit"],
      chunkReason: "The chunks reinforce the correct short-vowel structure without adding extra complexity.",
    },
    conceptLabels: {
      originLabels: [],
      patternLabels: ["closed syllables"],
      morphologyLabels: [],
    },
    nextStep: {
      practiceFocus: "Keep checking both short-vowel chunks in similar two-part words.",
      shouldReviewSoon: false,
      suggestedSimilarWordTypes: ["two-syllable short-vowel words"],
    },
  });

  const result = await runSpellingCoachAgent(input, {
    agent: createMockAgent(expected),
  });

  assert.equal(result.correctness.isCorrect, true);
  assert.equal(result.correctness.reinforceSuccess, true);
  assert.equal(result.missAnalysis.primaryErrorType, null);
  assert.deepEqual(result.missAnalysis.secondaryErrorTypes, []);
  assert.equal(result.coachingText.fullExplanation, "");
});

test("uses minimal Level 1 coaching output and clears advanced sections", async () => {
  const input: SpellingCoachInput = {
    targetWord: "about",
    childAttempt: "abot",
    childProfile: {
      childId: "c-level1",
      age: 7,
      grade: "2",
      spellingLevel: "developing",
    },
    wordMetadata: {
      definition: "Near or around a place or time.",
      origin: "Old English",
      partOfSpeech: "preposition",
      pronunciation: "uh-BOUT",
    },
    missSignals: {
      isCorrect: false,
      nearMiss: true,
      missingLetters: ["u"],
      extraLetters: [],
      substitutedLetters: [],
      transposedLetters: [],
      repeatedLetterIssue: false,
      likelyRushed: false,
      editDistance: 1,
    },
    structuralHints: {
      syllables: ["a", "bout"],
      likelyChunks: ["a", "bout"],
      detectedPatterns: [],
    },
    sessionContext: {
      mode: "practice",
      previousAttemptsOnThisWord: 0,
      previousMissPatterns: [],
      recentlyPracticedWords: [],
    },
  };

  const result = await runSpellingCoachAgent(input, {
    directModel: createSequenceMockModel([
      JSON.stringify({
        wordTeaching: {
          conceptTeaching: {
            summary: "",
            meaningFocus: "",
            originFocus: "",
            morphologyFocus: "",
            originLabels: [],
            morphologyLabels: [],
          },
        },
        wordBreakdown: {
          displayChunks: ["ab", "out"],
          chunkReason: "",
        },
        conceptLabels: {
          originLabels: [],
          patternLabels: [],
          morphologyLabels: [],
        },
      }),
      JSON.stringify({
        shortFeedback: "Nice try.",
        sayAloudTip: "Say a-bout.",
      }),
    ]),
    runtime: "direct",
  });

  assert.equal(result.correctness.isCorrect, false);
  assert.equal(result.coachingText.shortFeedback, "Nice try.");
  assert.equal(result.coachingText.sayAloudTip, "Say it slowly: uh-BOWT");
  assert.equal(result.coachingText.fullExplanation, "");
  const storedAboutBreakdown = getStoredWordBreakdown("about");
  assert.deepEqual(
    result.wordBreakdown.displayChunks,
    storedAboutBreakdown?.displayChunks ?? ["ab", "out"],
  );
  assert.equal(
    result.wordBreakdown.chunkReason,
    storedAboutBreakdown?.chunkReason ?? "The word breaks as ab + out.",
  );
  assert.deepEqual(
    result.wordBreakdown.matchedPatterns,
    storedAboutBreakdown?.matchedPatterns ?? [
      { label: "vowel pattern ou" },
      {
        label: "two syllables",
        matchedParts: ["a", "bout"],
        alternateMatchedParts: [["ab", "out"]],
      },
    ],
  );
  assert.equal(result.wordTeaching.conceptTeaching.summary, "");
  assert.deepEqual(result.conceptLabels.patternLabels, []);
  assert.deepEqual(result.nextStep.suggestedSimilarWordTypes, []);
});

test("handles fictitious with missing middle chunk", async () => {
  const input: SpellingCoachInput = {
    targetWord: "fictitious",
    childAttempt: "fictous",
    childProfile: baseProfile,
    wordMetadata: {
      definition: "made up; not real",
      origin: "Latin",
      partOfSpeech: "adjective",
      exampleSentence: "The story used a fictitious town.",
      pronunciation: "fik-TISH-us",
    },
    missSignals: {
      isCorrect: false,
      nearMiss: false,
      missingLetters: ["i", "t", "i"],
      extraLetters: [],
      substitutedLetters: [],
      transposedLetters: [],
      repeatedLetterIssue: false,
      likelyRushed: false,
      editDistance: 3,
    },
    structuralHints: {
      syllables: ["fic", "ti", "tious"],
      likelyChunks: ["fic", "ti", "tious"],
      detectedPatterns: ["-itious"],
      likelySuffix: "itious",
    },
    sessionContext: {
      mode: "practice",
      previousAttemptsOnThisWord: 2,
      previousMissPatterns: ["dropped middle chunk"],
      recentlyPracticedWords: ["ambitious", "nutritious"],
    },
  };

  const expected = makeOutput({
    missAnalysis: {
      summary: "The ending was started correctly, but the middle ti chunk disappeared.",
      primaryErrorType: "chunk_mismatch",
      secondaryErrorTypes: ["ending_confusion"],
      errorTypeEvidence: {
        chunk_mismatch: "A larger chunk from the word is missing in the attempt.",
        ending_confusion: "The ending was compressed rather than preserved accurately.",
      },
      primaryErrorFocus: "Keep the full ti + tious ending instead of shrinking it to tous.",
    },
    wordTeaching: {
      conceptTeaching: {
        summary: "This word belongs to the made-up or not-real word family.",
        meaningFocus: "made up; not real",
      },
    },
    errorRelevance: {
      mostRelevantToError: "form",
      confidence: 0.91,
      reason: "The child omitted the middle chunk, so chunking is the clearest correction path.",
    },
    teachingDecision: {
      strategy: "chunking",
      primaryFocus: "Spell fictitious as fic + ti + tious.",
      secondaryFocuses: ["Match it to other -itious words"],
      confidence: 0.91,
      rationale: "The cleanest fix is to restore the missing middle chunk and rehearse the ending in three parts.",
    },
    coachingText: {
      shortFeedback: "You had the start, but the middle chunk dropped out.",
      fullExplanation: "Write fictitious in three pieces: fic + ti + tious. That middle ti matters before the tious ending.",
      memoryTip: "Think: fic, then a small ti bridge, then tious.",
      sayAloudTip: "Say it in beats: fic - ti - tious.",
    },
    wordBreakdown: {
      displayChunks: ["fic", "ti", "tious"],
      chunkReason: "The chunks expose the exact section that was omitted.",
    },
    conceptLabels: {
      originLabels: [],
      patternLabels: ["-itious"],
      morphologyLabels: [],
    },
    nextStep: {
      practiceFocus: "Practice words that end in -itious without skipping the middle ti.",
      shouldReviewSoon: true,
      suggestedSimilarWordTypes: ["-itious words", "multi-part adjective endings"],
    },
  });

  const result = await runSpellingCoachAgent(input, {
    agent: createMockAgent(expected),
  });

  assert.equal(result.teachingDecision.strategy, "chunking");
  assert.deepEqual(result.wordBreakdown.displayChunks, ["fic", "ti", "tious"]);
  assert.equal(result.missAnalysis.primaryErrorFocus.includes("ti + tious"), true);
});

test("retries when the model returns the wrong JSON shape first", async () => {
  const input: SpellingCoachInput = {
    targetWord: "pulpit",
    childAttempt: "pulpit",
    childProfile: {
      childId: "c2",
      age: 9,
      grade: "4",
      spellingLevel: "on-grade",
    },
    wordMetadata: {
      definition: "a raised platform in a church",
      partOfSpeech: "noun",
      pronunciation: "PUHL-pit",
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
      syllables: ["pul", "pit"],
      likelyChunks: ["pul", "pit"],
      detectedPatterns: ["closed syllables"],
    },
    sessionContext: {
      mode: "practice",
      previousAttemptsOnThisWord: 0,
      previousMissPatterns: [],
      recentlyPracticedWords: ["puppet", "pulp"],
    },
  };

  const corrected = makeOutput({
    correctness: {
      isCorrect: true,
      reinforceSuccess: true,
    },
    missAnalysis: {
      summary: "The word was spelled correctly.",
      primaryErrorType: null,
      secondaryErrorTypes: [],
      errorTypeEvidence: {},
      primaryErrorFocus: "Accurate spelling",
      usedMeaningDisambiguationWell: true,
    },
    errorRelevance: {
      mostRelevantToError: "unclear",
      confidence: 0.4,
      reason: "There is no miss to diagnose, so no single teaching layer is clearly tied to an error.",
    },
    teachingDecision: {
      strategy: "pattern",
      primaryFocus: "Notice the two short-vowel chunks pul + pit.",
      secondaryFocuses: [],
      confidence: 0.84,
      rationale: "The child got it right, so a light reusable pattern reminder is enough.",
    },
    coachingText: {
      shortFeedback: "Correct. You spelled pulpit exactly right.",
      fullExplanation: "Nice job. One quick reminder: pulpit has two short chunks, pul + pit.",
      memoryTip: "",
      sayAloudTip: "Say pul-pit.",
    },
    wordBreakdown: {
      displayChunks: ["pul", "pit"],
      chunkReason: "The chunks reinforce the correct short-vowel structure without adding extra complexity.",
    },
    conceptLabels: {
      originLabels: [],
      patternLabels: ["closed syllables"],
      morphologyLabels: [],
    },
    nextStep: {
      practiceFocus: "Keep checking both short-vowel chunks in similar two-part words.",
      shouldReviewSoon: false,
      suggestedSimilarWordTypes: ["two-syllable short-vowel words"],
    },
  });

  const result = await runSpellingCoachAgent(input, {
    agent: createSequenceMockAgent([
      JSON.stringify({
        diagnosis: "correct",
        teachingStrategy: "pattern",
        explanation: "Nice job",
        confidence: 0.9,
      }),
      JSON.stringify(corrected),
    ]),
  });

  assert.equal(result.correctness.isCorrect, true);
  assert.equal(result.coachingText.shortFeedback, "Correct. You spelled pulpit exactly right.");
});

test("finds local Greek root hints for arachnophagous", () => {
  const input: SpellingCoachInput = {
    targetWord: "arachnophagous",
    childAttempt: "araconofagus",
    childProfile: baseProfile,
    wordMetadata: {
      definition: "feeding on spiders",
      origin: "Greek",
      partOfSpeech: "adjective",
      pronunciation: "uh-RAK-nof-uh-gus",
    },
    missSignals: {
      isCorrect: false,
      nearMiss: false,
      missingLetters: ["h", "p", "h"],
      extraLetters: [],
      substitutedLetters: ["c for ch", "f for ph"],
      transposedLetters: [],
      repeatedLetterIssue: false,
      likelyRushed: false,
      editDistance: 5,
    },
    structuralHints: {
      syllables: ["a", "rach", "no", "pha", "gous"],
      likelyChunks: ["arachno", "phagous"],
      detectedPatterns: ["ch says k", "ph says f", "-gous"],
      likelySuffix: "gous",
    },
    sessionContext: {
      mode: "practice",
      previousAttemptsOnThisWord: 0,
      previousMissPatterns: ["phonetic spelling"],
      recentlyPracticedWords: ["arachnid", "phosphorus"],
    },
  };

  const hints = getReferenceHints(input);
  const hintText = buildReferenceHintsText(input);

  assert.equal(hints.some((hint) => hint.root.includes("arachn-")), true);
  assert.equal(hints.some((hint) => hint.root.includes("phag-")), true);
  assert.equal(hintText.includes("spider"), true);
  assert.equal(hintText.includes("eat"), true);
});

test("finds local prefix hints from prefixes.csv", () => {
  const input: SpellingCoachInput = {
    targetWord: "preview",
    childAttempt: "preveiw",
    childProfile: {
      childId: "c3",
      age: 10,
      grade: "5",
      spellingLevel: "on-grade",
    },
    wordMetadata: {
      definition: "a view in advance",
      partOfSpeech: "noun",
      pronunciation: "PREE-vyoo",
    },
    missSignals: {
      isCorrect: false,
      nearMiss: true,
      missingLetters: [],
      extraLetters: [],
      substitutedLetters: [],
      transposedLetters: ["ie"],
      repeatedLetterIssue: false,
      likelyRushed: false,
      editDistance: 2,
    },
    structuralHints: {
      syllables: ["pre", "view"],
      likelyChunks: ["pre", "view"],
      detectedPatterns: ["prefix pre-"],
      likelyPrefix: "pre",
    },
    sessionContext: {
      mode: "practice",
      previousAttemptsOnThisWord: 0,
      previousMissPatterns: [],
      recentlyPracticedWords: ["predict", "review"],
    },
  };

  const hints = getReferenceHints(input);
  const prefixHint = hints.find((hint) => hint.root.includes("pre-"));

  assert.equal(prefixHint?.role, "prefix");
  assert.equal(prefixHint?.source, "prefixes_csv");
  assert.equal(prefixHint?.meaning, "before");
});

test("finds local suffix hints from suffixes.csv", () => {
  const input: SpellingCoachInput = {
    targetWord: "gracious",
    childAttempt: "gracous",
    childProfile: {
      childId: "c4",
      age: 10,
      grade: "5",
      spellingLevel: "on-grade",
    },
    wordMetadata: {
      definition: "kind and pleasant",
      partOfSpeech: "adjective",
      pronunciation: "GRAY-shus",
    },
    missSignals: {
      isCorrect: false,
      nearMiss: true,
      missingLetters: ["i"],
      extraLetters: [],
      substitutedLetters: [],
      transposedLetters: [],
      repeatedLetterIssue: false,
      likelyRushed: false,
      editDistance: 1,
    },
    structuralHints: {
      syllables: ["gra", "cious"],
      likelyChunks: ["gra", "cious"],
      detectedPatterns: ["-ious"],
      likelySuffix: "ious",
    },
    sessionContext: {
      mode: "practice",
      previousAttemptsOnThisWord: 0,
      previousMissPatterns: [],
      recentlyPracticedWords: ["curious", "delicious"],
    },
  };

  const hints = getReferenceHints(input);
  const suffixHint = hints.find((hint) => hint.root.includes("-ious"));

  assert.equal(suffixHint?.role, "suffix_family");
  assert.equal(suffixHint?.source, "suffixes_csv");
  assert.equal(suffixHint?.meaning, "having qualities of");
});

test("prompt frames csv data as sample affix families, not a closed list", () => {
  const input: SpellingCoachInput = {
    targetWord: "arachnophagous",
    childAttempt: "araconofagus",
    childProfile: baseProfile,
    wordMetadata: {
      definition: "feeding on spiders",
      origin: "Greek",
      partOfSpeech: "adjective",
      pronunciation: "uh-RAK-nof-uh-gus",
    },
    missSignals: {
      isCorrect: false,
      nearMiss: false,
      missingLetters: ["h", "p", "h"],
      extraLetters: [],
      substitutedLetters: ["c for ch", "f for ph"],
      transposedLetters: [],
      repeatedLetterIssue: false,
      likelyRushed: false,
      editDistance: 5,
    },
    structuralHints: {
      syllables: ["a", "rach", "no", "pha", "gous"],
      likelyChunks: ["arachno", "phagous"],
      detectedPatterns: ["ch says k", "ph says f", "-gous"],
      likelySuffix: "gous",
    },
    sessionContext: {
      mode: "practice",
      previousAttemptsOnThisWord: 0,
      previousMissPatterns: ["phonetic spelling"],
      recentlyPracticedWords: ["arachnid", "phosphorus"],
    },
  };

  const prompt = buildSpellingCoachPrompt(input);

  assert.equal(prompt.includes("sample affix and morpheme families"), true);
  assert.equal(prompt.includes("not as a closed dictionary"), true);
  assert.equal(prompt.includes("Local reference hints from curated Greek/Latin morpheme CSVs:"), true);
  assert.equal(
    SPELLING_COACH_SYSTEM_PROMPT.includes('"relatedForms": string[]'),
    true,
  );
  assert.equal(
    SPELLING_COACH_SYSTEM_PROMPT.includes(
      "first look for a helpful similar-word, word-family, or comparison cue that genuinely supports the spelling.",
    ),
    true,
  );
  assert.equal(
    SPELLING_COACH_SYSTEM_PROMPT.includes(
      "If a useful similar-word comparison is available, prefer it over repeating conceptTeaching.",
    ),
    true,
  );
  assert.equal(
    SPELLING_COACH_SYSTEM_PROMPT.includes(
      "After similar-word comparisons, use pattern, structure, chunking, or letter-choice cues as the next best explanation support.",
    ),
    true,
  );
  assert.equal(SPELLING_COACH_SYSTEM_PROMPT.includes('"mostRelevantToError": "form" | "concept" | "mixed" | "unclear"'), true);
  assert.equal(SPELLING_COACH_SYSTEM_PROMPT.includes('confidence is below 0.75'), true);
  assert.equal(SPELLING_COACH_SYSTEM_PROMPT.includes("Curated spelling-rule hints may be provided"), false);
});

test("defaults missing related forms to an empty array in concept precompute parsing", () => {
  const parsed = parseWordTeachingOnlyPrecompute({
    wordTeaching: {
      conceptTeaching: {
        summary: "A test summary.",
        meaningFocus: "A test meaning.",
        originFocus: "A test origin.",
        morphologyFocus: "A test morphology.",
        originLabels: [],
        morphologyLabels: [],
      },
    },
    conceptLabels: {
      originLabels: [],
      patternLabels: [],
      morphologyLabels: [],
    },
  });

  assert.deepEqual(parsed.wordTeaching.conceptTeaching.relatedForms, []);
});

test("builds a dedicated related-forms-only precompute prompt", () => {
  const prompt = buildRelatedFormsOnlyPrecomputePrompt({
    targetWord: "hypocritical",
    childAttempt: "hypocritical",
    childProfile: {
      childId: "c-related-forms",
      age: 11,
      grade: "6",
      spellingLevel: "advanced",
    },
    wordMetadata: {
      definition: "behaving in a way that says one thing but does another",
      partOfSpeech: "adjective",
      pronunciation: "hip-uh-KRIT-uh-kul",
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
      syllables: ["hy", "po", "crit", "i", "cal"],
      likelyChunks: ["hypo", "critic", "al"],
      detectedPatterns: [],
    },
    sessionContext: {
      mode: "practice",
      previousAttemptsOnThisWord: 0,
      previousMissPatterns: [],
      recentlyPracticedWords: [],
    },
  });

  assert.equal(
    prompt.includes("This is offline precompute for related forms only."),
    true,
  );
  assert.equal(
    prompt.includes("Do not return the target word itself in relatedForms."),
    true,
  );
  assert.equal(prompt.includes('"relatedForms": string[]'), true);
});

test("defaults missing related forms to an empty array in related-forms-only parsing", () => {
  const parsed = parseRelatedFormsOnlyPrecompute({});

  assert.deepEqual(parsed.relatedForms, []);
});

test("finds local numeric prefix hints from numeric_prefixes.csv", () => {
  const input: SpellingCoachInput = {
    targetWord: "triangle",
    childAttempt: "triangel",
    childProfile: {
      childId: "c5",
      age: 9,
      grade: "4",
      spellingLevel: "on-grade",
    },
    wordMetadata: {
      definition: "a three-sided shape",
      partOfSpeech: "noun",
      pronunciation: "TRY-ang-gul",
    },
    missSignals: {
      isCorrect: false,
      nearMiss: true,
      missingLetters: [],
      extraLetters: [],
      substitutedLetters: [],
      transposedLetters: ["le"],
      repeatedLetterIssue: false,
      likelyRushed: false,
      editDistance: 2,
    },
    structuralHints: {
      syllables: ["tri", "an", "gle"],
      likelyChunks: ["tri", "angle"],
      detectedPatterns: ["prefix tri-"],
      likelyPrefix: "tri",
    },
    sessionContext: {
      mode: "practice",
      previousAttemptsOnThisWord: 0,
      previousMissPatterns: [],
      recentlyPracticedWords: ["tricycle", "trio"],
    },
  };

  const hints = getReferenceHints(input);
  const numericHint = hints.find(
    (hint) =>
      hint.root.includes("tri-") && hint.source === "numeric_prefixes_csv",
  );

  assert.equal(numericHint?.role, "prefix");
  assert.equal(numericHint?.source, "numeric_prefixes_csv");
  assert.equal(numericHint?.meaning, "3");
});

test("finds supplemental prefix hints from PrefixList.txt", () => {
  const input: SpellingCoachInput = {
    targetWord: "antebellum",
    childAttempt: "antabellum",
    childProfile: {
      childId: "c6",
      age: 12,
      grade: "7",
      spellingLevel: "advanced",
    },
    wordMetadata: {
      definition: "existing before a war",
      partOfSpeech: "adjective",
      pronunciation: "an-tee-BEL-um",
    },
    missSignals: {
      isCorrect: false,
      nearMiss: true,
      missingLetters: ["e"],
      extraLetters: [],
      substitutedLetters: ["a for e"],
      transposedLetters: [],
      repeatedLetterIssue: false,
      likelyRushed: false,
      editDistance: 1,
    },
    structuralHints: {
      syllables: ["an", "te", "bel", "lum"],
      likelyChunks: ["ante", "bellum"],
      detectedPatterns: ["prefix ante-"],
      likelyPrefix: "ante",
    },
    sessionContext: {
      mode: "practice",
      previousAttemptsOnThisWord: 0,
      previousMissPatterns: [],
      recentlyPracticedWords: ["anterior", "preview"],
    },
  };

  const hints = getReferenceHints(input);
  const txtHint = hints.find(
    (hint) => hint.root.includes("ante-") && hint.source === "prefix_list_csv",
  );

  assert.equal(txtHint?.role, "prefix");
  assert.equal(txtHint?.source, "prefix_list_csv");
});

test("finds supplemental suffix hints from SuffixList.txt", () => {
  const input: SpellingCoachInput = {
    targetWord: "hesitate",
    childAttempt: "hesitait",
    childProfile: {
      childId: "c7",
      age: 11,
      grade: "6",
      spellingLevel: "on-grade",
    },
    wordMetadata: {
      definition: "to pause before acting",
      partOfSpeech: "verb",
      pronunciation: "HEZ-uh-tate",
    },
    missSignals: {
      isCorrect: false,
      nearMiss: false,
      missingLetters: ["e"],
      extraLetters: [],
      substitutedLetters: ["i for e"],
      transposedLetters: [],
      repeatedLetterIssue: false,
      likelyRushed: false,
      editDistance: 2,
    },
    structuralHints: {
      syllables: ["hes", "i", "tate"],
      likelyChunks: ["hesi", "tate"],
      detectedPatterns: ["-ate"],
      likelySuffix: "ate",
    },
    sessionContext: {
      mode: "practice",
      previousAttemptsOnThisWord: 0,
      previousMissPatterns: [],
      recentlyPracticedWords: ["demonstrate", "celebrate"],
    },
  };

  const hints = getReferenceHints(input);
  const txtHint = hints.find(
    (hint) => hint.root.includes("-ate") && hint.source === "suffix_list_csv",
  );

  assert.equal(txtHint?.role, "suffix_family");
  assert.equal(txtHint?.source, "suffix_list_csv");
});

test("uses short memory tip guidance for Level 2 runtime prompts", () => {
  const input: SpellingCoachInput = {
    targetWord: "hesitate",
    childAttempt: "hesitait",
    childProfile: {
      childId: "c7",
      age: 11,
      grade: "6",
      spellingLevel: "on-grade",
    },
    wordMetadata: {
      definition: "to pause before acting",
      partOfSpeech: "verb",
      pronunciation: "HEZ-uh-tate",
    },
    missSignals: {
      isCorrect: false,
      nearMiss: false,
      missingLetters: ["e"],
      extraLetters: [],
      substitutedLetters: ["i for e"],
      transposedLetters: [],
      repeatedLetterIssue: false,
      likelyRushed: false,
      editDistance: 2,
    },
    structuralHints: {
      syllables: ["hes", "i", "tate"],
      likelyChunks: ["hesi", "tate"],
      detectedPatterns: ["-ate"],
      likelySuffix: "ate",
    },
    sessionContext: {
      mode: "practice",
      previousAttemptsOnThisWord: 0,
      previousMissPatterns: [],
      recentlyPracticedWords: ["demonstrate", "celebrate"],
    },
  };

  const prompt = buildSpellingCoachPrompt(input);

  assert.equal(
    prompt.includes(
      "For Level 2 words, always provide coachingText.memoryTip even if the child spelled the word correctly.",
    ),
    true,
  );
  assert.equal(
    prompt.includes(
      "For Level 3 words, coachingText.memoryTip may be up to two short lines when that genuinely helps recall.",
    ),
    false,
  );
});

test("allows a longer memory tip for Level 3 miss-only prompts", () => {
  const input: SpellingCoachInput = {
    targetWord: "arachnophagous",
    childAttempt: "arachnofagus",
    childProfile: {
      childId: "c17",
      age: 12,
      grade: "7",
      spellingLevel: "advanced",
    },
    wordMetadata: {
      definition: "Describes animals that eat spiders.",
      partOfSpeech: "adjective",
      exampleSentence: "Some birds are arachnophagous and like to eat spiders.",
      pronunciation: "uh-RAK-no-FAY-gus",
    },
    missSignals: {
      isCorrect: false,
      nearMiss: false,
      missingLetters: ["ph", "ou"],
      extraLetters: [],
      substitutedLetters: [],
      transposedLetters: [],
      repeatedLetterIssue: false,
      likelyRushed: false,
      editDistance: 3,
    },
    structuralHints: {
      syllables: ["a", "rach", "no", "pha", "gous"],
      likelyChunks: ["arachno", "phagous"],
      detectedPatterns: ["ch says k", "ph says f", "-gous"],
      likelySuffix: "gous",
    },
    sessionContext: {
      mode: "practice",
      previousAttemptsOnThisWord: 0,
      previousMissPatterns: [],
      recentlyPracticedWords: ["arachnid", "phosphorus"],
    },
  };

  const missOnlyPrompt = buildMissOnlyPrompt(
    input,
    JSON.stringify(
      {
        wordTeaching: {
          conceptTeaching: {
            summary: "",
            meaningFocus: "",
            originFocus: "",
            morphologyFocus: "",
            originLabels: [],
            morphologyLabels: [],
          },
        },
        wordBreakdown: {
          displayChunks: ["arachn", "ophag", "ous"],
          alternateDisplayChunks: [],
          chunkReason: "",
          matchedPatterns: [],
        },
        conceptLabels: {
          originLabels: [],
          patternLabels: [],
          morphologyLabels: [],
        },
      },
      null,
      2,
    ),
  );

  assert.equal(
    missOnlyPrompt.includes(
      "coachingText.memoryTip may be up to two short lines when that genuinely helps recall.",
    ),
    true,
  );
  assert.equal(
    missOnlyPrompt.includes(
      "Keep coachingText.memoryTip focused on memory support rather than turning it into another explanation.",
    ),
    true,
  );
  assert.equal(
    missOnlyPrompt.includes(
      "Do not use phrases like 'the child wrote', 'the child added', or 'the child substituted' in miss analysis text.",
    ),
    false,
  );
  assert.equal(
    missOnlyPrompt.includes(
      "Never mention field names, keys, or booleans such as rawSignals, substitutedLetters, extraLetters, repeatedLetterIssue, likelyChunks, detectedPatterns, true, or false.",
    ),
    false,
  );
  assert.equal(
    missOnlyPrompt.includes(
      "Independently judge missAnalysis.likelyWrongWordInterpretation by asking whether the attempt itself is another real word or a real related word-form.",
    ),
    true,
  );
  assert.equal(missOnlyPrompt.includes("wrongWordInterpretationHints"), true);
  assert.equal(
    missOnlyPrompt.includes("\"substantialStructuralOverlap\""),
    true,
  );
});

test("streaming runtime prompt includes mandatory memory tip instruction for correct Level 2 words", () => {
  const input: SpellingCoachInput = {
    targetWord: "toreador",
    childAttempt: "toreador",
    childProfile: {
      childId: "c-correct",
      age: 11,
      grade: "5",
      spellingLevel: "on-grade",
    },
    wordMetadata: {
      definition: "A bullfighter who fights on foot.",
      partOfSpeech: "noun",
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
      syllables: ["to", "rea", "dor"],
      likelyChunks: ["to", "rea", "dor"],
      detectedPatterns: [],
    },
    sessionContext: {
      mode: "practice",
      previousAttemptsOnThisWord: 0,
      previousMissPatterns: [],
      recentlyPracticedWords: [],
    },
    level: 2,
  };

  const prompt = buildStreamingRuntimePrompt(input, "{}");

  assert.equal(
    prompt.includes(
      "The child spelled the word correctly. You MUST still output the [[MEMORY_TIP]] section with a helpful memory cue for this word.",
    ),
    true,
  );
  assert.equal(
    prompt.includes("Do NOT leave the [[MEMORY_TIP]] section empty. It is required."),
    true,
  );
});

test("streaming runtime prompt does NOT add mandatory memory tip instruction for correct Level 1 words", () => {
  const input: SpellingCoachInput = {
    targetWord: "cat",
    childAttempt: "cat",
    childProfile: {
      childId: "c-l1",
      age: 6,
      grade: "1",
      spellingLevel: "on-grade",
    },
    wordMetadata: {
      definition: "A small domesticated animal.",
      partOfSpeech: "noun",
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
      syllables: ["cat"],
      likelyChunks: ["cat"],
      detectedPatterns: [],
    },
    sessionContext: {
      mode: "practice",
      previousAttemptsOnThisWord: 0,
      previousMissPatterns: [],
      recentlyPracticedWords: [],
    },
    level: 1,
  };

  const prompt = buildStreamingRuntimePrompt(input, "{}");

  assert.equal(
    prompt.includes(
      "The child spelled the word correctly. You MUST still output the [[MEMORY_TIP]] section with a helpful memory cue for this word.",
    ),
    false,
  );
});

test("supports direct runtime path with the same validated output", async () => {
  const input: SpellingCoachInput = {
    targetWord: "pulpit",
    childAttempt: "pulpit",
    childProfile: {
      childId: "c2",
      age: 9,
      grade: "4",
      spellingLevel: "on-grade",
    },
    wordMetadata: {
      definition: "a raised platform in a church",
      partOfSpeech: "noun",
      pronunciation: "PUHL-pit",
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
      syllables: ["pul", "pit"],
      likelyChunks: ["pul", "pit"],
      detectedPatterns: ["closed syllables"],
    },
    sessionContext: {
      mode: "practice",
      previousAttemptsOnThisWord: 0,
      previousMissPatterns: [],
      recentlyPracticedWords: ["puppet", "pulp"],
    },
  };

  const expected = makeOutput({
    correctness: {
      isCorrect: true,
      reinforceSuccess: true,
    },
    missAnalysis: {
      summary: "The word was spelled correctly.",
      primaryErrorType: null,
      secondaryErrorTypes: [],
      errorTypeEvidence: {},
      primaryErrorFocus: "Accurate spelling",
      usedMeaningDisambiguationWell: true,
    },
    errorRelevance: {
      mostRelevantToError: "unclear",
      confidence: 0.4,
      reason: "There is no miss to diagnose, so no single teaching layer is clearly tied to an error.",
    },
    teachingDecision: {
      strategy: "pattern",
      primaryFocus: "Notice the two short-vowel chunks pul + pit.",
      secondaryFocuses: [],
      confidence: 0.84,
      rationale: "The child got it right, so a light reusable pattern reminder is enough.",
    },
    coachingText: {
      shortFeedback: "Correct. You spelled pulpit exactly right.",
      fullExplanation: "Nice job. One quick reminder: pulpit has two short chunks, pul + pit.",
      memoryTip: "",
      sayAloudTip: "Say pul-pit.",
    },
    wordBreakdown: {
      displayChunks: ["pul", "pit"],
      chunkReason: "The chunks reinforce the correct short-vowel structure without adding extra complexity.",
    },
    conceptLabels: {
      originLabels: [],
      patternLabels: ["closed syllables"],
      morphologyLabels: [],
    },
    nextStep: {
      practiceFocus: "Keep checking both short-vowel chunks in similar two-part words.",
      shouldReviewSoon: false,
      suggestedSimilarWordTypes: ["two-syllable short-vowel words"],
    },
  });

  const result = await runSpellingCoachAgent(input, {
    runtime: "direct",
    directModel: createDirectMockModel(expected),
  });

  assert.equal(result.correctness.isCorrect, true);
  assert.equal(result.teachingDecision.strategy, "pattern");
});

test("builds structural wrong-word interpretation hints for consternation -> constellation", () => {
  const input: SpellingCoachInput = {
    targetWord: "consternation",
    childAttempt: "constellation",
    childProfile: baseProfile,
    wordMetadata: {
      definition: "a feeling of worry or confusion.",
      origin: "Latin",
      partOfSpeech: "noun",
      exampleSentence: "The sudden noise caused consternation in the room.",
    },
    missSignals: {
      isCorrect: false,
      nearMiss: false,
      missingLetters: [],
      extraLetters: [],
      substitutedLetters: ["l for r", "l for n"],
      transposedLetters: [],
      repeatedLetterIssue: true,
      likelyRushed: false,
      editDistance: 3,
    },
    structuralHints: {
      syllables: ["con", "ster", "na", "tion"],
      likelyChunks: ["con", "stern", "ation"],
      detectedPatterns: ["tion"],
      likelySuffix: "ation",
    },
    sessionContext: {
      mode: "practice",
      previousAttemptsOnThisWord: 0,
      previousMissPatterns: [],
      recentlyPracticedWords: [],
    },
  };

  const missOnlyPrompt = buildMissOnlyPrompt(
    input,
    JSON.stringify(
      {
        wordTeaching: {
          conceptTeaching: {
            summary: "",
            meaningFocus: "",
            originFocus: "",
            morphologyFocus: "",
            originLabels: [],
            morphologyLabels: [],
          },
        },
        wordBreakdown: {
          displayChunks: ["con", "stern", "ation"],
          alternateDisplayChunks: [],
          chunkReason: "",
          matchedPatterns: [],
        },
        conceptLabels: {
          originLabels: [],
          patternLabels: [],
          morphologyLabels: [],
        },
      },
      null,
      2,
    ),
  );

  const evidenceMatch = missOnlyPrompt.match(
    /Deterministic miss evidence JSON:\n\n([\s\S]*?)\n\nPrecomputed word-level teaching JSON:/,
  );
  assert.equal(Boolean(evidenceMatch), true);

  const evidence = JSON.parse(evidenceMatch?.[1] ?? "{}");
  const hints = evidence.wrongWordInterpretationHints;

  assert.equal(hints.targetNormalized, "consternation");
  assert.equal(hints.attemptNormalized, "constellation");
  assert.equal(hints.sharedPrefixLength >= 6, true);
  assert.equal(hints.sharedSuffixLength >= 5, true);
  assert.equal(hints.longestCommonSubsequenceRatio >= 0.75, true);
  assert.equal(hints.bigramOverlapRatio >= 0.6, true);
  assert.equal(hints.substantialStructuralOverlap, true);
});

test("replaces generic precompute chunk reasoning with concrete chunk split", () => {
  const normalized = normalizeWordTeachingPrecomputeChunkReason({
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
    wordBreakdown: {
      displayChunks: ["cent", "er"],
      chunkReason:
        "These chunks are easy to say and remember, breaking the word into two simple parts that match natural sound groups.",
    },
    conceptLabels: {
      originLabels: [],
      patternLabels: [],
      morphologyLabels: [],
    },
  });

  assert.equal(normalized.wordBreakdown.chunkReason, "The word breaks as cent + er.");
});

test("replaces generic output chunk reasoning with concrete chunk split", () => {
  const normalized = normalizeSpellingCoachOutputChunkReason(
    makeOutput({
      wordBreakdown: {
        displayChunks: ["mon", "ster"],
        chunkReason:
          "These chunks are easy to say and remember, breaking the word into two simple parts that match natural sound groups.",
      },
    }),
  );

  assert.equal(normalized.wordBreakdown.chunkReason, "The word breaks as mon + ster.");
});

test("picks next word from the requested level", () => {
  const word = pickNextWord("2");
  assert.equal(word.level, "2");
});

test("builds a coaching input from app-level request data", () => {
  const input = buildSpellingCoachInput({
    targetWord: "abandon",
    childAttempt: "abando",
    childProfile: {
      childId: "c1",
      age: 9,
      grade: "4",
      spellingLevel: "on-grade",
    },
    supportsUsed: {
      definitionViewed: true,
      exampleViewed: false,
      originViewed: false,
    },
    sessionContext: {
      mode: "practice",
      previousAttemptsOnThisWord: 1,
      previousMissPatterns: ["missing ending"],
      recentlyPracticedWords: ["about"],
    },
  });

  assert.equal(input.targetWord, "abandon");
  assert.equal(input.missSignals.isCorrect, false);
  assert.equal(input.missSignals.editDistance > 0, true);
  assert.equal(input.wordMetadata?.definition?.includes("leave"), true);
  assert.equal(
    input.missSignals.wrongWordInterpretationHints?.targetNormalized,
    "abandon",
  );
  assert.equal(
    input.missSignals.deterministicLikelyWrongWordInterpretation,
    false,
  );
});

test("builds structured deterministic miss facts for dropped double letters", () => {
  const input = buildSpellingCoachInput({
    targetWord: "especially",
    childAttempt: "especialy",
    childProfile: {
      childId: "c1",
      age: 9,
      grade: "4",
      spellingLevel: "on-grade",
    },
    sessionContext: {
      mode: "practice",
      previousAttemptsOnThisWord: 0,
      previousMissPatterns: [],
      recentlyPracticedWords: [],
    },
  });

  assert.equal(input.missSignals.repeatedLetterIssue, true);
  assert.equal(input.missSignals.doubleLetterMismatch?.detected, true);
  assert.deepEqual(input.missSignals.doubleLetterMismatch?.missingFromDouble, [
    "l",
  ]);
});

test("keeps vowel substitution pairs limited to true vowel-for-vowel changes", () => {
  const input = buildSpellingCoachInput({
    targetWord: "manufacture",
    childAttempt: "manufakcher",
    childProfile: {
      childId: "c1",
      age: 10,
      grade: "5",
      spellingLevel: "on-grade",
    },
    sessionContext: {
      mode: "practice",
      previousAttemptsOnThisWord: 0,
      previousMissPatterns: [],
      recentlyPracticedWords: [],
    },
  });

  assert.deepEqual(input.missSignals.vowelSubstitutionPairs ?? [], []);
});

test("does not flag wrong-word interpretation for nonword chunk rewrites", () => {
  const input = buildSpellingCoachInput({
    targetWord: "manufacture",
    childAttempt: "manufakcher",
    childProfile: {
      childId: "c1",
      age: 10,
      grade: "5",
      spellingLevel: "on-grade",
    },
    sessionContext: {
      mode: "practice",
      previousAttemptsOnThisWord: 0,
      previousMissPatterns: [],
      recentlyPracticedWords: [],
    },
  });

  assert.equal(
    input.missSignals.deterministicLikelyWrongWordInterpretation,
    false,
  );
  assert.equal(input.missSignals.chunkMismatchFacts?.[0]?.expectedChunk, "facture");
  assert.equal(input.missSignals.chunkMismatchFacts?.[0]?.observedFragment, "fakcher");
  assert.equal(input.missSignals.chunkMismatchFacts?.[0]?.phoneticRewrite, true);
});

test("builds a public word response from generated word data", () => {
  const word = getWordByText("phlox");
  assert.ok(word);

  const response = buildWordResponse(word);
  assert.equal(response.word, "phlox");
  assert.equal(response.level, word.level);
  assert.equal(typeof response.definition, "string");
  assert.equal(response.exampleSentence.toLowerCase().includes("phlox"), false);
});

test("builds a detailed word response with offline precomputed metadata", () => {
  const word = getWordByText("dentifrice");
  assert.ok(word);

  const response = buildDetailedWordResponse(word);
  assert.equal(response.word, "dentifrice");
  assert.equal(response.definition.includes("toothpaste"), true);
  assert.equal(response.exampleSentence.includes("dentifrice"), true);
  assert.deepEqual(response.wordBreakdown?.displayChunks, ["dent", "i", "frice"]);
  assert.equal(response.wordBreakdown?.matchedPatterns.length, 4);
  assert.deepEqual(response.conceptLabels?.originLabels, ["Latin"]);
  assert.equal(response.wordTeaching?.conceptTeaching.relatedForms.length, 0);
  assert.equal(response.phonemeMetadata?.source, "g2p-en");
  assert.equal(
    response.phonemeMetadata?.sayAloudTip,
    "Say it slowly: DEN-tuh-fruhs.\nThe final e is there, but the i does not say its name.",
  );
  assert.equal(response.phonemeMetadata?.trickyParts[0]?.soundsLike, "s");
});

test("masks the target word in example sentences for UI responses", () => {
  const masked = maskWordInExampleSentence(
    "The affenpinscher trotted proudly around the show ring.",
    "affenpinscher",
  );

  assert.equal(
    masked,
    "The ***** trotted proudly around the show ring.",
  );
});

test("masks contained word forms in example sentences for UI responses", () => {
  const masked = maskWordInExampleSentence(
    "The town lionized the hero after the big game.",
    "lionize",
  );

  assert.equal(
    masked,
    "The town *****d the hero after the big game.",
  );
});

test("masks contained word forms in definitions for UI responses", () => {
  const word = getWordByText("wensleydale");

  assert.ok(word);

  const response = buildWordResponse(word);

  assert.equal(
    response.definition,
    "A type of cheese from *****, England.",
  );
});

test("parses word search query defaults and guardrails", () => {
  const query = WordSearchQuerySchema.parse({
    q: "dent",
  });

  assert.equal(query.mode, "startsWith");
  assert.equal(query.limit, 20);
});

test("rejects too-short word search queries", () => {
  assert.throws(() => {
    WordSearchQuerySchema.parse({
      q: "d",
      mode: "contains",
    });
  });
});

test("finds words by startsWith search", () => {
  const results = searchWords("dent", "startsWith", 10);
  assert.equal(results.some((entry) => entry.word === "dentifrice"), true);
  assert.equal(
    results.every((entry) => entry.word.toLowerCase().startsWith("dent")),
    true,
  );
});

test("finds words by contains search", () => {
  const results = searchWords("frice", "contains", 10);
  assert.equal(results.some((entry) => entry.word === "dentifrice"), true);
});

test("prioritizes exact matches in word search results", () => {
  const results = searchWords("phlox", "contains", 10);
  assert.equal(results[0]?.word, "phlox");
});

test("caps word search result count at requested limit", () => {
  const results = searchWords("tion", "contains", 3);
  assert.ok(results.length <= 3);
});

test("accepts explicit empty concept teaching fields when concept support is weak", async () => {
  const input: SpellingCoachInput = {
    targetWord: "pulpit",
    childAttempt: "pulpet",
    childProfile: {
      childId: "c8",
      age: 9,
      grade: "4",
      spellingLevel: "on-grade",
    },
    wordMetadata: {
      definition: "a raised platform in a church",
      partOfSpeech: "noun",
      pronunciation: "PUHL-pit",
    },
    missSignals: {
      isCorrect: false,
      nearMiss: true,
      missingLetters: [],
      extraLetters: [],
      substitutedLetters: ["e for i"],
      transposedLetters: [],
      repeatedLetterIssue: false,
      likelyRushed: false,
      editDistance: 1,
    },
    structuralHints: {
      syllables: ["pul", "pit"],
      likelyChunks: ["pul", "pit"],
      detectedPatterns: ["closed syllables"],
    },
    sessionContext: {
      mode: "practice",
      previousAttemptsOnThisWord: 0,
      previousMissPatterns: [],
      recentlyPracticedWords: [],
    },
  };

  const result = await runSpellingCoachAgent(input, {
    agent: createMockAgent(
      makeOutput({
        missAnalysis: {
          summary: "The second vowel was changed from i to e.",
          primaryErrorType: "vowel_confusion",
          secondaryErrorTypes: [],
          errorTypeEvidence: {
            vowel_confusion: "The attempt reflects the wrong vowel sound choice.",
          },
          primaryErrorFocus: "Keep the short i in the second chunk.",
        },
        wordTeaching: {
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
          mostRelevantToError: "form",
          confidence: 0.82,
          reason: "The mistake is a vowel substitution inside the second chunk.",
        },
      }),
    ),
  });

  assert.equal(result.wordTeaching.conceptTeaching.summary, "");
  assert.deepEqual(result.wordTeaching.conceptTeaching.originLabels, []);
});

test("supports unclear error relevance below the confidence threshold", async () => {
  const input: SpellingCoachInput = {
    targetWord: "phlox",
    childAttempt: "flox",
    childProfile: baseProfile,
    wordMetadata: {
      definition: "a flowering plant with clustered blooms",
      origin: "Greek",
      partOfSpeech: "noun",
      pronunciation: "floks",
    },
    missSignals: {
      isCorrect: false,
      nearMiss: true,
      missingLetters: ["h"],
      extraLetters: [],
      substitutedLetters: ["f for ph"],
      transposedLetters: [],
      repeatedLetterIssue: false,
      likelyRushed: false,
      editDistance: 1,
    },
    structuralHints: {
      syllables: ["phlox"],
      likelyChunks: ["ph", "lox"],
      detectedPatterns: ["ph says f", "x ending"],
    },
    sessionContext: {
      mode: "practice",
      previousAttemptsOnThisWord: 0,
      previousMissPatterns: [],
      recentlyPracticedWords: [],
    },
  };

  const result = await runSpellingCoachAgent(input, {
    agent: createMockAgent(
      makeOutput({
        missAnalysis: {
          summary: "The child simplified ph to f.",
          primaryErrorType: "phonetic_spelling",
          secondaryErrorTypes: [],
          errorTypeEvidence: {
            phonetic_spelling: "The attempt was written more by sound than by standard spelling.",
          },
          primaryErrorFocus: "Use ph for the f sound in this word.",
        },
        wordTeaching: {
          conceptTeaching: {
            summary: "This is the flower word.",
            meaningFocus: "flower name",
            originFocus: "Greek-derived",
            morphologyFocus: "",
            originLabels: ["greek-derived"],
            morphologyLabels: [],
          },
        },
        errorRelevance: {
          mostRelevantToError: "unclear",
          confidence: 0.6,
          reason: "Form looks more likely, but the evidence is not strong enough for a confident call.",
        },
      }),
    ),
  });

  assert.equal(result.errorRelevance.mostRelevantToError, "unclear");
  assert.equal(result.errorRelevance.confidence < 0.75, true);
});

test("prefers vowel confusion over letter substitution for vowel-for-vowel misses", async () => {
  const input: SpellingCoachInput = {
    targetWord: "turpentine",
    childAttempt: "terpentine",
    childProfile: baseProfile,
    wordMetadata: {
      definition: "a strong-smelling liquid from pine trees",
      origin: "Greek",
      partOfSpeech: "noun",
      pronunciation: "TER-pin-teen",
    },
    missSignals: {
      isCorrect: false,
      nearMiss: true,
      missingLetters: [],
      extraLetters: [],
      substitutedLetters: ["e for u"],
      transposedLetters: [],
      repeatedLetterIssue: false,
      likelyRushed: false,
      editDistance: 1,
    },
    structuralHints: {
      syllables: ["tur", "pen", "tine"],
      likelyChunks: ["tur", "pen", "tine"],
      detectedPatterns: ["vowel change"],
    },
    sessionContext: {
      mode: "practice",
      previousAttemptsOnThisWord: 0,
      previousMissPatterns: [],
      recentlyPracticedWords: [],
    },
  };

  const result = await runSpellingCoachAgent(input, {
    agent: createMockAgent(
      makeOutput({
        missAnalysis: {
          summary:
            "The spelling uses 'e' instead of 'u' in the first syllable.",
          primaryErrorType: "letter_substitution",
          secondaryErrorTypes: [],
          errorTypeEvidence: {
            letter_substitution:
              "The spelling substitutes one letter for another in the first syllable.",
          },
          primaryErrorFocus:
            "Keep the vowel 'u' in the first syllable instead of changing it to 'e'.",
        },
      }),
    ),
  });

  assert.equal(result.missAnalysis.primaryErrorType, "vowel_confusion");
  assert.deepEqual(result.missAnalysis.secondaryErrorTypes, []);
  assert.equal(
    result.missAnalysis.errorTypeEvidence.vowel_confusion,
    "The spelling substitutes one letter for another in the first syllable.",
  );
  assert.equal(
    "letter_substitution" in result.missAnalysis.errorTypeEvidence,
    false,
  );
});

test("prefers vowel confusion when a 2-vowel span is transposed", async () => {
  const input: SpellingCoachInput = {
    targetWord: "friend",
    childAttempt: "freind",
    childProfile: baseProfile,
    wordMetadata: {
      definition: "a person you like and trust",
      origin: "Old English",
      partOfSpeech: "noun",
      pronunciation: "frend",
    },
    missSignals: {
      isCorrect: false,
      nearMiss: true,
      missingLetters: [],
      extraLetters: [],
      substitutedLetters: [],
      transposedLetters: ["ie"],
      repeatedLetterIssue: false,
      likelyRushed: false,
      editDistance: 1,
    },
    structuralHints: {
      syllables: ["friend"],
      likelyChunks: ["friend"],
      detectedPatterns: ["ie/ei"],
    },
    sessionContext: {
      mode: "practice",
      previousAttemptsOnThisWord: 0,
      previousMissPatterns: [],
      recentlyPracticedWords: [],
    },
  };

  const result = await runSpellingCoachAgent(input, {
    agent: createMockAgent(
      makeOutput({
        missAnalysis: {
          summary:
            "The spelling uses the right letters in the vowel team, but they are not in the correct order.",
          primaryErrorType: "letter_substitution",
          secondaryErrorTypes: [],
          errorTypeEvidence: {
            letter_substitution:
              "The spelling changes the letter sequence in the vowel team.",
          },
          primaryErrorFocus:
            "Keep the letters in the vowel team in the correct order.",
        },
      }),
    ),
  });

  assert.equal(result.missAnalysis.primaryErrorType, "vowel_confusion");
  assert.equal(
    result.missAnalysis.errorTypeEvidence.vowel_confusion,
    "The spelling changes the letter sequence in the vowel team.",
  );
  assert.equal(
    "letter_substitution" in result.missAnalysis.errorTypeEvidence,
    false,
  );
});

test("treats a 2-vowel transposed span as vowel confusion instead of letter transposition", () => {
  const input: SpellingCoachInput = {
    targetWord: "friend",
    childAttempt: "freind",
    childProfile: baseProfile,
    wordMetadata: {
      definition: "a person you like and trust",
      origin: "Old English",
      partOfSpeech: "noun",
      pronunciation: "frend",
    },
    missSignals: {
      isCorrect: false,
      nearMiss: true,
      missingLetters: [],
      extraLetters: [],
      substitutedLetters: [],
      transposedLetters: ["ie"],
      repeatedLetterIssue: false,
      likelyRushed: false,
      editDistance: 1,
    },
    structuralHints: {
      syllables: ["friend"],
      likelyChunks: ["friend"],
      detectedPatterns: ["vowel team ie/ei"],
    },
    sessionContext: {
      mode: "practice",
      previousAttemptsOnThisWord: 0,
      previousMissPatterns: [],
      recentlyPracticedWords: [],
    },
  };

  const result = normalizeMissAnalysisErrorTypes(
    input,
    makeOutput({
      missAnalysis: {
        summary: "The spelling swaps two letters in the vowel team.",
        primaryErrorType: "letter_transposition",
        secondaryErrorTypes: [],
        errorTypeEvidence: {
          letter_transposition:
            "The vowel letters are in the wrong order in the team.",
        },
        primaryErrorFocus: "Keep the vowel team in the correct order.",
      },
    }),
  );

  assert.equal(result.missAnalysis.primaryErrorType, "vowel_confusion");
  assert.equal(
    result.missAnalysis.secondaryErrorTypes.includes("letter_transposition"),
    false,
  );
});

test("prefers double-letter error when the raw signals show a repeated-letter issue", async () => {
  const input: SpellingCoachInput = {
    targetWord: "accommodate",
    childAttempt: "acommodate",
    childProfile: baseProfile,
    wordMetadata: {
      definition: "to provide room or space for",
      origin: "Latin",
      partOfSpeech: "verb",
      pronunciation: "uh-KAH-muh-date",
    },
    missSignals: {
      isCorrect: false,
      nearMiss: true,
      missingLetters: ["c"],
      extraLetters: [],
      substitutedLetters: [],
      transposedLetters: [],
      repeatedLetterIssue: true,
      likelyRushed: false,
      editDistance: 1,
    },
    structuralHints: {
      syllables: ["ac", "com", "mo", "date"],
      likelyChunks: ["ac", "commodate"],
      detectedPatterns: ["double consonant cc", "double consonant mm"],
    },
    sessionContext: {
      mode: "practice",
      previousAttemptsOnThisWord: 0,
      previousMissPatterns: [],
      recentlyPracticedWords: [],
    },
  };

  const result = await runSpellingCoachAgent(input, {
    agent: createMockAgent(
      makeOutput({
        missAnalysis: {
          summary:
            "The spelling leaves out one of the c letters near the start of the word.",
          primaryErrorType: "missing_letter",
          secondaryErrorTypes: [],
          errorTypeEvidence: {
            missing_letter:
              "One required letter from the double consonant is missing.",
          },
          primaryErrorFocus:
            "Keep both c letters together near the start of the word.",
        },
      }),
    ),
  });

  assert.equal(result.missAnalysis.primaryErrorType, "double_letter_error");
  assert.equal(
    result.missAnalysis.errorTypeEvidence.double_letter_error,
    "One required letter from the double consonant is missing.",
  );
  assert.equal(
    "missing_letter" in result.missAnalysis.errorTypeEvidence,
    true,
  );
});

test("prefers double-letter error when one letter from a doubled pair is dropped", async () => {
  const input: SpellingCoachInput = {
    targetWord: "especially",
    childAttempt: "especialy",
    childProfile: baseProfile,
    wordMetadata: {
      definition: "more than usual; particularly",
      origin: "Latin",
      partOfSpeech: "adverb",
      pronunciation: "ih-SPESH-uh-lee",
    },
    missSignals: {
      isCorrect: false,
      nearMiss: true,
      missingLetters: ["l"],
      extraLetters: [],
      substitutedLetters: [],
      transposedLetters: [],
      repeatedLetterIssue: false,
      doubleLetterMismatch: {
        detected: true,
        missingFromDouble: ["l"],
        extraDouble: [],
        affectedLetters: ["l"],
      },
      likelyRushed: true,
      editDistance: 1,
    },
    structuralHints: {
      syllables: ["es", "pe", "cial", "ly"],
      likelyChunks: ["es", "pe", "cial", "ly"],
      detectedPatterns: ["double consonant ll", "ending -ly"],
    },
    sessionContext: {
      mode: "practice",
      previousAttemptsOnThisWord: 0,
      previousMissPatterns: [],
      recentlyPracticedWords: [],
    },
  };

  const result = await runSpellingCoachAgent(input, {
    agent: createMockAgent(
      makeOutput({
        missAnalysis: {
          summary:
            "The spelling is missing one letter near the end of the word.",
          primaryErrorType: "missing_letter",
          secondaryErrorTypes: [],
          errorTypeEvidence: {
            missing_letter:
              "One letter is missing near the final doubled consonant.",
          },
          primaryErrorFocus:
            "Keep the final letters in the correct ending pattern.",
        },
      }),
    ),
  });

  assert.equal(result.missAnalysis.primaryErrorType, "double_letter_error");
  assert.equal(
    result.missAnalysis.errorTypeEvidence.double_letter_error,
    "One letter is missing near the final doubled consonant.",
  );
  assert.equal(
    "missing_letter" in result.missAnalysis.errorTypeEvidence,
    true,
  );
});

test("does not prefer double-letter error for unrelated substitution misses", async () => {
  const input: SpellingCoachInput = {
    targetWord: "irrevocable",
    childAttempt: "irrevokable",
    childProfile: baseProfile,
    wordMetadata: {
      definition: "impossible to change or take back",
      origin: "Latin",
      partOfSpeech: "adjective",
      pronunciation: "i-REV-uh-kuh-buhl",
    },
    missSignals: {
      isCorrect: false,
      nearMiss: true,
      missingLetters: [],
      extraLetters: [],
      substitutedLetters: ["k for c"],
      transposedLetters: [],
      repeatedLetterIssue: true,
      likelyRushed: false,
      editDistance: 1,
    },
    structuralHints: {
      syllables: ["ir", "rev", "o", "ca", "ble"],
      likelyChunks: ["irr", "evo", "ca", "ble"],
      detectedPatterns: ["double consonant rr", "ending -le"],
    },
    sessionContext: {
      mode: "practice",
      previousAttemptsOnThisWord: 0,
      previousMissPatterns: [],
      recentlyPracticedWords: [],
    },
  };

  const result = await runSpellingCoachAgent(input, {
    agent: createMockAgent(
      makeOutput({
        missAnalysis: {
          summary:
            "The spelling uses 'k' instead of 'c' in the middle of the word.",
          primaryErrorType: "letter_substitution",
          secondaryErrorTypes: [],
          errorTypeEvidence: {
            letter_substitution:
              "The spelling replaces the expected 'c' with 'k' in the root part of the word.",
          },
          primaryErrorFocus:
            "Keep the letter 'c' in the root part of the word instead of changing it to 'k'.",
        },
      }),
    ),
  });

  assert.equal(result.missAnalysis.primaryErrorType, "letter_substitution");
  assert.equal(
    result.missAnalysis.errorTypeEvidence.letter_substitution,
    "The spelling replaces the expected 'c' with 'k' in the root part of the word.",
  );
  assert.equal(
    "double_letter_error" in result.missAnalysis.errorTypeEvidence,
    false,
  );
});

test("classifies irrevocabel as an ending confusion miss", async () => {
  const input: SpellingCoachInput = {
    targetWord: "irrevocable",
    childAttempt: "irrevocabel",
    childProfile: baseProfile,
    wordMetadata: {
      definition: "impossible to change or take back",
      origin: "Latin",
      partOfSpeech: "adjective",
      pronunciation: "i-REV-uh-kuh-buhl",
    },
    missSignals: {
      isCorrect: false,
      nearMiss: true,
      missingLetters: [],
      extraLetters: [],
      substitutedLetters: [],
      transposedLetters: ["el"],
      repeatedLetterIssue: true,
      endingConfusionFacts: [
        {
          suffix: "able",
          attemptedEnding: "abel",
          phoneticRewrite: false,
        },
      ],
      likelyRushed: true,
      editDistance: 2,
    },
    structuralHints: {
      syllables: ["ir", "rev", "o", "ca", "ble"],
      likelyChunks: ["irr", "evo", "ca", "ble"],
      detectedPatterns: ["double consonant rr", "ending -le"],
    },
    sessionContext: {
      mode: "practice",
      previousAttemptsOnThisWord: 0,
      previousMissPatterns: [],
      recentlyPracticedWords: [],
    },
  };

  const result = await runSpellingCoachAgent(input, {
    agent: createMockAgent(
      makeOutput({
        missAnalysis: {
          summary:
            "The spelling swaps the final two letters in the ending.",
          primaryErrorType: "letter_substitution",
          secondaryErrorTypes: ["likely_rushed"],
          errorTypeEvidence: {
            letter_substitution:
              "The final two letters are reversed from 'le' to 'el'.",
            likely_rushed:
              "The attempt was likely rushed, contributing to the letter order mistake.",
          },
          primaryErrorFocus:
            "Keep the ending letters in the correct order as 'le'.",
        },
      }),
    ),
  });

  assert.equal(result.missAnalysis.primaryErrorType, "letter_transposition");
  assert.equal(
    result.missAnalysis.secondaryErrorTypes.includes("ending_confusion"),
    true,
  );
  assert.equal(
    result.missAnalysis.secondaryErrorTypes.includes("pattern_rule_mismatch"),
    true,
  );
  assert.equal(
    result.missAnalysis.errorTypeEvidence.letter_transposition,
    "The final two letters are reversed from 'le' to 'el'.",
  );
});

test("classifies errevocable as a vowel confusion miss", async () => {
  const input: SpellingCoachInput = {
    targetWord: "irrevocable",
    childAttempt: "errevocable",
    childProfile: baseProfile,
    wordMetadata: {
      definition: "impossible to change or take back",
      origin: "Latin",
      partOfSpeech: "adjective",
      pronunciation: "i-REV-uh-kuh-buhl",
    },
    missSignals: {
      isCorrect: false,
      nearMiss: true,
      missingLetters: [],
      extraLetters: [],
      substitutedLetters: ["e for i"],
      transposedLetters: [],
      repeatedLetterIssue: true,
      likelyRushed: false,
      editDistance: 1,
    },
    structuralHints: {
      syllables: ["ir", "rev", "o", "ca", "ble"],
      likelyChunks: ["irr", "evo", "ca", "ble"],
      detectedPatterns: ["double consonant rr", "ending -le"],
    },
    sessionContext: {
      mode: "practice",
      previousAttemptsOnThisWord: 0,
      previousMissPatterns: [],
      recentlyPracticedWords: [],
    },
  };

  const result = await runSpellingCoachAgent(input, {
    agent: createMockAgent(
      makeOutput({
        missAnalysis: {
          summary:
            "The spelling uses 'e' instead of 'i' at the start of the word.",
          primaryErrorType: "letter_substitution",
          secondaryErrorTypes: [],
          errorTypeEvidence: {
            letter_substitution:
              "The first letter changes from 'i' to 'e' in the prefix.",
          },
          primaryErrorFocus:
            "Keep the first vowel as 'i' in the prefix 'ir-'.",
        },
      }),
    ),
  });

  assert.equal(result.missAnalysis.primaryErrorType, "vowel_confusion");
  assert.deepEqual(result.missAnalysis.secondaryErrorTypes, []);
  assert.equal(
    result.missAnalysis.errorTypeEvidence.vowel_confusion,
    "The first letter changes from 'i' to 'e' in the prefix.",
  );
  assert.equal(
    "letter_substitution" in result.missAnalysis.errorTypeEvidence,
    false,
  );
});

test("classifies known suffix endings gone wrong as ending confusion", async () => {
  const input: SpellingCoachInput = {
    targetWord: "requisition",
    childAttempt: "requisishun",
    childProfile: baseProfile,
    wordMetadata: {
      definition: "an official request for something needed",
      origin: "Latin",
      partOfSpeech: "noun",
      pronunciation: "rek-wuh-ZI-shun",
    },
    missSignals: {
      isCorrect: false,
      nearMiss: false,
      missingLetters: ["t", "i", "o"],
      extraLetters: ["s", "h", "u"],
      substitutedLetters: ["s for t", "h for i", "u for o"],
      transposedLetters: [],
      repeatedLetterIssue: false,
      endingConfusionFacts: [
        {
          suffix: "ition",
          attemptedEnding: "ishun",
          phoneticRewrite: true,
        },
      ],
      likelyRushed: false,
      editDistance: 4,
    },
    structuralHints: {
      syllables: ["re", "qui", "si", "tion"],
      likelyChunks: ["requis", "ition"],
      detectedPatterns: ["qu", "tion", "ition"],
      likelySuffix: "ition",
    },
    sessionContext: {
      mode: "practice",
      previousAttemptsOnThisWord: 0,
      previousMissPatterns: [],
      recentlyPracticedWords: [],
    },
  };

  const result = await runSpellingCoachAgent(input, {
    agent: createMockAgent(
      makeOutput({
        missAnalysis: {
          summary:
            "The spelling writes the ending by sound instead of using the expected suffix letters.",
          primaryErrorType: "letter_substitution",
          secondaryErrorTypes: [],
          errorTypeEvidence: {
            letter_substitution:
              "The ending letters were replaced with a sound-based spelling.",
          },
          primaryErrorFocus:
            "Keep the expected ending letters in the suffix instead of writing them by sound.",
        },
      }),
    ),
  });

  assert.equal(result.missAnalysis.primaryErrorType, "ending_confusion");
  assert.equal(
    result.missAnalysis.secondaryErrorTypes.includes("phonetic_spelling"),
    true,
  );
  assert.equal(result.missAnalysis.errorTypeEvidence.ending_confusion.length > 0, true);
});

test("classifies missing plural ending as ending confusion", async () => {
  const input = buildSpellingCoachInput({
    targetWord: "anemometers",
    childAttempt: "anemometer",
    childProfile: {
      childId: "c1",
      age: 10,
      grade: "5",
      spellingLevel: "on-grade",
    },
    sessionContext: {
      mode: "practice",
      previousAttemptsOnThisWord: 0,
      previousMissPatterns: [],
      recentlyPracticedWords: [],
    },
  });

  const result = normalizeMissAnalysisErrorTypes(
    input,
    makeOutput({
      missAnalysis: {
        summary: "The spelling is missing the final s.",
        primaryErrorType: "missing_letter",
        secondaryErrorTypes: ["likely_rushed"],
        errorTypeEvidence: {
          missing_letter: "The final letter is missing.",
          likely_rushed: "The miss is short and at the end.",
        },
        primaryErrorFocus: "Add the final s.",
        likelyWrongWordInterpretation: false,
      },
    }),
  );

  assert.equal(result.missAnalysis.primaryErrorType, "missing_letter");
  assert.equal(
    result.missAnalysis.secondaryErrorTypes.includes("pattern_rule_mismatch"),
    true,
  );
  assert.equal(
    result.missAnalysis.secondaryErrorTypes.includes("ending_confusion"),
    true,
  );
});

test("classifies missing past-tense ed ending as ending confusion", async () => {
  const input: SpellingCoachInput = {
    targetWord: "consented",
    childAttempt: "consent",
    childProfile: baseProfile,
    wordMetadata: {
      definition: "agreed or gave permission",
      origin: "Latin",
      partOfSpeech: "verb",
      pronunciation: "kuhn-SEN-tid",
    },
    missSignals: {
      isCorrect: false,
      nearMiss: true,
      missingLetters: ["e", "d"],
      extraLetters: [],
      substitutedLetters: [],
      transposedLetters: [],
      repeatedLetterIssue: false,
      endingConfusionFacts: [
        {
          suffix: "ed",
          attemptedEnding: "",
          phoneticRewrite: false,
        },
      ],
      likelyRushed: true,
      editDistance: 2,
    },
    structuralHints: {
      syllables: ["con", "sent", "ed"],
      likelyChunks: ["con", "sent", "ed"],
      detectedPatterns: ["ending -ed"],
      likelySuffix: "ed",
    },
    sessionContext: {
      mode: "practice",
      previousAttemptsOnThisWord: 0,
      previousMissPatterns: [],
      recentlyPracticedWords: [],
    },
  };

  const result = normalizeMissAnalysisErrorTypes(
    input,
    makeOutput({
      missAnalysis: {
        summary: "The spelling leaves off the ending.",
        primaryErrorType: "missing_letter",
        secondaryErrorTypes: ["likely_rushed"],
        errorTypeEvidence: {
          missing_letter: "The final letters are missing.",
          likely_rushed: "The missing letters are at the end.",
        },
        primaryErrorFocus: "Keep the final ending.",
        likelyWrongWordInterpretation: false,
      },
    }),
  );

  assert.equal(result.missAnalysis.primaryErrorType, "missing_letter");
  assert.equal(
    result.missAnalysis.secondaryErrorTypes.includes("pattern_rule_mismatch"),
    true,
  );
  assert.equal(
    result.missAnalysis.secondaryErrorTypes.includes("ending_confusion"),
    true,
  );
});

test("keeps extra_letter as a secondary signal for ending confusion with an inserted ending letter", () => {
  const input = buildSpellingCoachInput({
    targetWord: "flabbergast",
    childAttempt: "flabberghast",
    childProfile: {
      childId: "c1",
      age: 10,
      grade: "5",
      spellingLevel: "on-grade",
    },
    sessionContext: {
      mode: "practice",
      previousAttemptsOnThisWord: 0,
      previousMissPatterns: [],
      recentlyPracticedWords: [],
    },
  });

  const result = normalizeMissAnalysisErrorTypes(
    input,
    makeOutput({
      missAnalysis: {
        summary: "An extra letter was added near the ending.",
        primaryErrorType: "ending_confusion",
        secondaryErrorTypes: ["ending_confusion", "pattern_rule_mismatch"],
        errorTypeEvidence: {
          ending_confusion: "The ending ast was spelled as hast.",
          pattern_rule_mismatch: "The ending does not match the expected suffix pattern ast.",
        },
        primaryErrorFocus: "Remove the extra letter near the ending.",
        likelyWrongWordInterpretation: false,
      },
    }),
  );

  assert.equal(result.missAnalysis.primaryErrorType, "extra_letter");
  assert.equal(
    result.missAnalysis.secondaryErrorTypes.includes("ending_confusion"),
    true,
  );
  assert.equal(
    result.missAnalysis.secondaryErrorTypes.includes("pattern_rule_mismatch"),
    true,
  );
});

test("prefers vowel confusion over ending confusion for a pure ending vowel swap", () => {
  const input = buildSpellingCoachInput({
    targetWord: "concatenate",
    childAttempt: "concatinate",
    childProfile: {
      childId: "c1",
      age: 10,
      grade: "5",
      spellingLevel: "on-grade",
    },
    sessionContext: {
      mode: "practice",
      previousAttemptsOnThisWord: 0,
      previousMissPatterns: [],
      recentlyPracticedWords: [],
    },
  });

  const result = normalizeMissAnalysisErrorTypes(
    input,
    makeOutput({
      missAnalysis: {
        summary: "One vowel changed in the ending.",
        primaryErrorType: "letter_substitution",
        secondaryErrorTypes: [],
        errorTypeEvidence: {
          letter_substitution: "The ending changes one vowel.",
        },
        primaryErrorFocus: "Keep the vowel in the ending steady.",
        likelyWrongWordInterpretation: false,
      },
    }),
  );

  assert.equal(result.missAnalysis.primaryErrorType, "vowel_confusion");
  assert.equal(
    result.missAnalysis.secondaryErrorTypes.includes("ending_confusion"),
    true,
  );
  assert.equal(
    result.missAnalysis.secondaryErrorTypes.includes("pattern_rule_mismatch"),
    true,
  );
});

test("keeps ending confusion primary when the ending miss combines substitution and omission", () => {
  const input = buildSpellingCoachInput({
    targetWord: "dentifrice",
    childAttempt: "dentifris",
    childProfile: {
      childId: "c1",
      age: 10,
      grade: "5",
      spellingLevel: "on-grade",
    },
    sessionContext: {
      mode: "practice",
      previousAttemptsOnThisWord: 0,
      previousMissPatterns: [],
      recentlyPracticedWords: [],
    },
  });

  const result = normalizeMissAnalysisErrorTypes(
    input,
    makeOutput({
      missAnalysis: {
        summary: "The ending changed shape.",
        primaryErrorType: "letter_substitution",
        secondaryErrorTypes: [],
        errorTypeEvidence: {
          letter_substitution: "The ending changed shape.",
        },
        primaryErrorFocus: "Keep the ending intact.",
        likelyWrongWordInterpretation: false,
      },
    }),
  );

  assert.equal(result.missAnalysis.primaryErrorType, "ending_confusion");
  assert.equal(
    result.missAnalysis.secondaryErrorTypes.includes("phonetic_spelling"),
    true,
  );
  assert.equal(
    result.missAnalysis.secondaryErrorTypes.includes("missing_letter"),
    true,
  );
});

test("suppresses transposition for one-edit final-e drops and prefers missing letter", () => {
  const input = buildSpellingCoachInput({
    targetWord: "stratosphere",
    childAttempt: "stratospher",
    childProfile: {
      childId: "c1",
      age: 10,
      grade: "5",
      spellingLevel: "on-grade",
    },
    sessionContext: {
      mode: "practice",
      previousAttemptsOnThisWord: 0,
      previousMissPatterns: [],
      recentlyPracticedWords: [],
    },
  });

  const result = normalizeMissAnalysisErrorTypes(
    input,
    makeOutput({
      missAnalysis: {
        summary: "The final e is missing.",
        primaryErrorType: "letter_substitution",
        secondaryErrorTypes: [],
        errorTypeEvidence: {
          letter_substitution: "Fallback substitution evidence.",
        },
        primaryErrorFocus: "Keep the final letter.",
        likelyWrongWordInterpretation: false,
      },
    }),
  );

  assert.equal(result.missAnalysis.primaryErrorType, "vowel_confusion");
  assert.equal(
    result.missAnalysis.secondaryErrorTypes.includes("ending_confusion"),
    true,
  );
  assert.equal(
    result.missAnalysis.secondaryErrorTypes.includes("missing_letter"),
    true,
  );
  assert.equal(
    result.missAnalysis.secondaryErrorTypes.includes("letter_transposition"),
    false,
  );
});

test("adds phonetic spelling for tion rewritten as shun", () => {
  const input = buildSpellingCoachInput({
    targetWord: "scintillation",
    childAttempt: "scintilashun",
    childProfile: {
      childId: "c1",
      age: 10,
      grade: "5",
      spellingLevel: "on-grade",
    },
    sessionContext: {
      mode: "practice",
      previousAttemptsOnThisWord: 0,
      previousMissPatterns: [],
      recentlyPracticedWords: [],
    },
  });

  const result = normalizeMissAnalysisErrorTypes(
    input,
    makeOutput({
      missAnalysis: {
        summary: "The ending was rewritten by sound.",
        primaryErrorType: "letter_substitution",
        secondaryErrorTypes: [],
        errorTypeEvidence: {
          letter_substitution: "Fallback substitution evidence.",
        },
        primaryErrorFocus: "Keep the ending spelling steady.",
        likelyWrongWordInterpretation: false,
      },
    }),
  );

  assert.equal(result.missAnalysis.primaryErrorType, "ending_confusion");
  assert.equal(
    result.missAnalysis.secondaryErrorTypes.includes("phonetic_spelling"),
    true,
  );
  assert.equal(
    result.missAnalysis.secondaryErrorTypes.includes("double_letter_error"),
    true,
  );
});

test("suppresses extra_letter when an added doubled consonant already explains the same span", () => {
  const input = buildSpellingCoachInput({
    targetWord: "stratosphere",
    childAttempt: "sttratosphere",
    childProfile: {
      childId: "c1",
      age: 10,
      grade: "5",
      spellingLevel: "on-grade",
    },
    sessionContext: {
      mode: "practice",
      previousAttemptsOnThisWord: 0,
      previousMissPatterns: [],
      recentlyPracticedWords: [],
    },
  });

  const result = normalizeMissAnalysisErrorTypes(
    input,
    makeOutput({
      missAnalysis: {
        summary: "An extra repeated consonant was added.",
        primaryErrorType: "letter_substitution",
        secondaryErrorTypes: [],
        errorTypeEvidence: {
          letter_substitution: "Fallback substitution evidence.",
        },
        primaryErrorFocus: "Watch the beginning of the word.",
        likelyWrongWordInterpretation: false,
      },
    }),
  );

  assert.equal(result.missAnalysis.primaryErrorType, "double_letter_error");
  assert.equal(
    result.missAnalysis.secondaryErrorTypes.includes("extra_letter"),
    false,
  );
});

test("keeps ending confusion alongside other deterministic signals for fleberghost", () => {
  const input = buildSpellingCoachInput({
    targetWord: "flabbergast",
    childAttempt: "fleberghost",
    childProfile: {
      childId: "c1",
      age: 10,
      grade: "5",
      spellingLevel: "on-grade",
    },
    sessionContext: {
      mode: "practice",
      previousAttemptsOnThisWord: 0,
      previousMissPatterns: [],
      recentlyPracticedWords: [],
    },
  });

  const result = normalizeMissAnalysisErrorTypes(
    input,
    makeOutput({
      missAnalysis: {
        summary: "Several changes affect the ending and middle of the word.",
        primaryErrorType: "letter_substitution",
        secondaryErrorTypes: [],
        errorTypeEvidence: {
          letter_substitution: "Fallback substitution evidence.",
        },
        primaryErrorFocus: "Keep the ending and repeated letters steady.",
        likelyWrongWordInterpretation: false,
      },
    }),
  );

  assert.equal(result.missAnalysis.primaryErrorType, "ending_confusion");
  assert.equal(
    result.missAnalysis.secondaryErrorTypes.includes("double_letter_error"),
    true,
  );
  assert.equal(
    result.missAnalysis.secondaryErrorTypes.includes("vowel_confusion"),
    true,
  );
  assert.equal(
    result.missAnalysis.secondaryErrorTypes.length,
    2,
  );
});

test("keeps ending confusion with extra and missing letters for fabbergasted", () => {
  const input = buildSpellingCoachInput({
    targetWord: "flabbergast",
    childAttempt: "fabbergasted",
    childProfile: {
      childId: "c1",
      age: 10,
      grade: "5",
      spellingLevel: "on-grade",
    },
    sessionContext: {
      mode: "practice",
      previousAttemptsOnThisWord: 0,
      previousMissPatterns: [],
      recentlyPracticedWords: [],
    },
  });

  const result = normalizeMissAnalysisErrorTypes(
    input,
    makeOutput({
      missAnalysis: {
        summary: "The ending was expanded and the beginning lost a letter.",
        primaryErrorType: "letter_substitution",
        secondaryErrorTypes: [],
        errorTypeEvidence: {
          letter_substitution: "Fallback substitution evidence.",
        },
        primaryErrorFocus: "Watch both the ending and the missing letter.",
        likelyWrongWordInterpretation: false,
      },
    }),
  );

  assert.equal(result.missAnalysis.primaryErrorType, "ending_confusion");
  assert.equal(
    result.missAnalysis.secondaryErrorTypes.includes("extra_letter"),
    true,
  );
  assert.equal(
    result.missAnalysis.secondaryErrorTypes.includes("missing_letter"),
    true,
  );
  assert.equal(result.missAnalysis.secondaryErrorTypes.length, 2);
});

test("classifies the flabbergast miss table with stable primary and secondary signals", () => {
  const cases: Array<{
    attempt: string;
    primary: string;
    secondaries: string[];
  }> = [
    {
      attempt: "flabberghast",
      primary: "extra_letter",
      secondaries: ["ending_confusion", "pattern_rule_mismatch"],
    },
    {
      attempt: "flabergast",
      primary: "double_letter_error",
      secondaries: ["missing_letter"],
    },
    {
      attempt: "flabbergasted",
      primary: "phonetic_spelling",
      secondaries: ["ending_confusion", "extra_letter"],
    },
    {
      attempt: "flebbergast",
      primary: "vowel_confusion",
      secondaries: [],
    },
    {
      attempt: "phlabbergast",
      primary: "phonetic_spelling",
      secondaries: [],
    },
    {
      attempt: "flebergast",
      primary: "double_letter_error",
      secondaries: ["vowel_confusion", "missing_letter"],
    },
    {
      attempt: "flebberghost",
      primary: "ending_confusion",
      secondaries: ["vowel_confusion"],
    },
    {
      attempt: "fleberghost",
      primary: "ending_confusion",
      secondaries: ["double_letter_error", "vowel_confusion"],
    },
    {
      attempt: "fabbergast",
      primary: "missing_letter",
      secondaries: [],
    },
    {
      attempt: "fabberghost",
      primary: "ending_confusion",
      secondaries: ["vowel_confusion", "missing_letter"],
    },
    {
      attempt: "fabbergasted",
      primary: "ending_confusion",
      secondaries: ["extra_letter", "missing_letter"],
    },
  ];

  for (const testCase of cases) {
    const input = buildSpellingCoachInput({
      targetWord: "flabbergast",
      childAttempt: testCase.attempt,
      childProfile: {
        childId: "c1",
        age: 10,
        grade: "5",
        spellingLevel: "on-grade",
      },
      sessionContext: {
        mode: "practice",
        previousAttemptsOnThisWord: 0,
        previousMissPatterns: [],
        recentlyPracticedWords: [],
      },
    });

    const result = normalizeMissAnalysisErrorTypes(
      input,
      makeOutput({
        missAnalysis: {
          summary: "Fallback summary.",
          primaryErrorType: "letter_substitution",
          secondaryErrorTypes: [],
          errorTypeEvidence: {
            letter_substitution: "Fallback substitution evidence.",
          },
          primaryErrorFocus: "Fallback focus.",
          likelyWrongWordInterpretation: false,
        },
      }),
    );

    assert.equal(
      result.missAnalysis.primaryErrorType,
      testCase.primary,
      `unexpected primary for ${testCase.attempt}`,
    );

    for (const secondary of testCase.secondaries) {
      assert.equal(
        result.missAnalysis.secondaryErrorTypes.includes(secondary as any),
        true,
        `missing secondary ${secondary} for ${testCase.attempt}`,
      );
    }

    assert.equal(
      result.missAnalysis.secondaryErrorTypes.length,
      testCase.secondaries.length,
      `unexpected secondary count for ${testCase.attempt}`,
    );
  }
});

test("promotes high-edit-distance misses to far_from_target", () => {
  const input = buildSpellingCoachInput({
    targetWord: "strychnine",
    childAttempt: "stricma",
    childProfile: {
      childId: "c1",
      age: 10,
      grade: "5",
      spellingLevel: "on-grade",
    },
    sessionContext: {
      mode: "practice",
      previousAttemptsOnThisWord: 0,
      previousMissPatterns: [],
      recentlyPracticedWords: [],
    },
  });

  assert.equal(input.missSignals.editDistance, 6);

  const result = normalizeMissAnalysisErrorTypes(
    input,
    makeOutput({
      missAnalysis: {
        summary: "Fallback summary.",
        primaryErrorType: "letter_substitution",
        secondaryErrorTypes: [],
        errorTypeEvidence: {
          letter_substitution: "Fallback substitution evidence.",
        },
        primaryErrorFocus: "Fallback focus.",
        likelyWrongWordInterpretation: false,
      },
    }),
  );

  assert.equal(result.missAnalysis.primaryErrorType, "far_from_target");
  assert.equal(
    result.missAnalysis.errorTypeEvidence.far_from_target,
    "The spelling changed several parts of the word, so it drifted far from the target word.",
  );
});

test("promotes silent-letter misses to silent_letter_error", () => {
  const input: SpellingCoachInput = {
    targetWord: "answer",
    childAttempt: "anser",
    childProfile: baseProfile,
    wordMetadata: {
      definition: "a reply or solution",
      origin: "Old English",
      partOfSpeech: "noun",
      pronunciation: "AN-ser",
    },
    missSignals: {
      isCorrect: false,
      nearMiss: true,
      missingLetters: ["w"],
      extraLetters: [],
      substitutedLetters: [],
      transposedLetters: [],
      repeatedLetterIssue: false,
      silentLetterFactsTouched: [
        {
          text: "w",
          label: "silent w",
          reason: "The w is written but not pronounced.",
        },
      ],
      likelyRushed: false,
      editDistance: 1,
    },
    structuralHints: {
      syllables: ["ans", "wer"],
      likelyChunks: ["ans", "wer"],
      detectedPatterns: ["silent w"],
    },
    sessionContext: {
      mode: "practice",
      previousAttemptsOnThisWord: 0,
      previousMissPatterns: [],
      recentlyPracticedWords: [],
    },
  };

  const result = normalizeMissAnalysisErrorTypes(
    input,
    makeOutput({
      missAnalysis: {
        summary: "The spelling leaves out a letter in the middle of the word.",
        primaryErrorType: "missing_letter",
        secondaryErrorTypes: [],
        errorTypeEvidence: {
          missing_letter: "One written letter is missing from the middle of the word.",
        },
        primaryErrorFocus: "Keep the missing letter in the middle of the word.",
      },
    }),
  );

  assert.equal(result.missAnalysis.primaryErrorType, "silent_letter_error");
  assert.equal(
    result.missAnalysis.errorTypeEvidence.silent_letter_error,
    "The w is written but not pronounced.",
  );
  assert.equal(
    "missing_letter" in result.missAnalysis.errorTypeEvidence,
    true,
  );
});

test("promotes sound-based tricky-part misses to phonetic_spelling", () => {
  const input: SpellingCoachInput = {
    targetWord: "phlox",
    childAttempt: "flox",
    childProfile: baseProfile,
    wordMetadata: {
      definition: "a flowering plant with clustered blooms",
      origin: "Greek",
      partOfSpeech: "noun",
      pronunciation: "floks",
    },
    missSignals: {
      isCorrect: false,
      nearMiss: true,
      missingLetters: ["ph"],
      extraLetters: [],
      substitutedLetters: ["f for ph"],
      transposedLetters: [],
      repeatedLetterIssue: false,
      trickyPartFactsTouched: [
        {
          text: "ph",
          label: "ph says f",
          reason: "The ph is spelled ph but sounds like f here.",
          sounds_like: "f",
          phoneticRewrite: true,
        },
      ],
      likelyRushed: false,
      editDistance: 1,
    },
    structuralHints: {
      syllables: ["phlox"],
      likelyChunks: ["ph", "lox"],
      detectedPatterns: ["ph says f"],
    },
    sessionContext: {
      mode: "practice",
      previousAttemptsOnThisWord: 0,
      previousMissPatterns: [],
      recentlyPracticedWords: [],
    },
  };

  const result = normalizeMissAnalysisErrorTypes(
    input,
    makeOutput({
      missAnalysis: {
        summary: "The spelling simplifies the opening letters.",
        primaryErrorType: "missing_letter",
        secondaryErrorTypes: [],
        errorTypeEvidence: {
          missing_letter: "Letters from the opening chunk are missing.",
        },
        primaryErrorFocus: "Keep the full opening chunk.",
      },
    }),
  );

  assert.equal(result.missAnalysis.primaryErrorType, "phonetic_spelling");
  assert.equal(
    result.missAnalysis.errorTypeEvidence.phonetic_spelling,
    "The ph is spelled ph but sounds like f here.",
  );
  assert.equal(
    result.missAnalysis.secondaryErrorTypes.includes("pattern_rule_mismatch"),
    true,
  );
});

test("promotes tricky spelling-pattern misses to pattern_rule_mismatch", () => {
  const input: SpellingCoachInput = {
    targetWord: "made",
    childAttempt: "mad",
    childProfile: baseProfile,
    wordMetadata: {
      definition: "created or finished",
      origin: "Old English",
      partOfSpeech: "verb",
      pronunciation: "mayd",
    },
    missSignals: {
      isCorrect: false,
      nearMiss: true,
      missingLetters: ["e"],
      extraLetters: [],
      substitutedLetters: [],
      transposedLetters: [],
      repeatedLetterIssue: false,
      trickyPartFactsTouched: [
        {
          text: "e",
          label: "final e pattern",
          reason: "The final e helps the a say its name.",
          sounds_like: undefined,
          phoneticRewrite: false,
        },
      ],
      likelyRushed: false,
      editDistance: 1,
    },
    structuralHints: {
      syllables: ["made"],
      likelyChunks: ["mad", "e"],
      detectedPatterns: ["silent e"],
    },
    sessionContext: {
      mode: "practice",
      previousAttemptsOnThisWord: 0,
      previousMissPatterns: [],
      recentlyPracticedWords: [],
    },
  };

  const result = normalizeMissAnalysisErrorTypes(
    input,
    makeOutput({
      missAnalysis: {
        summary: "The spelling drops the last letter.",
        primaryErrorType: "missing_letter",
        secondaryErrorTypes: [],
        errorTypeEvidence: {
          missing_letter: "The last written letter is missing.",
        },
        primaryErrorFocus: "Keep the last letter.",
      },
    }),
  );

  assert.equal(result.missAnalysis.primaryErrorType, "missing_letter");
  assert.equal(
    result.missAnalysis.errorTypeEvidence.pattern_rule_mismatch,
    "The final e helps the a say its name.",
  );
  assert.equal(
    result.missAnalysis.secondaryErrorTypes.includes("pattern_rule_mismatch"),
    true,
  );
});

test("keeps likelyWrongWordInterpretation under model control during normalization", () => {
  const input: SpellingCoachInput = {
    targetWord: "consternation",
    childAttempt: "constellation",
    childProfile: baseProfile,
    wordMetadata: {
      definition: "sudden confusion or dismay",
      origin: "Latin",
      partOfSpeech: "noun",
      pronunciation: "kon-ster-NAY-shun",
    },
    missSignals: {
      isCorrect: false,
      nearMiss: false,
      missingLetters: ["r", "n"],
      extraLetters: ["l", "l"],
      substitutedLetters: ["l for r", "l for n"],
      transposedLetters: [],
      repeatedLetterIssue: true,
      wrongWordInterpretationHints: {
        targetNormalized: "consternation",
        attemptNormalized: "constellation",
        sharedPrefixLength: 6,
        sharedSuffixLength: 5,
        longestCommonSubsequenceLength: 10,
        longestCommonSubsequenceRatio: 0.769,
        bigramOverlapRatio: 0.636,
        trigramOverlapRatio: 0.5,
        substantialStructuralOverlap: true,
      },
      deterministicLikelyWrongWordInterpretation: true,
      likelyRushed: false,
      editDistance: 4,
    },
    structuralHints: {
      syllables: ["con", "ster", "na", "tion"],
      likelyChunks: ["con", "ster", "nation"],
      detectedPatterns: ["tion"],
    },
    sessionContext: {
      mode: "practice",
      previousAttemptsOnThisWord: 0,
      previousMissPatterns: [],
      recentlyPracticedWords: [],
    },
  };

  const result = normalizeMissAnalysisErrorTypes(
    input,
    makeOutput({
      missAnalysis: {
        summary: "The middle of the word changed to a different familiar-looking form.",
        primaryErrorType: "letter_substitution",
        secondaryErrorTypes: [],
        errorTypeEvidence: {
          letter_substitution:
            "Several middle letters changed and the attempt now reads like a different familiar-looking word-form.",
        },
        primaryErrorFocus: "Keep the middle chunk stern instead of changing it.",
        likelyWrongWordInterpretation: false,
      },
    }),
  );

  assert.equal(result.missAnalysis.likelyWrongWordInterpretation, false);
});

test("appends wrong_word_interpretation as an extra secondary when the model flags it", () => {
  const input: SpellingCoachInput = {
    targetWord: "consternation",
    childAttempt: "constellation",
    childProfile: baseProfile,
    wordMetadata: {
      definition: "sudden confusion or dismay",
      origin: "Latin",
      partOfSpeech: "noun",
      pronunciation: "kon-ster-NAY-shun",
    },
    missSignals: {
      isCorrect: false,
      nearMiss: false,
      missingLetters: ["r", "n"],
      extraLetters: ["l", "l"],
      substitutedLetters: ["l for r", "l for n"],
      transposedLetters: [],
      repeatedLetterIssue: true,
      likelyRushed: false,
      editDistance: 4,
    },
    structuralHints: {
      syllables: ["con", "ster", "na", "tion"],
      likelyChunks: ["con", "ster", "nation"],
      detectedPatterns: ["tion"],
    },
    sessionContext: {
      mode: "practice",
      previousAttemptsOnThisWord: 0,
      previousMissPatterns: [],
      recentlyPracticedWords: [],
    },
  };

  const result = normalizeMissAnalysisErrorTypes(
    input,
    makeOutput({
      missAnalysis: {
        summary: "The middle of the word changed to another familiar-looking form.",
        primaryErrorType: "chunk_mismatch",
        secondaryErrorTypes: ["phonetic_spelling", "pattern_rule_mismatch"],
        errorTypeEvidence: {
          chunk_mismatch:
            "The middle chunk changed to a different familiar-looking form.",
          phonetic_spelling:
            "Part of the attempt follows a sound-based rewrite.",
          pattern_rule_mismatch:
            "The expected spelling pattern in the middle does not match.",
        },
        primaryErrorFocus: "Keep the original middle chunk.",
        likelyWrongWordInterpretation: true,
      },
    }),
  );

  assert.equal(result.missAnalysis.primaryErrorType, "extra_letter");
  assert.equal(
    result.missAnalysis.secondaryErrorTypes.includes("wrong_word_interpretation"),
    true,
  );
  assert.equal(
    result.missAnalysis.errorTypeEvidence.wrong_word_interpretation,
    "The attempt reads as another real English word.",
  );
});

test("treats manufacture -> manufakcher as ending confusion with phonetic rewrite support", () => {
  const input = buildSpellingCoachInput({
    targetWord: "manufacture",
    childAttempt: "manufakcher",
    childProfile: {
      childId: "c1",
      age: 10,
      grade: "5",
      spellingLevel: "on-grade",
    },
    sessionContext: {
      mode: "practice",
      previousAttemptsOnThisWord: 0,
      previousMissPatterns: [],
      recentlyPracticedWords: [],
    },
  });

  const result = normalizeMissAnalysisErrorTypes(
    input,
    makeOutput({
      missAnalysis: {
        summary: "The spelling changed letters in the middle and swapped the last two letters.",
        primaryErrorType: "letter_transposition",
        secondaryErrorTypes: [],
        errorTypeEvidence: {
          letter_transposition:
            "The ending 'er' is transposed but still present, showing a letter transposition error.",
        },
        primaryErrorFocus: "Keep the middle and ending letters in the right order.",
        likelyWrongWordInterpretation: false,
      },
    }),
  );

  assert.equal(result.missAnalysis.primaryErrorType, "ending_confusion");
  assert.equal(
    result.missAnalysis.secondaryErrorTypes.includes("phonetic_spelling"),
    true,
  );
  assert.equal(result.missAnalysis.secondaryErrorTypes.length, 1);
  assert.equal(result.missAnalysis.likelyWrongWordInterpretation, false);
});

test("sanitizes miss analysis wording and strips internal diagnostic tokens", async () => {
  const input: SpellingCoachInput = {
    targetWord: "materialize",
    childAttempt: "materialise",
    childProfile: baseProfile,
    wordMetadata: {
      definition: "to make something real or actual",
      origin: "Latin",
      partOfSpeech: "verb",
      pronunciation: "muh-TEER-ee-uh-lize",
    },
    missSignals: {
      isCorrect: false,
      nearMiss: true,
      missingLetters: [],
      extraLetters: [],
      substitutedLetters: ["s for z"],
      transposedLetters: [],
      repeatedLetterIssue: false,
      likelyRushed: false,
      editDistance: 1,
    },
    structuralHints: {
      syllables: ["ma", "te", "ri", "al", "ize"],
      likelyChunks: ["materi", "alize"],
      detectedPatterns: ["suffix -ize"],
    },
    sessionContext: {
      mode: "practice",
      previousAttemptsOnThisWord: 0,
      previousMissPatterns: [],
      recentlyPracticedWords: [],
    },
  };

  const result = await runSpellingCoachAgent(input, {
    agent: createMockAgent(
      makeOutput({
        missAnalysis: {
          summary:
            "The child substituted 's' for the correct 'z' in the suffix '-ize'.",
          primaryErrorType: "letter_substitution",
          secondaryErrorTypes: [],
          errorTypeEvidence: {
            letter_substitution:
              "rawSignals.substitutedLetters includes 's for z', showing the child substituted the ending letter. repeatedLetterIssue is false.",
          },
          primaryErrorFocus:
            "The child wrote 's' instead of 'z' in the ending '-ize'.",
        },
      }),
    ),
  });

  assert.equal(
    result.missAnalysis.summary,
    "The spelling substituted 's' for the correct 'z' in the suffix '-ize'.",
  );
  assert.equal(
    result.missAnalysis.primaryErrorFocus,
    "The spelling uses 's' instead of 'z' in the ending '-ize'.",
  );
  assert.equal(
    result.missAnalysis.errorTypeEvidence.letter_substitution.includes(
      "rawSignals",
    ),
    false,
  );
  assert.equal(
    result.missAnalysis.errorTypeEvidence.letter_substitution.includes(
      "repeatedLetterIssue",
    ),
    false,
  );
  assert.equal(
    result.missAnalysis.errorTypeEvidence.letter_substitution.includes(
      "The child",
    ),
    false,
  );
});

test("sanitizes coaching full explanation wording to avoid 'the child wrote'", async () => {
  const input: SpellingCoachInput = {
    targetWord: "diaphanous",
    childAttempt: "diaphonous",
    childProfile: baseProfile,
    wordMetadata: {
      definition: "light, delicate, and almost transparent.",
      origin: "Greek",
      partOfSpeech: "adjective",
      pronunciation: "dy-AF-uh-nus",
    },
    missSignals: {
      isCorrect: false,
      nearMiss: true,
      missingLetters: ["a"],
      extraLetters: ["o"],
      substitutedLetters: ["o for a"],
      transposedLetters: [],
      repeatedLetterIssue: false,
      likelyRushed: false,
      editDistance: 1,
    },
    structuralHints: {
      syllables: ["di", "aph", "a", "nous"],
      likelyChunks: ["dia", "phan", "ous"],
      detectedPatterns: ["ph says f"],
    },
    sessionContext: {
      mode: "practice",
      previousAttemptsOnThisWord: 0,
      previousMissPatterns: [],
      recentlyPracticedWords: [],
    },
  };

  const result = await runSpellingCoachAgent(input, {
    agent: createMockAgent(
      makeOutput({
        missAnalysis: {
          summary: "The spelling changed the middle vowel in the root chunk.",
          primaryErrorType: "letter_substitution",
          secondaryErrorTypes: [],
          errorTypeEvidence: {
            letter_substitution:
              "The middle vowel was changed in the root chunk.",
          },
          primaryErrorFocus: "Keep the root chunk 'phan' instead of 'phon'.",
        },
        coachingText: {
          shortFeedback: "Try again.",
          memoryTip: "Remember the phan chunk.",
          sayAloudTip: "Say di-a-phan-ous.",
          fullExplanation:
            "The tricky part here is the root chunk 'phan' which uses the 'ph' digraph to make the /f/ sound. The child wrote 'phon' instead, which is a common sound but not correct in this word. Remembering the 'ph' pattern and the chunk 'phan' will help you spell 'diaphanous' correctly.",
        },
      }),
    ),
  });

  assert.equal(
    result.coachingText.fullExplanation.includes("The child wrote"),
    false,
  );
  assert.equal(
    result.coachingText.fullExplanation.toLowerCase().includes("child"),
    false,
  );
});

test("sanitizes plain-English raw signals leakage from miss analysis text", async () => {
  const input: SpellingCoachInput = {
    targetWord: "filament",
    childAttempt: "filiment",
    childProfile: baseProfile,
    wordMetadata: {
      definition: "a slender threadlike object or fiber",
      origin: "Latin",
      partOfSpeech: "noun",
      pronunciation: "FIH-luh-ment",
    },
    missSignals: {
      isCorrect: false,
      nearMiss: true,
      missingLetters: ["a"],
      extraLetters: [],
      substitutedLetters: [],
      transposedLetters: [],
      repeatedLetterIssue: false,
      likelyRushed: true,
      editDistance: 1,
    },
    structuralHints: {
      syllables: ["fil", "a", "ment"],
      likelyChunks: ["fil", "iament"],
      detectedPatterns: [],
    },
    sessionContext: {
      mode: "practice",
      previousAttemptsOnThisWord: 0,
      previousMissPatterns: [],
      recentlyPracticedWords: [],
    },
  };

  const result = await runSpellingCoachAgent(input, {
    agent: createMockAgent(
      makeOutput({
        missAnalysis: {
          summary:
            "The spelling is missing the letter 'a' after the 'l' in the middle part of the word.",
          primaryErrorType: "missing_letter",
          secondaryErrorTypes: ["likely_rushed"],
          errorTypeEvidence: {
            missing_letter:
              "The raw signals show one missing letter 'a' in the middle chunk 'liament' which was spelled as 'liment'.",
            likely_rushed:
              "The raw signals indicate the attempt was likely rushed, which often causes small omissions like this.",
          },
          primaryErrorFocus:
            "missing the letter 'a' in the middle chunk 'liament'",
        },
      }),
    ),
  });

  assert.equal(
    result.missAnalysis.errorTypeEvidence.missing_letter.includes(
      "raw signals",
    ),
    false,
  );
  assert.equal(
    result.missAnalysis.errorTypeEvidence.likely_rushed.includes(
      "raw signals",
    ),
    false,
  );
  assert.equal(
    result.missAnalysis.errorTypeEvidence.likely_rushed.startsWith("The attempt"),
    true,
  );
});

test("builds spelling rule hints text from the spelling-rules csv", () => {
  const hints = buildSpellingRuleHintsText(5);

  assert.equal(hints.startsWith("- "), true);
  assert.equal(hints.includes("matcher_scope="), true);
  assert.equal(hints.includes("pattern_role="), true);
  assert.equal(hints.includes("pattern_match_type="), true);
  assert.equal(hints.includes("pattern="), true);
  assert.equal(hints.includes("Applies when:"), true);
});

test("shortlists literal spelling rules when the feature flag is enabled", () => {
  const originalFlag = process.env.SPELLING_COACH_RULE_SHORTLIST;
  process.env.SPELLING_COACH_RULE_SHORTLIST = "on";

  try {
    const hints = buildSpellingRuleHintsText(24, "molecule");

    assert.equal(hints.includes("soft_c_before_e_i_y"), false);
    assert.equal(hints.includes("c_before_a_o_u_l_r"), true);
  } finally {
    if (originalFlag === undefined) {
      delete process.env.SPELLING_COACH_RULE_SHORTLIST;
    } else {
      process.env.SPELLING_COACH_RULE_SHORTLIST = originalFlag;
    }
  }
});

test("word-level precompute prompt omits curated spelling-rule guidance by default", () => {
  const input = buildWordPrecomputeInput("torsion");
  const prompt = buildWordTeachingPrecomputePrompt(input);

  assert.equal(
    prompt.includes(
      "Use the curated spelling-rules CSV as a reference list of common spelling rules and rule labels.",
    ),
    false,
  );
  assert.equal(prompt.includes("Curated spelling-rule hints:"), false);
});

test("word-level precompute prompt includes curated spelling-rule guidance when enabled", () => {
  const originalFlag = process.env.SPELLING_COACH_RULE_PROMPT_HINTS;
  process.env.SPELLING_COACH_RULE_PROMPT_HINTS = "on";

  try {
    const input = buildWordPrecomputeInput("torsion");
    const prompt = buildWordTeachingPrecomputePrompt(input);

    assert.equal(
      prompt.includes(
        "Use the curated spelling-rules CSV as a reference list of common spelling rules and rule labels.",
      ),
      true,
    );
    assert.equal(prompt.includes("Curated spelling-rule hints:"), true);
    assert.equal(prompt.includes("soft_g_before_e_i_y"), true);
    assert.equal(
      prompt.includes(
        "Use phonetic spelling or simple sound-by-syllable reasoning internally to check whether a sound-based rule truly matches the word.",
      ),
      true,
    );
    assert.equal(
      prompt.includes(
        "Use the curated spelling-rules CSV to identify meaningful pattern labels only when they help conceptTeaching or conceptLabels.",
      ),
      true,
    );
  } finally {
    if (originalFlag === undefined) {
      delete process.env.SPELLING_COACH_RULE_PROMPT_HINTS;
    } else {
      process.env.SPELLING_COACH_RULE_PROMPT_HINTS = originalFlag;
    }
  }
});

test("word-level precompute prompt separates spelling chunks from concept grouping", () => {
  const prompt = buildWordTeachingPrecomputePrompt({
    targetWord: "center",
    childAttempt: "center",
    childProfile: baseProfile,
    wordMetadata: {
      definition: "the middle point",
      origin: "Latin",
      partOfSpeech: "noun",
      exampleSentence: "Stand in the center of the circle.",
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

  assert.equal(
    prompt.includes("choose spelling-teaching chunks that are easy to say, easy to remember"),
    true,
  );
  assert.equal(
    prompt.includes("explain that separately in conceptTeaching instead of forcing wordBreakdown.displayChunks to match it"),
    true,
  );
  assert.equal(
    prompt.includes("Do not use generic filler such as 'easy to say and remember' by itself."),
    true,
  );
  assert.equal(
    prompt.includes("wordBreakdown.chunkReason must mention the actual chunk boundary, ending, blend, digraph, or spelling pattern"),
    true,
  );
});

test("warms word teaching precompute on a word-only input", async () => {
  const input = buildWordPrecomputeInput("torsion");
  const storedTorsionBreakdown = getStoredWordBreakdown("torsion");
  assert.equal(
    hasWordTeachingPrecompute(input, { runtime: "direct" }),
    Boolean(storedTorsionBreakdown),
  );

  const result = await warmWordTeachingPrecompute(input, {
    runtime: "direct",
    directModel: {
      async invoke() {
        throw new Error("Runtime concept precompute should be disabled by default.");
      },
    },
  });

  assert.deepEqual(
    result.wordBreakdown.displayChunks,
    storedTorsionBreakdown?.displayChunks ?? ["tor", "sion"],
  );
  assert.equal(
    result.wordTeaching.conceptTeaching.summary,
    getStoredWordTeachingOnlyPrecompute("torsion")?.wordTeaching.conceptTeaching
      .summary ?? "",
  );
  assert.deepEqual(
    result.conceptLabels.originLabels,
    getStoredWordTeachingOnlyPrecompute("torsion")?.conceptLabels.originLabels ??
      [],
  );
  assert.equal(hasWordTeachingPrecompute(input, { runtime: "direct" }), true);
});

test("merges cached word teaching with miss-only analysis on submit", async () => {
  const originalRuntimeConceptTeaching =
    process.env.SPELLING_COACH_RUNTIME_CONCEPT_TEACHING;
  process.env.SPELLING_COACH_RUNTIME_CONCEPT_TEACHING = "on";

  try {
  const submitInput: SpellingCoachInput = {
    targetWord: "torsion",
    childAttempt: "torshun",
    childProfile: baseProfile,
    wordMetadata: {
      definition: "The act of twisting something.",
      origin: "Latin",
      partOfSpeech: "noun",
      exampleSentence: "The gymnast showed torsion by twisting her body in the air.",
    },
    missSignals: {
      isCorrect: false,
      nearMiss: false,
      missingLetters: ["i", "o"],
      extraLetters: ["h", "u"],
      substitutedLetters: [],
      transposedLetters: [],
      repeatedLetterIssue: false,
      likelyRushed: false,
      editDistance: 4,
    },
    structuralHints: {
      syllables: [],
      likelyChunks: ["tor", "sion"],
      detectedPatterns: ["sion"],
      likelySuffix: "sion",
    },
    sessionContext: {
      mode: "practice",
      previousAttemptsOnThisWord: 0,
      previousMissPatterns: [],
      recentlyPracticedWords: [],
    },
  };

  const precomputeOutput = {
    wordTeaching: {
      conceptTeaching: {
        summary: "The concept centers on twisting.",
        meaningFocus: "twisting",
        originFocus: "Latin-derived",
        morphologyFocus: "",
        originLabels: ["latin-derived"],
        morphologyLabels: [],
      },
    },
    wordBreakdown: {
      displayChunks: ["tor", "sion"],
      chunkReason: "The ending chunk carries the main pattern.",
    },
    conceptLabels: {
      originLabels: ["latin-derived"],
      patternLabels: ["sion"],
      morphologyLabels: [],
    },
  };

  const missOnlyOutput = {
    correctness: {
      isCorrect: false,
      reinforceSuccess: false,
    },
    missAnalysis: {
      summary: "The ending was rewritten phonetically as shun.",
      primaryErrorType: "phonetic_spelling",
      secondaryErrorTypes: ["ending_confusion"],
      errorTypeEvidence: {
        phonetic_spelling: "The ending was rewritten by sound rather than by standard spelling.",
        ending_confusion: "The word ending does not match the expected spelling pattern.",
      },
      primaryErrorFocus: "Use the -sion spelling instead of writing shun by sound.",
      likelyWrongWordInterpretation: false,
      usedMeaningDisambiguationWell: false,
    },
    errorRelevance: {
      mostRelevantToError: "form",
      confidence: 0.9,
      reason: "The miss is centered on the -sion ending pattern.",
    },
    teachingDecision: {
      strategy: "pattern",
      primaryFocus: "Keep the -sion ending.",
      secondaryFocuses: ["Chunk it as tor + sion"],
      confidence: 0.88,
      rationale: "The word-level chunking is already known, and the miss is specifically about the ending pattern.",
    },
    coachingText: {
      shortFeedback: "You heard the ending, but wrote it by sound.",
      fullExplanation: "Torsion ends with -sion, not shun. Use the chunk tor + sion to hold the ending in place.",
      memoryTip: "See the word as tor + sion.",
      sayAloudTip: "Say tor-sion and hold the sion ending.",
    },
    nextStep: {
      practiceFocus: "Practice words that end in -sion.",
      shouldReviewSoon: true,
      suggestedSimilarWordTypes: ["-sion words"],
    },
  };

  const storedTorsionBreakdownForSubmit = getStoredWordBreakdown("torsion");
  const firstModelResponse = storedTorsionBreakdownForSubmit
    ? {
        wordTeaching: {
          conceptTeaching: precomputeOutput.wordTeaching.conceptTeaching,
        },
        conceptLabels: precomputeOutput.conceptLabels,
      }
    : precomputeOutput;

  const result = await runSplitSpellingCoachAgent(submitInput, {
    runtime: "direct",
    directModel: createSequenceMockModel([
      JSON.stringify(firstModelResponse),
      JSON.stringify(missOnlyOutput),
    ]),
  });

  assert.equal(
    result.coachingText.sayAloudTip,
    "Say it slowly: TAWR-shuhn",
  );
  assert.deepEqual(result.nextStep, {
    practiceFocus: "",
    shouldReviewSoon: false,
    suggestedSimilarWordTypes: [],
  });
  assert.equal(result.missAnalysis.primaryErrorFocus.includes("-sion"), true);
  assert.equal(result.errorRelevance.mostRelevantToError, "form");
  } finally {
    if (originalRuntimeConceptTeaching === undefined) {
      delete process.env.SPELLING_COACH_RUNTIME_CONCEPT_TEACHING;
    } else {
      process.env.SPELLING_COACH_RUNTIME_CONCEPT_TEACHING =
        originalRuntimeConceptTeaching;
    }
  }
});

test("sanitizes miss analysis text in the split runtime path", async () => {
  const submitInput: SpellingCoachInput = {
    targetWord: "altercation",
    childAttempt: "altrecation",
    childProfile: baseProfile,
    wordMetadata: {
      definition: "A noisy argument or fight between people.",
      origin: "Latin",
      partOfSpeech: "noun",
      exampleSentence: "The kids had an altercation over the last toy.",
    },
    missSignals: {
      isCorrect: false,
      nearMiss: true,
      missingLetters: [],
      extraLetters: [],
      substitutedLetters: [],
      transposedLetters: ["re"],
      repeatedLetterIssue: false,
      likelyRushed: true,
      editDistance: 2,
    },
    structuralHints: {
      syllables: [],
      likelyChunks: ["alter", "cation"],
      detectedPatterns: ["tion"],
      likelySuffix: "ation",
    },
    sessionContext: {
      mode: "practice",
      previousAttemptsOnThisWord: 0,
      previousMissPatterns: [],
      recentlyPracticedWords: [],
    },
  };

  const precomputeOutput = await warmWordTeachingPrecompute(submitInput, {
    runtime: "direct",
    directModel: createSequenceMockModel([
      JSON.stringify({
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
        conceptLabels: {
          originLabels: [],
          patternLabels: [],
          morphologyLabels: [],
        },
      }),
    ]),
  });

  const missOnlyOutput = {
    correctness: {
      isCorrect: false,
      reinforceSuccess: false,
    },
    missAnalysis: {
      summary:
        "The child substituted the order of 'r' and 'e' in the middle part of the word.",
      primaryErrorType: "letter_substitution",
      secondaryErrorTypes: ["likely_rushed"],
      errorTypeEvidence: {
        letter_substitution:
          "The child substituted the order of 'r' and 'e' in the middle of the word.",
        likely_rushed:
          "The raw signals indicate the attempt was likely rushed, which can cause letter order mistakes.",
      },
      primaryErrorFocus:
        "The child wrote the letters 'r' and 'e' in the wrong order.",
      likelyWrongWordInterpretation: false,
      usedMeaningDisambiguationWell: false,
    },
    errorRelevance: {
      mostRelevantToError: "form",
      confidence: 0.9,
      reason: "The miss is centered on letter order.",
    },
    teachingDecision: {
      strategy: "pattern",
      primaryFocus: "Letter order",
      secondaryFocuses: [],
      confidence: 0.85,
      rationale: "The main miss is a letter-order issue.",
    },
    coachingText: {
      shortFeedback: "Watch the order of the middle letters.",
      fullExplanation: "",
      memoryTip: "",
      sayAloudTip: "",
    },
    nextStep: {
      practiceFocus: "",
      shouldReviewSoon: false,
      suggestedSimilarWordTypes: [],
    },
  };

  const result = await runSplitSpellingCoachAgent(submitInput, {
    runtime: "direct",
    directModel: createSequenceMockModel([
      JSON.stringify({
        wordTeaching: {
          conceptTeaching: precomputeOutput.wordTeaching.conceptTeaching,
        },
        conceptLabels: precomputeOutput.conceptLabels,
      }),
      JSON.stringify(missOnlyOutput),
    ]),
  });

  assert.equal(result.missAnalysis.primaryErrorType, "letter_transposition");
  assert.equal(
    result.missAnalysis.errorTypeEvidence.likely_rushed.includes("raw signals"),
    false,
  );
  assert.equal(
    result.missAnalysis.errorTypeEvidence.letter_transposition.includes(
      "The child",
    ),
    false,
  );
  assert.equal(
    result.missAnalysis.primaryErrorFocus.includes("The child"),
    false,
  );
});

test("sanitizes schema-style evidence names and edit-distance phrasing from miss analysis text", async () => {
  const input: SpellingCoachInput = {
    targetWord: "background",
    childAttempt: "bakground",
    childProfile: baseProfile,
    wordMetadata: {
      definition: "the area or scenery behind the main thing",
      origin: "English",
      partOfSpeech: "noun",
      pronunciation: "BAK-ground",
    },
    missSignals: {
      isCorrect: false,
      nearMiss: true,
      missingLetters: ["c"],
      extraLetters: [],
      substitutedLetters: [],
      transposedLetters: [],
      repeatedLetterIssue: false,
      likelyRushed: true,
      editDistance: 1,
    },
    structuralHints: {
      syllables: ["back", "ground"],
      likelyChunks: ["back", "ground"],
      detectedPatterns: ["consonant cluster ck"],
    },
    sessionContext: {
      mode: "practice",
      previousAttemptsOnThisWord: 0,
      previousMissPatterns: [],
      recentlyPracticedWords: [],
    },
  };

  const result = await runSpellingCoachAgent(input, {
    agent: createMockAgent(
      makeOutput({
        missAnalysis: {
          summary:
            "The spelling missed the letter 'c' in the first chunk 'back', writing 'bak' instead.",
          primaryErrorType: "missing_letter",
          secondaryErrorTypes: ["likely_rushed"],
          errorTypeEvidence: {
            missing_letter:
              "The letter 'c' is missing from the expected chunk 'back', as shown by the missingLetters signal and chunkMismatchFacts.",
            likely_rushed:
              "The edit distance is 1 with a missing letter, and the rawSignals indicate likelyRushed is true.",
          },
          primaryErrorFocus: "Missing the letter 'c' in the first chunk 'back'",
        },
      }),
    ),
  });

  assert.equal(
    result.missAnalysis.errorTypeEvidence.missing_letter.includes(
      "missingLetters",
    ),
    false,
  );
  assert.equal(
    result.missAnalysis.errorTypeEvidence.missing_letter.includes(
      "chunkMismatchFacts",
    ),
    false,
  );
  assert.equal(
    result.missAnalysis.errorTypeEvidence.likely_rushed.includes(
      "edit distance",
    ),
    false,
  );
  assert.equal(
    result.missAnalysis.errorTypeEvidence.likely_rushed.includes(
      "likelyRushed",
    ),
    false,
  );
  assert.equal(
    result.missAnalysis.errorTypeEvidence.likely_rushed.includes("rawSignals"),
    false,
  );
});

test("sanitizes vowel-substitution signal names from miss analysis evidence", async () => {
  const input: SpellingCoachInput = {
    targetWord: "serenade",
    childAttempt: "sarinade",
    childProfile: baseProfile,
    wordMetadata: {
      definition: "a song performed in someone's honor",
      origin: "Italian",
      partOfSpeech: "noun",
      pronunciation: "SEH-ruh-nayd",
    },
    missSignals: {
      isCorrect: false,
      nearMiss: true,
      missingLetters: [],
      extraLetters: [],
      substitutedLetters: ["a for e"],
      transposedLetters: [],
      repeatedLetterIssue: false,
      likelyRushed: false,
      editDistance: 1,
    },
    structuralHints: {
      syllables: ["ser", "e", "nade"],
      likelyChunks: ["ser", "enade"],
      detectedPatterns: ["final silent e"],
    },
    sessionContext: {
      mode: "practice",
      previousAttemptsOnThisWord: 0,
      previousMissPatterns: [],
      recentlyPracticedWords: [],
    },
  };

  const result = await runSpellingCoachAgent(input, {
    agent: createMockAgent(
      makeOutput({
        missAnalysis: {
          summary:
            "The spelling used 'a' instead of 'e' in the vowel part of the word.",
          primaryErrorType: "vowel_confusion",
          secondaryErrorTypes: [],
          errorTypeEvidence: {
            vowel_confusion:
              "The attempt substituted 'a' for the expected 'e' in the vowel position, as shown by the vowelSubstitutionPairs signal.",
          },
          primaryErrorFocus: "Using 'a' instead of 'e' in the vowel part",
        },
      }),
    ),
  });

  assert.equal(
    result.missAnalysis.errorTypeEvidence.vowel_confusion.includes(
      "vowelSubstitutionPairs",
    ),
    false,
  );
  assert.equal(
    result.missAnalysis.errorTypeEvidence.vowel_confusion.includes("signal"),
    false,
  );
});

test("clears runtime concept teaching in the full response path when the feature flag is off", async () => {
  const input: SpellingCoachInput = {
    targetWord: "torsion",
    childAttempt: "torshun",
    childProfile: baseProfile,
    wordMetadata: {
      definition: "The act of twisting something.",
      origin: "Latin",
      partOfSpeech: "noun",
      exampleSentence: "The gymnast showed torsion by twisting her body in the air.",
    },
    missSignals: {
      isCorrect: false,
      nearMiss: false,
      missingLetters: ["i", "o"],
      extraLetters: ["h", "u"],
      substitutedLetters: [],
      transposedLetters: [],
      repeatedLetterIssue: false,
      likelyRushed: false,
      editDistance: 4,
    },
    structuralHints: {
      syllables: [],
      likelyChunks: ["tor", "sion"],
      detectedPatterns: ["sion"],
      likelySuffix: "sion",
    },
    sessionContext: {
      mode: "practice",
      previousAttemptsOnThisWord: 0,
      previousMissPatterns: [],
      recentlyPracticedWords: [],
    },
  };

  const result = await runSpellingCoachAgent(input, {
    runtime: "direct",
    directModel: createDirectMockModel({
      correctness: {
        isCorrect: false,
        reinforceSuccess: false,
      },
      missAnalysis: {
        summary: "The ending was rewritten phonetically as shun.",
        primaryErrorType: "phonetic_spelling",
        secondaryErrorTypes: ["ending_confusion"],
        errorTypeEvidence: {
          phonetic_spelling:
            "The ending was rewritten by sound rather than by standard spelling.",
          ending_confusion:
            "The word ending does not match the expected spelling pattern.",
        },
        primaryErrorFocus:
          "Use the -sion spelling instead of writing shun by sound.",
        likelyWrongWordInterpretation: false,
        usedMeaningDisambiguationWell: false,
      },
      wordTeaching: {
        conceptTeaching: {
          summary: "The concept centers on twisting.",
          meaningFocus: "twisting",
          originFocus: "Latin-derived",
          morphologyFocus: "",
          originLabels: ["latin-derived"],
          morphologyLabels: [],
        },
      },
      errorRelevance: {
        mostRelevantToError: "form",
        confidence: 0.9,
        reason: "The miss is centered on the -sion ending pattern.",
      },
      teachingDecision: {
        strategy: "pattern",
        primaryFocus: "Keep the -sion ending.",
        secondaryFocuses: ["Chunk it as tor + sion"],
        confidence: 0.88,
        rationale:
          "The miss is specifically about the ending pattern.",
      },
      coachingText: {
        shortFeedback: "You heard the ending, but wrote it by sound.",
        fullExplanation:
          "Torsion ends with -sion, not shun. Use the chunk tor + sion to hold the ending in place.",
        memoryTip: "See the word as tor + sion.",
        sayAloudTip: "Say tor-sion and hold the sion ending.",
      },
      wordBreakdown: {
        displayChunks: ["tor", "sion"],
        alternateDisplayChunks: [],
        chunkReason: "The ending chunk carries the main pattern.",
        matchedPatterns: [],
      },
      conceptLabels: {
        originLabels: ["latin-derived"],
        patternLabels: ["sion"],
        morphologyLabels: [],
      },
      nextStep: {
        practiceFocus: "Practice words that end in -sion.",
        shouldReviewSoon: true,
        suggestedSimilarWordTypes: ["-sion words"],
      },
    }),
  });

  assert.equal(
    result.wordTeaching.conceptTeaching.summary,
    getStoredWordTeachingOnlyPrecompute("torsion")?.wordTeaching.conceptTeaching
      .summary ?? "",
  );
  assert.deepEqual(
    result.conceptLabels.originLabels,
    getStoredWordTeachingOnlyPrecompute("torsion")?.conceptLabels.originLabels ??
      [],
  );
});

test("collects new matcher patterns from concrete and structural rules", () => {
  const matched = getNewMatchedPatterns("center");

  assert.deepEqual(matched, [
    { label: "blend nt" },
    { label: "r-controlled vowel er" },
    { label: "suffix -er" },
    {
      label: "two syllables",
      matchedParts: ["cen", "ter"],
      alternateMatchedParts: [["cent", "er"]],
    },
  ]);
});

test("detects double consonant patterns in the new matcher", () => {
  const matched = getNewMatchedPatterns("occur");

  assert.equal(
    matched.some((pattern) => pattern.label === "double consonant cc"),
    true,
  );
});

test("adds new matcher patterns to word breakdown", () => {
  const output = applyNewPatternsToOutput("ghost", makeOutput({}));

  assert.equal(
    output.wordBreakdown.matchedPatterns?.some(
      (pattern) => pattern.label === "digraph gh",
    ) ?? false,
    true,
  );
  assert.equal(
    output.wordBreakdown.matchedPatterns?.some(
      (pattern) => pattern.label === "blend st",
    ) ?? false,
    true,
  );
});

test("computes sound-aware matched patterns from phonemes", () => {
  assert.deepEqual(
    getSoundAwareMatchedPatterns("city", ["S", "IH1", "T", "IY0"]),
    [
      { label: "soft c (phoneme-validated)" },
      { label: "final y says long e (phoneme-validated)" },
    ],
  );

  assert.deepEqual(
    getSoundAwareMatchedPatterns("boy", ["B", "OY1"]),
    [{ label: "oi/oy says oi (phoneme-validated)" }],
  );

  assert.deepEqual(
    getSoundAwareMatchedPatterns("comb", ["K", "OW1", "M"]),
    [{ label: "silent b (phoneme-validated)" }],
  );

  assert.deepEqual(
    getSoundAwareMatchedPatterns("science", ["S", "AY1", "AH0", "N", "S"]),
    [{ label: "silent c (phoneme-validated)" }],
  );

  assert.deepEqual(
    getSoundAwareMatchedPatterns("sign", ["S", "AY1", "N"]),
    [{ label: "silent g (phoneme-validated)" }],
  );

  assert.deepEqual(
    getSoundAwareMatchedPatterns("high", ["HH", "AY1"]),
    [
      { label: "silent gh (phoneme-validated)" },
      { label: "igh says long i (phoneme-validated)" },
    ],
  );

  assert.deepEqual(
    getSoundAwareMatchedPatterns("what", ["W", "AH1", "T"]),
    [{ label: "silent h (phoneme-validated)" }],
  );

  assert.deepEqual(
    getSoundAwareMatchedPatterns("knee", ["N", "IY1"]),
    [{ label: "silent k (phoneme-validated)" }],
  );

  assert.deepEqual(
    getSoundAwareMatchedPatterns("half", ["HH", "AE1", "F"]),
    [{ label: "silent l (phoneme-validated)" }],
  );

  assert.deepEqual(
    getSoundAwareMatchedPatterns("autumn", ["AO1", "T", "AH0", "M"]),
    [{ label: "silent n (phoneme-validated)" }],
  );

  assert.deepEqual(
    getSoundAwareMatchedPatterns("castle", ["K", "AE1", "S", "AH0", "L"]),
    [{ label: "silent t (phoneme-validated)" }],
  );

  assert.deepEqual(
    getSoundAwareMatchedPatterns("answer", ["AE1", "N", "S", "ER0"]),
    [{ label: "silent w (phoneme-validated)" }],
  );

  assert.deepEqual(getSoundAwareMatchedPatterns("ghost", ["G", "OW1", "S", "T"]), []);
});

test("derives child-friendly pronunciation from ARPAbet-like phonemes", () => {
  assert.deepEqual(
    deriveFriendlyPronunciationChunks(["AO0", "TH", "EH1", "N", "T", "AH0", "K", "EY2", "T"]),
    ["aw", "THEN", "tuh", "kayt"],
  );
  assert.equal(
    deriveFriendlyPronunciation(["S", "IH1", "G", "N", "AH0", "T"]),
    "SIG-nuht",
  );
  assert.equal(
    deriveFriendlyPronunciation(["F", "OW1", "T", "AH0", "G", "R", "AE2", "F"]),
    "FOH-tuh-graf",
  );
  assert.equal(deriveFriendlyPronunciation(["T", "AY1", "M"]), "tym");
  assert.equal(deriveFriendlyPronunciation(["W", "AY1", "L"]), "wyl");
});

test("merges stored sound-aware patterns into matched patterns for pilot words", () => {
  const originalCustomLists = loadCustomWordLists();

  try {
    saveCustomWordLists([
      {
        id: "g2p-pilot-test",
        name: "G2P Pilot Test",
        owner_user_id: "legacy",
        words: [
          {
            word: "city",
            level: "2",
            grade_band: "",
            difficulty: "",
            origin: "",
            definition: "",
            example_sentence: "",
            patterns: [],
            common_mistakes: [],
            coach_tip: "",
            part_of_speech: "",
            word_breakdown: {
              display_chunks: ["ci", "ty"],
              alternate_display_chunks: [],
              chunk_reason: "The word breaks as ci + ty.",
              matched_patterns: [
                {
                  label: "two syllables",
                  matchedParts: ["ci", "ty"],
                },
              ],
            },
            phoneme_metadata: {
              source: "g2p-en",
              phonemes: ["S", "IH1", "T", "IY0"],
              sound_aware_patterns: [
                {
                  label: "soft c (phoneme-validated)",
                },
              ],
              friendly_chunks: ["SIH", "tee"],
              say_aloud_tip: "Say it slowly: SIH-tee",
              silent_letters: [],
              tricky_parts: [],
            },
          },
        ],
      },
    ]);

    const storedBreakdown = getStoredWordBreakdown("city");
    assert.ok(storedBreakdown);
    assert.equal(
      storedBreakdown?.matchedPatterns?.some(
        (pattern) => pattern.label === "soft c (phoneme-validated)",
      ) ?? false,
      true,
    );

    const output = applyNewPatternsToOutput("city", makeOutput({}));
    assert.equal(
      output.wordBreakdown.matchedPatterns?.some(
        (pattern) => pattern.label === "soft c (phoneme-validated)",
      ) ?? false,
      true,
    );
  } finally {
    saveCustomWordLists(originalCustomLists);
  }
});

test("uses friendly pronunciation cue for words with stored phonemes", () => {
  const originalCustomLists = loadCustomWordLists();
  const aberration = getWordByText("aberration");
  assert.ok(aberration);

  try {
    saveCustomWordLists([
      {
        id: "friendly-pronunciation-test",
        name: "Friendly Pronunciation Test",
        owner_user_id: "test-user",
        words: [
          {
            ...aberration,
            phoneme_metadata: {
              source: "g2p-en",
              phonemes: ["AE2", "B", "ER0", "EY1", "SH", "AH0", "N"],
              sound_aware_patterns: [],
              friendly_chunks: ["a", "ber", "AY", "shuhn"],
              say_aloud_tip: "Say it slowly: a-ber-AY-shuhn",
              pronunciation_confidence: "high",
              silent_letters: [],
              tricky_parts: [],
            },
          },
        ],
      },
    ]);

    assert.equal(getFriendlyPronunciationCue("about"), "Say it slowly: uh-BOWT");
    assert.equal(
      getFriendlyPronunciationCue("aberration"),
      "Say it slowly: a-ber-AY-shuhn",
    );
  } finally {
    saveCustomWordLists(originalCustomLists);
  }
});

test("adds deterministic sound-aware notes to stored say-aloud tips", () => {
  assert.equal(
    buildStoredSayAloudTip("phlox", ["F", "L", "AA1", "K", "S"], ["flahks"]),
    "Sounds like: flahks.\nThe ph makes the f sound.",
  );
  assert.equal(
    buildStoredSayAloudTip("muscle", ["M", "AH1", "S", "AH0", "L"], [
      "MUS",
      "suhl",
    ]),
    "Say it slowly: MUS-suhl.\nThe final e is there, but the u does not say its name.",
  );
});

test("builds sound-aware notes for silent letters and long-u silent-e words", () => {
  assert.equal(
    buildSoundAwareTipNote("answer", ["AE1", "N", "S", "ER0"]),
    "The w is silent here.",
  );
  assert.equal(
    buildSoundAwareTipNote("altitude", [
      "AE1",
      "L",
      "T",
      "AH0",
      "T",
      "UW2",
      "D",
    ]),
    "The final e helps the u say its name.",
  );
  assert.equal(
    buildSoundAwareTipNote("tide", ["T", "AY1", "D"]),
    "The final e helps the i say its name.",
  );
  assert.equal(
    buildSoundAwareTipNote("note", ["N", "OW1", "T"]),
    "The final e helps the o say its name.",
  );
  assert.equal(
    buildSoundAwareTipNote("made", ["M", "EY1", "D"]),
    "The final e helps the a say its name.",
  );
});

test("derives structured silent letters and tricky parts from phoneme facts", () => {
  assert.deepEqual(
    derivePhonemeTeachingFacts("answer", ["AE1", "N", "S", "ER0"]).silentLetters,
    [
      {
        text: "w",
        label: "silent w",
        reason: "The w is written but not pronounced.",
        source: "phoneme-validated",
      },
    ],
  );

  assert.deepEqual(
    derivePhonemeTeachingFacts("phlox", ["F", "L", "AA1", "K", "S"]).trickyParts,
    [
      {
        text: "ph",
        label: "ph says f",
        sounds_like: "f",
        reason: "The ph is spelled ph but sounds like f here.",
        source: "derived-rule",
      },
    ],
  );

  assert.equal(
    derivePhonemeTeachingFacts("requisition", [
      "R",
      "EH2",
      "K",
      "W",
      "AH0",
      "Z",
      "IH1",
      "SH",
      "AH0",
      "N",
    ]).trickyParts.some((fact) => fact.label === "tion says shun"),
    true,
  );

  assert.equal(
    derivePhonemeTeachingFacts("tide", ["T", "AY1", "D"]).trickyParts.some(
      (fact) =>
        fact.label === "final e pattern" &&
        fact.reason === "The final e helps the i say its name.",
    ),
    true,
  );
});

test("suppresses soft c for cious endings", () => {
  const patterns = getSoundAwareMatchedPatterns("gracious", [
    "G",
    "R",
    "EY1",
    "SH",
    "AH0",
    "S",
  ]);

  assert.equal(
    patterns.some((pattern) => pattern.label === "soft c (phoneme-validated)"),
    false,
  );
});

test("prefers stored say-aloud tip metadata when available", () => {
  const originalCustomLists = loadCustomWordLists();

  try {
    saveCustomWordLists([
      {
        id: "say-aloud-test",
        name: "Say Aloud Test",
        owner_user_id: "legacy",
        words: [
          {
            word: "city",
            level: "2",
            grade_band: "",
            difficulty: "",
            origin: "",
            definition: "",
            example_sentence: "",
            patterns: [],
            common_mistakes: [],
            coach_tip: "",
            part_of_speech: "",
            phoneme_metadata: {
              source: "g2p-en",
              phonemes: ["S", "IH1", "T", "IY0"],
              sound_aware_patterns: [],
              friendly_chunks: ["sih", "TEE"],
              say_aloud_tip: "Say it slowly: SIH-tee",
              silent_letters: [],
              tricky_parts: [],
            },
          },
        ],
      },
    ]);

    assert.equal(getFriendlyPronunciationCue("city"), "Say it slowly: SIH-tee");
  } finally {
    saveCustomWordLists(originalCustomLists);
  }
});

test("flags awkward child-friendly pronunciation renderings with medium confidence", () => {
  const audit = auditFriendlyPronunciation({
    word: "time",
    level: "1",
    phoneme_metadata: {
      phonemes: ["T", "AY1", "M"],
      friendly_chunks: ["TEYEM"],
      say_aloud_tip: "Sounds like: TEYEM",
    },
  });

  assert.equal(audit.confidence, "low");
  assert.equal(
    audit.flags.includes("known awkward respelling pattern"),
    true,
  );
  assert.equal(audit.flags.includes("all-caps single lump"), true);
});

test("groups flagged pronunciations into review buckets", () => {
  const entry = auditFriendlyPronunciation({
    word: "toreador",
    level: "2",
    phoneme_metadata: {
      phonemes: ["T", "AO2", "R", "IY0", "AH0", "D", "AO1", "R"],
      friendly_chunks: ["TROY", "er", "dawr"],
      say_aloud_tip: "Say it slowly: TROY-er-dawr",
    },
  });

  const buckets = buildPronunciationReviewBuckets([entry]);
  assert.equal(buckets.suspicious_oy_without_oi_oy.length, 1);
});

test("imports named custom lists and supports list-scoped practice lookup", async () => {
  const originalCustomLists = loadCustomWordLists();

  try {
    const result = await importCustomWords(
      {
        listName: "Wind Words",
        words: ["zephyrette"],
        overwriteList: true,
      },
      {
        generateMetadata: async (words) =>
          words.map((word) => ({
            word,
            definition: `A playful definition for ${word}.`,
            origin: "French",
            exampleSentence: `${word} drifted across the page.`,
            partOfSpeech: "noun",
          })),
      },
    );

    assert.equal(result.importedCount, 1);
    assert.equal(result.list.name, "Wind Words");
    assert.equal(result.words[0]?.word, "zephyrette");

    const customList = getCustomWordListById(result.list.id, "legacy");
    assert.ok(customList);
    assert.equal(customList?.words.length, 1);
    assert.equal(customList?.name, "Wind Words");
    assert.equal(buildWordResponse(customList?.words[0]!).word, "zephyrette");

    const importedWord = getWordByText("zephyrette");
    assert.ok(importedWord);
    assert.equal(importedWord?.origin, "French");
    assert.equal(importedWord?.level, "custom");

    const nextWord = pickNextWord(undefined, [], result.list.id, undefined, "legacy");
    assert.equal(nextWord.word, "zephyrette");

    const publicWord = buildWordResponse(importedWord!);
    assert.equal(publicWord.definition, "A playful definition for *****.");
    assert.equal(publicWord.exampleSentence, "***** drifted across the page.");
  } finally {
    saveCustomWordLists(originalCustomLists);
  }
});

test("rotates through custom-list words before repeating", () => {
  const originalCustomLists = loadCustomWordLists();

  try {
    saveCustomWordLists([
      {
        id: "rotation-list",
        name: "Rotation List",
        owner_user_id: "legacy",
        words: [
          {
            word: "alpha",
            level: "custom",
            grade_band: "",
            difficulty: "",
            origin: "",
            definition: "",
            example_sentence: "",
            patterns: [],
            common_mistakes: [],
            coach_tip: "",
            part_of_speech: "",
          },
          {
            word: "beta",
            level: "custom",
            grade_band: "",
            difficulty: "",
            origin: "",
            definition: "",
            example_sentence: "",
            patterns: [],
            common_mistakes: [],
            coach_tip: "",
            part_of_speech: "",
          },
          {
            word: "gamma",
            level: "custom",
            grade_band: "",
            difficulty: "",
            origin: "",
            definition: "",
            example_sentence: "",
            patterns: [],
            common_mistakes: [],
            coach_tip: "",
            part_of_speech: "",
          },
        ],
      },
    ]);

    const picks = [
      pickNextWord(undefined, [], "rotation-list", undefined, "legacy").word,
      pickNextWord(undefined, [], "rotation-list", undefined, "legacy").word,
      pickNextWord(undefined, [], "rotation-list", undefined, "legacy").word,
    ];

    assert.equal(new Set(picks).size, 3);
  } finally {
    saveCustomWordLists(originalCustomLists);
  }
});

test("reuses built-in metadata for imported words already in the main word bank", async () => {
  const originalCustomLists = loadCustomWordLists();
  let generateCalled = false;

  try {
    const result = await importCustomWords(
      {
        listName: "Flower Words",
        words: ["phlox"],
        overwriteList: true,
      },
      {
        generateMetadata: async () => {
          generateCalled = true;
          return [];
        },
      },
    );

    assert.equal(generateCalled, false);
    assert.equal(result.importedCount, 1);
    assert.equal(result.words[0]?.word, "phlox");
    assert.equal(result.words[0]?.definition.length > 0, true);
    assert.equal(result.words[0]?.level, getWordByText("phlox")?.level);
  } finally {
    saveCustomWordLists(originalCustomLists);
  }
});

test("scopes custom-list listing and retrieval by owner user id", () => {
  const originalCustomLists = loadCustomWordLists();

  try {
    saveCustomWordLists([
      {
        id: "list-a",
        name: "List A",
        owner_user_id: "user-a",
        words: [],
      },
      {
        id: "list-b",
        name: "List B",
        owner_user_id: "user-b",
        words: [],
      },
    ]);

    const userALists = listCustomWordListsForUser("user-a");
    assert.equal(userALists.length, 1);
    assert.equal(userALists[0]?.id, "list-a");

    assert.equal(getCustomWordListById("list-b", "user-a"), undefined);
    assert.equal(getCustomWordListById("list-b", "user-b")?.name, "List B");
  } finally {
    saveCustomWordLists(originalCustomLists);
  }
});

test("throws when importing into a listId not owned by caller", async () => {
  const originalCustomLists = loadCustomWordLists();

  try {
    saveCustomWordLists([
      {
        id: "owned-by-a",
        name: "Owned A",
        owner_user_id: "user-a",
        words: [],
      },
    ]);

    await assert.rejects(
      importCustomWords(
        {
          listId: "owned-by-a",
          listName: "Owned A",
          words: ["pulpit"],
        },
        {
          ownerUserId: "user-b",
          generateMetadata: async () => [],
        },
      ),
      /Unknown custom list: owned-by-a/,
    );
  } finally {
    saveCustomWordLists(originalCustomLists);
  }
});

test("imports foreign-origin words and supports origin-scoped practice lookup", async () => {
  const originalForeignOriginLists = loadForeignOriginWordLists();

  try {
    const result = await importForeignOriginWords(
      {
        entries: [
          { word: "staccato", origin: "Italian" },
          { word: "tornado", origin: "Spanish" },
        ],
        overwriteOrigin: true,
      },
      {
        generateMetadata: async (entries) =>
          entries.map((entry) => ({
            word: entry.word,
            origin: entry.origin,
            definition: `Definition for ${entry.word}.`,
            exampleSentence: `${entry.word} appears in the sentence.`,
            partOfSpeech: "noun",
          })),
      },
    );

    assert.equal(result.importedCount, 2);
    assert.equal(result.origins.some((origin) => origin.origin === "Italian"), true);
    assert.equal(result.origins.some((origin) => origin.origin === "Spanish"), true);

    const italianList = getForeignOriginWordListByOrigin("Italian");
    assert.ok(italianList);
    assert.equal(italianList?.words.length, 1);
    assert.equal(italianList?.words[0]?.level, "foreign");

    const spanishList = getForeignOriginWordListByOrigin("Spanish");
    assert.ok(spanishList);
    assert.equal(spanishList?.words.length, 1);

    const nextItalianWord = pickNextWord(
      undefined,
      [],
      undefined,
      "Italian",
    );
    assert.equal(nextItalianWord.word.toLowerCase(), "staccato");

    const nextSpanishWord = pickNextWord(
      undefined,
      [],
      undefined,
      "Spanish",
    );
    assert.equal(nextSpanishWord.word.toLowerCase(), "tornado");
  } finally {
    saveForeignOriginWordLists(originalForeignOriginLists);
  }
});

test("accepts custom-list next-word queries when level is sent as NaN", () => {
  const query = LevelQuerySchema.parse({
    level: "NaN",
    customListId: "list-123",
    exclude: undefined,
  });

  assert.equal(query.level, undefined);
  assert.equal(query.customListId, "list-123");
});

test("accepts foreign-origin next-word queries without level", () => {
  const query = LevelQuerySchema.parse({
    level: undefined,
    customListId: undefined,
    foreignOrigin: "Italian",
    exclude: undefined,
  });

  assert.equal(query.foreignOrigin, "Italian");
  assert.equal(query.level, undefined);
});

test("classifies sentence request as a voice support intent", () => {
  const result = interpretVoiceUtterance("about", "Can you use it in a sentence?");

  assert.equal(result.intent, "example_sentence");
  assert.equal(result.displayText, "can you use it in a sentence");
  if (result.intent === "example_sentence") {
    assert.equal(result.spokenText.length > 0, true);
  }
});

test("classifies origin request as a voice support intent", () => {
  const result = interpretVoiceUtterance("about", "What is the language of origin?");

  assert.equal(result.intent, "origin");
  assert.equal(result.displayText, "what is the language of origin");
  if (result.intent === "origin") {
    assert.equal(result.spokenText.includes("comes from"), true);
  }
});

test("classifies 'say the word' as a repeat-word voice support intent", () => {
  const result = interpretVoiceUtterance("about", "Say the word");

  assert.equal(result.intent, "repeat_word");
  assert.equal(result.displayText, "say the word");
  if (result.intent === "repeat_word") {
    assert.equal(result.spokenText.includes("Spell this word: about"), true);
  }
});

test("classifies 'what's the word again' as a repeat-word voice support intent", () => {
  const result = interpretVoiceUtterance("about", "What's the word again?");

  assert.equal(result.intent, "repeat_word");
  assert.equal(result.displayText, "what's the word again");
  if (result.intent === "repeat_word") {
    assert.equal(result.spokenText.includes("Spell this word: about"), true);
  }
});

test("classifies 'what's the word' as a repeat-word voice support intent", () => {
  const result = interpretVoiceUtterance("about", "What's the word?");

  assert.equal(result.intent, "repeat_word");
  assert.equal(result.displayText, "what's the word");
  if (result.intent === "repeat_word") {
    assert.equal(result.spokenText.includes("Spell this word: about"), true);
  }
});

test("classifies 'whats the word' as a repeat-word voice support intent", () => {
  const result = interpretVoiceUtterance("about", "Whats the word?");

  assert.equal(result.intent, "repeat_word");
  assert.equal(result.displayText, "whats the word");
  if (result.intent === "repeat_word") {
    assert.equal(result.spokenText.includes("Spell this word: about"), true);
  }
});

test("classifies 'what is the word' as a repeat-word voice support intent", () => {
  const result = interpretVoiceUtterance("about", "What is the word?");

  assert.equal(result.intent, "repeat_word");
  assert.equal(result.displayText, "what is the word");
  if (result.intent === "repeat_word") {
    assert.equal(result.spokenText.includes("Spell this word: about"), true);
  }
});

test("classifies likely transcript variant 'what's the world' as a repeat-word voice support intent", () => {
  const result = interpretVoiceUtterance("about", "What's the world?");

  assert.equal(result.intent, "repeat_word");
  assert.equal(result.displayText, "what's the world");
  if (result.intent === "repeat_word") {
    assert.equal(result.spokenText.includes("Spell this word: about"), true);
  }
});

test("classifies 'tell me the word' as a repeat-word voice support intent", () => {
  const result = interpretVoiceUtterance("about", "Tell me the word");

  assert.equal(result.intent, "repeat_word");
  assert.equal(result.displayText, "tell me the word");
  if (result.intent === "repeat_word") {
    assert.equal(result.spokenText.includes("Spell this word: about"), true);
  }
});

test("classifies 'tell me the word again' as a repeat-word voice support intent", () => {
  const result = interpretVoiceUtterance("about", "Tell me the word again");

  assert.equal(result.intent, "repeat_word");
  assert.equal(result.displayText, "tell me the word again");
  if (result.intent === "repeat_word") {
    assert.equal(result.spokenText.includes("Spell this word: about"), true);
  }
});

test("classifies 'tell the word' as a repeat-word voice support intent", () => {
  const result = interpretVoiceUtterance("about", "Tell the word");

  assert.equal(result.intent, "repeat_word");
  assert.equal(result.displayText, "tell the word");
  if (result.intent === "repeat_word") {
    assert.equal(result.spokenText.includes("Spell this word: about"), true);
  }
});

test("classifies 'tell the word again' as a repeat-word voice support intent", () => {
  const result = interpretVoiceUtterance("about", "Tell the word again");

  assert.equal(result.intent, "repeat_word");
  assert.equal(result.displayText, "tell the word again");
  if (result.intent === "repeat_word") {
    assert.equal(result.spokenText.includes("Spell this word: about"), true);
  }
});

test("classifies 'can i have the word please' as a repeat-word voice support intent", () => {
  const result = interpretVoiceUtterance("about", "Can I have the word please?");

  assert.equal(result.intent, "repeat_word");
  assert.equal(result.displayText, "can i have the word please");
  if (result.intent === "repeat_word") {
    assert.equal(result.spokenText.includes("Spell this word: about"), true);
  }
});

test("masks the target word in support-request display text", () => {
  const result = interpretVoiceUtterance(
    "cranium",
    "What does cranium mean?",
  );

  assert.equal(result.intent, "definition");
  assert.equal(result.displayText, "what does the challenge word mean");
});

test("blocks exact target-word utterances from appearing as spelling input", () => {
  const result = interpretVoiceUtterance("cranium", "cranium");

  assert.equal(result.intent, "unknown");
  assert.equal(result.displayText, "Sorry, I can't spell it for you.");
});

test("normalizes hyphenated spoken spelling into letters", () => {
  const result = normalizeSpokenSpelling("u-n-i-c-o-r-n");

  assert.equal(result?.parsedAttempt, "unicorn");
  assert.equal((result?.confidence ?? 0) > 0.9, true);
});

test("normalizes common spoken letter names into spelling text", () => {
  const result = normalizeSpokenSpelling("you en eye see oh ar en");

  assert.equal(result?.parsedAttempt, "unicorn");
  assert.equal((result?.confidence ?? 0) >= 0.85, true);
});

test("classifies spoken letters as a spelling attempt", () => {
  const result = interpretVoiceUtterance("about", "a b o u t");

  assert.equal(result.intent, "spelling_attempt");
  if (result.intent === "spelling_attempt") {
    assert.equal(result.parsedAttempt, "about");
    assert.equal(result.shouldAutoSubmit, false);
  }
});

test("recovers collapsed short transcripts as spelling attempts", () => {
  const result = interpretVoiceUtterance("bias", "bia");

  assert.equal(result.intent, "spelling_attempt");
  if (result.intent === "spelling_attempt") {
    assert.equal(result.parsedAttempt, "bia");
    assert.equal(result.shouldAutoSubmit, false);
    assert.equal(result.confidence >= 0.7, true);
  }
});

test("treats short incorrect collapsed spellings as attempts instead of unknown", () => {
  const result = interpretVoiceUtterance("about", "abot");

  assert.equal(result.intent, "spelling_attempt");
  if (result.intent === "spelling_attempt") {
    assert.equal(result.parsedAttempt, "abot");
  }
});

test("uses strict read-exactly TTS instructions for pronunciation prompts", () => {
  const original = process.env.SPELLING_COACH_TTS_INSTRUCTIONS;
  process.env.SPELLING_COACH_TTS_INSTRUCTIONS = "on";

  try {
    assert.equal(
      buildDefaultTtsInstructions("Spell this word: anorak."),
      "Read the provided text exactly. Do not omit the target word. Pronounce the target word as a spoken word. Do not spell letters. Do not read it character by character. Say the word once clearly and naturally.",
    );
    assert.equal(buildDefaultTtsInstructions("Hello there."), undefined);
  } finally {
    if (original === undefined) {
      delete process.env.SPELLING_COACH_TTS_INSTRUCTIONS;
    } else {
      process.env.SPELLING_COACH_TTS_INSTRUCTIONS = original;
    }
  }
});

test("adds guided pronunciation instructions for risky words with friendly chunks", () => {
  const original = process.env.SPELLING_COACH_TTS_INSTRUCTIONS;
  process.env.SPELLING_COACH_TTS_INSTRUCTIONS = "on";

  try {
    assert.equal(
      buildDefaultTtsInstructions("Spell this word: xylyl."),
      "Read the provided text exactly. Do not omit the target word. Pronounce the target word as a spoken word. Do not spell letters. Do not read it character by character. Say the word once clearly and naturally. Pronounce xylyl as ZY-luh.",
    );
  } finally {
    if (original === undefined) {
      delete process.env.SPELLING_COACH_TTS_INSTRUCTIONS;
    } else {
      process.env.SPELLING_COACH_TTS_INSTRUCTIONS = original;
    }
  }
});
