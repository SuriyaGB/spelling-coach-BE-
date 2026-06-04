import test from "node:test";
import assert from "node:assert/strict";
import {
  buildSpellingCoachInput,
  buildWordPrecomputeInput,
  buildWordResponse,
  LevelQuerySchema,
  maskWordInExampleSentence,
} from "./inputBuilder.js";
import {
  applyBlendPatternsToPrecompute,
  getMatchedBlendPatterns,
} from "./blendPatterns.js";
import { importCustomWords } from "./customWordImport.js";
import { importForeignOriginWords } from "./foreignOriginImport.js";
import { applyDeterministicPatternsToPrecompute } from "./deterministicPatterns.js";
import {
  hasWordTeachingPrecompute,
  runSplitSpellingCoachAgent,
  warmWordTeachingPrecompute,
} from "./optimizedCoach.js";
import {
  buildSpellingCoachPrompt,
  buildWordTeachingPrecomputePrompt,
  SPELLING_COACH_SYSTEM_PROMPT,
} from "./prompt.js";
import { runSpellingCoachAgent } from "./runAgent.js";
import {
  buildReferenceHintsText,
  buildSpellingRuleHintsText,
  getReferenceHints,
} from "./referenceData.js";
import type { DeepAgentLike } from "./agent.js";
import type { DirectModelLike } from "./directModel.js";
import type { SpellingCoachInput, SpellingCoachOutput } from "./schemas.js";
import {
  getCustomWordListById,
  getForeignOriginWordListByOrigin,
  getWordByText,
  listCustomWordListsForUser,
  loadCustomWordLists,
  loadForeignOriginWordLists,
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
    formTeaching?: Partial<SpellingCoachOutput["wordTeaching"]["formTeaching"]>;
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
      errorTypes: [],
      primaryErrorFocus: "",
      likelyWrongWordInterpretation: false,
      usedMeaningDisambiguationWell: false,
      ...overrides.missAnalysis,
    },
    wordTeaching: {
      formTeaching: {
        summary: "",
        patterns: [],
        chunks: [],
        chunkReason: "",
        sayAloudFocus: "",
        ...overrides.wordTeaching?.formTeaching,
      },
      conceptTeaching: {
        summary: "",
        meaningFocus: "",
        originFocus: "",
        morphologyFocus: "",
        originLabels: [],
        morphologyLabels: [],
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
      chunkReason: "",
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
      errorTypes: ["missing-letter deletion", "consonant cluster omission"],
      primaryErrorFocus: "Remember the sc cluster in ad + scititious.",
    },
    wordTeaching: {
      formTeaching: {
        summary: "Keep the early sc cluster intact before finishing the -itious ending.",
        patterns: ["sc-cluster", "-itious"],
        chunks: ["ad", "scit", "itious"],
        chunkReason: "The split highlights the missing sc cluster and the stable -itious ending.",
        sayAloudFocus: "ad-sci-ti-tious",
      },
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
      errorTypes: ["phonetic substitution", "pattern reduction"],
      primaryErrorFocus: "Use the stored patterns ch and ph instead of writing only the sounds you hear.",
    },
    wordTeaching: {
      formTeaching: {
        summary: "Chunk the word into arachno + phagous while locking in ch and ph.",
        patterns: ["ch says k", "ph says f", "-gous"],
        chunks: ["arachno", "phagous"],
        chunkReason: "The chunks match the strongest reusable teaching pieces and reduce overload.",
        sayAloudFocus: "a-rach-no-phag-ous",
      },
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
  assert.equal(result.nextStep.shouldReviewSoon, true);
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
      errorTypes: [],
      primaryErrorFocus: "Accurate spelling",
      usedMeaningDisambiguationWell: true,
    },
    wordTeaching: {
      formTeaching: {
        summary: "Notice the two short-vowel chunks pul + pit.",
        patterns: ["closed syllables"],
        chunks: ["pul", "pit"],
        chunkReason: "The chunks reinforce the correct short-vowel structure without adding extra complexity.",
        sayAloudFocus: "pul-pit",
      },
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
  assert.equal(result.missAnalysis.errorTypes.length, 0);
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
  assert.equal(result.coachingText.sayAloudTip, "Say a-bout.");
  assert.equal(result.coachingText.fullExplanation, "");
  assert.deepEqual(result.wordBreakdown.displayChunks, ["ab", "out"]);
  assert.equal(result.wordBreakdown.chunkReason, "Has vowel team ou.");
  assert.equal(result.wordTeaching.formTeaching.summary, "");
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
      errorTypes: ["missing chunk", "ending compression"],
      primaryErrorFocus: "Keep the full ti + tious ending instead of shrinking it to tous.",
    },
    wordTeaching: {
      formTeaching: {
        summary: "Spell fictitious as fic + ti + tious.",
        patterns: ["-itious"],
        chunks: ["fic", "ti", "tious"],
        chunkReason: "The chunks expose the exact section that was omitted.",
        sayAloudFocus: "fic - ti - tious",
      },
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
      errorTypes: [],
      primaryErrorFocus: "Accurate spelling",
      usedMeaningDisambiguationWell: true,
    },
    wordTeaching: {
      formTeaching: {
        summary: "Notice the two short-vowel chunks pul + pit.",
        patterns: ["closed syllables"],
        chunks: ["pul", "pit"],
        chunkReason: "The chunks reinforce the correct short-vowel structure without adding extra complexity.",
        sayAloudFocus: "pul-pit",
      },
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
  assert.equal(SPELLING_COACH_SYSTEM_PROMPT.includes('"mostRelevantToError": "form" | "concept" | "mixed" | "unclear"'), true);
  assert.equal(SPELLING_COACH_SYSTEM_PROMPT.includes('confidence is below 0.75'), true);
  assert.equal(SPELLING_COACH_SYSTEM_PROMPT.includes("Curated spelling-rule hints may be provided"), false);
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
      errorTypes: [],
      primaryErrorFocus: "Accurate spelling",
      usedMeaningDisambiguationWell: true,
    },
    wordTeaching: {
      formTeaching: {
        summary: "Notice the two short-vowel chunks pul + pit.",
        patterns: ["closed syllables"],
        chunks: ["pul", "pit"],
        chunkReason: "The chunks reinforce the correct short-vowel structure without adding extra complexity.",
        sayAloudFocus: "pul-pit",
      },
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
          errorTypes: ["short vowel confusion"],
          primaryErrorFocus: "Keep the short i in the second chunk.",
        },
        wordTeaching: {
          formTeaching: {
            summary: "The second chunk is pit, not pet.",
            patterns: ["closed syllables"],
            chunks: ["pul", "pit"],
            chunkReason: "The second chunk holds the short i that was changed.",
            sayAloudFocus: "pul-pit",
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
          errorTypes: ["phonetic substitution"],
          primaryErrorFocus: "Use ph for the f sound in this word.",
        },
        wordTeaching: {
          formTeaching: {
            summary: "This word uses ph for the f sound.",
            patterns: ["ph says f", "x ending"],
            chunks: ["ph", "lox"],
            chunkReason: "The opening pattern is the main spelling feature.",
            sayAloudFocus: "phlox",
          },
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
        "If you include a spelling rule label in wordTeaching.formTeaching.patterns, the summary, chunkReason, and sayAloudFocus must agree with that rule.",
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
});

test("warms word teaching precompute on a word-only input", async () => {
  const input = buildWordPrecomputeInput("torsion");
  const precomputeOutput = {
    wordTeaching: {
      formTeaching: {
        summary: "This word highlights the -sion ending after tors-.",
        patterns: ["sion"],
        chunks: ["tor", "sion"],
        chunkReason: "The ending chunk carries the main pattern.",
        sayAloudFocus: "tor-sion",
      },
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

  assert.equal(hasWordTeachingPrecompute(input, { runtime: "direct" }), false);

  const result = await warmWordTeachingPrecompute(input, {
    runtime: "direct",
    directModel: createDirectMockModel(precomputeOutput as never),
  });

  assert.deepEqual(result.wordBreakdown.displayChunks, ["tor", "sion"]);
  assert.equal(hasWordTeachingPrecompute(input, { runtime: "direct" }), true);
});

test("merges cached word teaching with miss-only analysis on submit", async () => {
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
      formTeaching: {
        summary: "This word highlights the -sion ending after tors-.",
        patterns: ["sion"],
        chunks: ["tor", "sion"],
        chunkReason: "The ending chunk carries the main pattern.",
        sayAloudFocus: "tor-sion",
      },
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
      errorTypes: ["phonetic substitution", "ending confusion"],
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

  const result = await runSplitSpellingCoachAgent(submitInput, {
    runtime: "direct",
    directModel: createSequenceMockModel([
      JSON.stringify(precomputeOutput),
      JSON.stringify(missOnlyOutput),
    ]),
  });

  assert.deepEqual(result.wordTeaching.formTeaching.chunks, ["tor", "sion"]);
  assert.equal(result.missAnalysis.primaryErrorFocus.includes("-sion"), true);
  assert.equal(result.errorRelevance.mostRelevantToError, "form");
});

test("applies deterministic spelling-rule descriptions for matched word patterns", () => {
  const precompute = applyDeterministicPatternsToPrecompute("mind", {
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
    wordBreakdown: {
      displayChunks: [],
      chunkReason: "",
    },
    conceptLabels: {
      originLabels: [],
      patternLabels: [],
      morphologyLabels: [],
    },
  });

  assert.equal(
    precompute.wordTeaching.formTeaching.patterns.includes(
      "I and O may sometimes say their long sounds before two consonants like -nd or -ld or -st or -lt. -D",
    ),
    true,
  );
  assert.equal(
    precompute.conceptLabels.patternLabels.includes(
      "i_o_long_before_two_consonants",
    ),
    true,
  );
});

test("identifies blend and word-ending patterns from blends.txt", () => {
  const matches = getMatchedBlendPatterns("brandy");

  assert.equal(matches.includes("blend br -D"), true);
  assert.equal(matches.includes("word ending -y -D"), true);
});

test("prefers the longest overlapping blend pattern from blends.txt", () => {
  const matches = getMatchedBlendPatterns("monstrosity");

  assert.equal(matches.includes("3-letter blend str -D"), true);
  assert.equal(matches.includes("blend st -D"), false);
  assert.equal(matches.includes("blend tr -D"), false);
});

test("appends blends.txt pattern matches without removing LLM patterns", () => {
  const precompute = applyBlendPatternsToPrecompute("ship", {
    wordTeaching: {
      formTeaching: {
        summary: "",
        patterns: ["LLM pattern"],
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
    wordBreakdown: {
      displayChunks: [],
      chunkReason: "",
    },
    conceptLabels: {
      originLabels: [],
      patternLabels: [],
      morphologyLabels: [],
    },
  }, "Greek");

  assert.equal(
    precompute.wordTeaching.formTeaching.patterns.includes("digraph sh -D"),
    true,
  );
  assert.equal(
    precompute.wordTeaching.formTeaching.patterns.includes("LLM pattern"),
    true,
  );
});

test("skips blends.txt deterministic patterns for unsupported origins", () => {
  const precompute = applyBlendPatternsToPrecompute("ship", {
    wordTeaching: {
      formTeaching: {
        summary: "",
        patterns: ["LLM pattern"],
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
    wordBreakdown: {
      displayChunks: [],
      chunkReason: "",
    },
    conceptLabels: {
      originLabels: [],
      patternLabels: [],
      morphologyLabels: [],
    },
  }, "French");

  assert.deepEqual(precompute.wordTeaching.formTeaching.patterns, ["LLM pattern"]);
});

test("applies blends.txt deterministic patterns for English origin", () => {
  const precompute = applyBlendPatternsToPrecompute("ghost", {
    wordTeaching: {
      formTeaching: {
        summary: "",
        patterns: ["LLM pattern"],
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
    wordBreakdown: {
      displayChunks: [],
      chunkReason: "",
    },
    conceptLabels: {
      originLabels: [],
      patternLabels: [],
      morphologyLabels: [],
    },
  }, "English");

  assert.equal(
    precompute.wordTeaching.formTeaching.patterns.includes(
      "silent letter digraph gh -D",
    ),
    true,
  );
  assert.equal(
    precompute.wordTeaching.formTeaching.patterns.includes("blend st -D"),
    true,
  );
});

test("applies blends.txt deterministic patterns for Old Norse origin", () => {
  const precompute = applyBlendPatternsToPrecompute("knife", {
    wordTeaching: {
      formTeaching: {
        summary: "",
        patterns: ["LLM pattern"],
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
    wordBreakdown: {
      displayChunks: [],
      chunkReason: "",
    },
    conceptLabels: {
      originLabels: [],
      patternLabels: [],
      morphologyLabels: [],
    },
  }, "Old Norse");

  assert.equal(
    precompute.wordTeaching.formTeaching.patterns.includes(
      "silent letter digraph kn -D",
    ),
    true,
  );
});

test("does not overmatch final ch context rules for which", () => {
  const precompute = applyDeterministicPatternsToPrecompute("which", {
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
    wordBreakdown: {
      displayChunks: [],
      chunkReason: "",
    },
    conceptLabels: {
      originLabels: [],
      patternLabels: [],
      morphologyLabels: [],
    },
  });

  assert.equal(
    precompute.conceptLabels.patternLabels.includes("final_ch_after_consonant"),
    false,
  );
  assert.equal(
    precompute.conceptLabels.patternLabels.includes(
      "final_ch_after_two_letter_vowel",
    ),
    false,
  );
});

test("suppresses deterministic rule matches when the word is listed as an exception", () => {
  const precompute = applyDeterministicPatternsToPrecompute("the", {
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
    wordBreakdown: {
      displayChunks: [],
      chunkReason: "",
    },
    conceptLabels: {
      originLabels: [],
      patternLabels: [],
      morphologyLabels: [],
    },
  });

  assert.equal(
    precompute.conceptLabels.patternLabels.includes("cv_one_syllable_long_vowel"),
    false,
  );
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
