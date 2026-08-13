import test from "node:test";
import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import type { IncomingMessage, ServerResponse } from "node:http";
import {
  buildStreamingSpellingCoachInput,
  readSpellingCoachStreamRequest,
  SpellingCoachStreamRequestError,
  SpellingCoachStreamRequestSchema,
  streamSpellingCoach,
} from "../streamingCoach.js";
import { runSpellingCoachAgent } from "../runAgent.js";
import { runSplitSpellingCoachAgent } from "../optimizedCoach.js";
import type { DirectModelLike } from "../directModel.js";
import type {
  SpellingCoachOutput,
  WordTeachingPrecompute,
} from "../schemas.js";

class FakeResponse extends EventEmitter {
  destroyed = false;
  writableEnded = false;
  statusCode = 0;
  headers: Record<string, string> = {};
  chunks: string[] = [];
  flushCount = 0;
  socket = { setNoDelay() {} };

  writeHead(statusCode: number, headers: Record<string, string>) {
    this.statusCode = statusCode;
    this.headers = headers;
    return this;
  }

  flushHeaders() {}

  flush() {
    this.flushCount += 1;
  }

  write(chunk: string) {
    this.chunks.push(chunk);
    return true;
  }

  end() {
    this.writableEnded = true;
    return this;
  }
}

class FakeRequest extends EventEmitter {
  aborted = false;
  destroyed = false;
  resumeCalled = false;

  resume() {
    this.resumeCalled = true;
    return this;
  }
}

const baseRequest = {
  targetWord: "about",
  childAttempt: "abot",
  level: 1,
  mode: "practice",
  definitionViewed: false,
  exampleViewed: false,
  originViewed: false,
  partOfSpeechViewed: false,
  repeatWordCount: 0,
  usedVoiceInput: false,
};

const precomputed: WordTeachingPrecompute = {
  wordTeaching: {
    conceptTeaching: {
      summary: "summary",
      meaningFocus: "meaning",
      originFocus: "origin",
      morphologyFocus: "morphology",
      originLabels: [],
      morphologyLabels: [],
      relatedForms: [],
    },
  },
  wordBreakdown: {
    displayChunks: ["a", "bout"],
    alternateDisplayChunks: [],
    chunkReason: "two chunks",
    matchedPatterns: [],
  },
  conceptLabels: {
    originLabels: [],
    patternLabels: [],
    morphologyLabels: [],
  },
};

const runtimeOutput: SpellingCoachOutput = {
  correctness: { isCorrect: false, reinforceSuccess: false },
  missAnalysis: {
    summary: "A letter is missing.",
    primaryErrorType: "missing_letter",
    secondaryErrorTypes: [],
    errorTypeEvidence: {},
    primaryErrorFocus: "missing u",
    likelyWrongWordInterpretation: false,
    usedMeaningDisambiguationWell: false,
  },
  wordTeaching: precomputed.wordTeaching,
  errorRelevance: {
    mostRelevantToError: "form",
    confidence: 1,
    reason: "The spelling form differs.",
  },
  teachingDecision: {
    strategy: "chunking",
    primaryFocus: "a-bout",
    secondaryFocuses: [],
    confidence: 1,
    rationale: "Chunk the word.",
  },
  coachingText: {
    shortFeedback: "Good try.",
    fullExplanation: "Remember the u.",
    memoryTip: "Think a-bout.",
    sayAloudTip: "Say a-bout.",
  },
  wordBreakdown: precomputed.wordBreakdown,
  conceptLabels: precomputed.conceptLabels,
  nextStep: {
    practiceFocus: "ou",
    shouldReviewSoon: true,
    suggestedSimilarWordTypes: [],
  },
};

function parseEvents(response: FakeResponse) {
  return response.chunks.map((chunk) => {
    const [eventLine, dataLine] = chunk.trim().split("\n");
    return {
      event: eventLine.slice("event: ".length),
      data: JSON.parse(dataLine.slice("data: ".length)),
    };
  });
}

async function* tokenStream(tokens: string[]): AsyncIterable<string> {
  for (const token of tokens) {
    yield token;
  }
}

function createRequest(): IncomingMessage {
  return new EventEmitter() as IncomingMessage;
}

function startReadingRequest(
  body: string,
  maxBytes?: number,
  timeoutMs?: number,
) {
  const request = new FakeRequest();
  const result = readSpellingCoachStreamRequest(
    request as unknown as IncomingMessage,
    maxBytes,
    timeoutMs,
  );
  request.emit("data", Buffer.from(body));
  request.emit("end");
  return { request, result };
}

test("stream request schema enforces the complete v1 body", () => {
  assert.equal(SpellingCoachStreamRequestSchema.safeParse(baseRequest).success, true);
  assert.equal(
    SpellingCoachStreamRequestSchema.safeParse({
      ...baseRequest,
      level: undefined,
    }).success,
    true,
  );
  assert.equal(
    SpellingCoachStreamRequestSchema.safeParse({
      ...baseRequest,
      definitionViewed: undefined,
    }).success,
    false,
  );
  assert.equal(
    SpellingCoachStreamRequestSchema.safeParse({
      ...baseRequest,
      unexpected: true,
    }).success,
    false,
  );
});

test("stream request reader validates JSON and enforces its body limit", async () => {
  const valid = startReadingRequest(JSON.stringify(baseRequest));
  assert.deepEqual(await valid.result, baseRequest);

  const invalid = startReadingRequest("{");
  await assert.rejects(invalid.result, (error: unknown) => {
    assert.equal(error instanceof SpellingCoachStreamRequestError, true);
    assert.equal((error as SpellingCoachStreamRequestError).statusCode, 400);
    return true;
  });

  const oversized = startReadingRequest(JSON.stringify(baseRequest), 8);
  await assert.rejects(oversized.result, (error: unknown) => {
    assert.equal(error instanceof SpellingCoachStreamRequestError, true);
    assert.equal((error as SpellingCoachStreamRequestError).statusCode, 413);
    return true;
  });
  assert.equal(oversized.request.resumeCalled, true);
  assert.equal(oversized.request.listenerCount("data"), 0);
  assert.equal(oversized.request.listenerCount("end"), 0);
  assert.equal(oversized.request.listenerCount("error"), 0);
  assert.equal(oversized.request.listenerCount("close"), 0);

  const abortedRequest = new FakeRequest();
  const abortedResult = readSpellingCoachStreamRequest(
    abortedRequest as unknown as IncomingMessage,
  );
  abortedRequest.emit("aborted");
  await assert.rejects(abortedResult, /aborted/);
  assert.equal(abortedRequest.listenerCount("data"), 0);
  assert.equal(abortedRequest.listenerCount("end"), 0);
  assert.equal(abortedRequest.listenerCount("error"), 0);
  assert.equal(abortedRequest.listenerCount("aborted"), 0);
  assert.equal(abortedRequest.listenerCount("close"), 0);

  const stalledRequest = new FakeRequest();
  const stalledResult = readSpellingCoachStreamRequest(
    stalledRequest as unknown as IncomingMessage,
    undefined,
    5,
  );
  await assert.rejects(stalledResult, (error: unknown) => {
    assert.equal(error instanceof SpellingCoachStreamRequestError, true);
    assert.equal((error as SpellingCoachStreamRequestError).statusCode, 408);
    return true;
  });
  assert.equal(stalledRequest.resumeCalled, true);
  stalledRequest.emit("end");
  assert.equal(stalledRequest.listenerCount("data"), 0);
  assert.equal(stalledRequest.listenerCount("end"), 0);
  assert.equal(stalledRequest.listenerCount("error"), 0);
  assert.equal(stalledRequest.listenerCount("aborted"), 0);
  assert.equal(stalledRequest.listenerCount("close"), 0);
});

test("correct spelling streams meta, precomputed, and done without runtime", async () => {
  const response = new FakeResponse();
  let runtimeCalls = 0;
  const logLines: string[] = [];
  const originalConsoleLog = console.log;
  console.log = (...args: unknown[]) => {
    logLines.push(args.map(String).join(" "));
  };

  try {
    await streamSpellingCoach(
      createRequest(),
      response as unknown as ServerResponse,
      { ...baseRequest, childAttempt: "ABOUT" },
      {
        precompute: async () => precomputed,
        runRuntime: async () => {
          runtimeCalls += 1;
          return runtimeOutput;
        },
        requestId: () => "request-correct",
      },
    );
  } finally {
    console.log = originalConsoleLog;
  }

  const events = parseEvents(response);
  assert.deepEqual(events.map(({ event }) => event), [
    "meta",
    "precomputed",
    "done",
  ]);
  assert.equal(events[0].data.isCorrect, true);
  assert.equal(
    logLines.some((line) =>
      line.includes(
        '[spelling-coach precompute timing] requestId=request-correct word="about"',
      ),
    ),
    true,
  );
  assert.equal(events[0].data.targetWordMasked, true);
  assert.deepEqual(
      Object.keys(events[0].data).sort(),
      [
        "isCorrect",
        "missAnalysis",
        "requestId",
        "sayAloudTip",
        "shortFeedback",
        "targetWord",
        "targetWordMasked",
        "timingMs",
      ].sort(),
  );
  assert.equal(events[0].data.missAnalysis.primaryErrorType, null);
  assert.deepEqual(events[0].data.missAnalysis.secondaryErrorTypes, []);
  assert.deepEqual(events[1].data.payload, precomputed);
  assert.deepEqual(Object.keys(events[1].data).sort(), ["payload", "timingMs"]);
  assert.equal(events[2].data.timings.runtimeCoachingMs, 0);
  assert.deepEqual(Object.keys(events[2].data).sort(), ["complete", "targetWord", "timings"]);
  assert.deepEqual(Object.keys(events[2].data.timings).sort(), [
    "metaMs",
    "precomputedMs",
    "runtimeCoachingMs",
    "runtimeMs",
    "totalMs",
  ]);
  assert.equal(runtimeCalls, 0);
  assert.equal(response.headers["Content-Type"], "text/event-stream");
  assert.equal(response.headers["X-Accel-Buffering"], "no");
  assert.equal(response.flushCount, events.length);
  assert.equal(response.listenerCount("close"), 0);
  assert.equal(response.listenerCount("error"), 0);
});

test("completed request bodies do not count as client disconnects", async () => {
  const request = createRequest();
  Object.assign(request, { aborted: false, destroyed: true });
  const response = new FakeResponse();

  await streamSpellingCoach(
    request,
    response as unknown as ServerResponse,
    { ...baseRequest, childAttempt: "about" },
    {
      precompute: async () => precomputed,
      requestId: () => "request-readable-destroyed",
    },
  );

  assert.equal(response.statusCode, 200);
  assert.deepEqual(parseEvents(response).map(({ event }) => event), [
    "meta",
    "precomputed",
    "done",
  ]);
});

test("fully valid stream emits section starts, chunks, completes, and done", async () => {
  const response = new FakeResponse();

  await streamSpellingCoach(
    createRequest(),
    response as unknown as ServerResponse,
    baseRequest,
    {
      precompute: async () => precomputed,
      runRuntimeStream: async () =>
        tokenStream([
          "[[SHORT_FEEDBACK]]Good try.[[END_SECTION]]",
          "[[MISS_",
          "ANALYSIS]]A letter is missing.[[END_SECTION]]",
          "[[EXPLANATION]]Remember the u.[[END_SECTION]]",
          "[[MEMORY_TIP]]Think a-bout.[[END_SECTION]]",
        ]),
      requestId: () => "request-incorrect",
      sectionTimeoutMs: 30_000,
    },
  );

  const events = parseEvents(response);
  assert.deepEqual(events.slice(0, 2).map(({ event }) => event), [
    "meta",
    "precomputed",
  ]);
  assert.equal(events[0].data.missAnalysis.primaryErrorType, "missing_letter");
  assert.deepEqual(events[0].data.missAnalysis.secondaryErrorTypes, [
    "likely_rushed",
  ]);
  assert.equal(events[0].data.missAnalysis.primaryErrorFocus, "Missing: u");
  assert.equal(events.at(-1)?.event, "done");
  const starts = events
    .filter(({ event }) => event === "section-start")
    .map(({ data }) => data.section)
  assert.deepEqual(starts, ["short_feedback", "miss_analysis", "explanation", "memory_tip"]);
  const chunks = events
    .filter(({ event }) => event === "section-chunk")
    .map(({ data }) => [data.section, data.text]);
  assert.deepEqual(chunks, [
    ["short_feedback", "Good try."],
    ["miss_analysis", "A letter is missing."],
    ["explanation", "Remember the u."],
    ["memory_tip", "Think a-bout."],
  ]);
  const completes = events
    .filter(({ event }) => event === "section-complete")
    .map(({ data }) => data.section);
  assert.deepEqual(completes, ["short_feedback", "miss_analysis", "explanation", "memory_tip"]);
  for (const event of events.filter(({ event }) => event === "section-chunk")) {
    assert.deepEqual(Object.keys(event.data).sort(), [
      "section",
      "text",
      "timingMs",
    ]);
  }
  assert.equal(events.some(({ event }) => event === "section-error"), false);
});

test("stream without END_SECTION closes sections on the next marker", async () => {
  const response = new FakeResponse();

  await streamSpellingCoach(
    createRequest(),
    response as unknown as ServerResponse,
    baseRequest,
    {
      precompute: async () => precomputed,
      runRuntimeStream: async () =>
        tokenStream([
          "[[SHORT_FEEDBACK]]Well done.[[END_SECTION]]",
          "[[MISS_ANALYSIS]]Miss text",
          "[[EXPLANATION]]Explain text",
          "[[MEMORY_TIP]]Tip text",
        ]),
      requestId: () => "request-no-end-marker",
    },
  );

  const events = parseEvents(response);
  assert.deepEqual(
    events.filter(({ event }) => event === "section-complete").map(({ data }) => data.section),
    ["short_feedback", "miss_analysis", "explanation", "memory_tip"],
  );
  assert.equal(events.some(({ event }) => event === "section-error"), false);
});

test("malformed markers emit section-error and recover at the next valid marker", async () => {
  const response = new FakeResponse();

  await streamSpellingCoach(
    createRequest(),
    response as unknown as ServerResponse,
    baseRequest,
    {
      precompute: async () => precomputed,
      runRuntimeStream: async () =>
        tokenStream([
          "[[MISS_ANALYSISS]]",
          "[[SHORT_FEEDBACK]]Recovered.[[END_SECTION]]",
          "[[MISS_ANALYSIS]]Recovered.[[END_SECTION]]",
          "[[EXPLANATION]]Explain.[[END_SECTION]]",
          "[[MEMORY_TIP]]Tip.[[END_SECTION]]",
        ]),
      requestId: () => "request-malformed-marker",
    },
  );

  const events = parseEvents(response);
  assert.equal(events.find(({ event }) => event === "section-error")?.data.error.code, "MALFORMED_MARKER");
  assert.deepEqual(
    events.filter(({ event }) => event === "section-start").map(({ data }) => data.section),
    ["short_feedback", "miss_analysis", "explanation", "memory_tip"],
  );
});

test("malformed markers after section text emit section-error and preserve prior text", async () => {
  const response = new FakeResponse();

  await streamSpellingCoach(
    createRequest(),
    response as unknown as ServerResponse,
    baseRequest,
    {
      precompute: async () => precomputed,
      runRuntimeStream: async () =>
        tokenStream([
          "[[SHORT_FEEDBACK]]Good try.[[END_SECTION]]",
          "[[MISS_ANALYSIS]]Useful text [[BROKEN_MARKER]] recovered.[[END_SECTION]]",
          "[[EXPLANATION]]Explain.[[END_SECTION]]",
          "[[MEMORY_TIP]]Tip.[[END_SECTION]]",
        ]),
      requestId: () => "request-malformed-after-text",
    },
  );

  const events = parseEvents(response);
  assert.equal(
    events.find(({ event }) => event === "section-error")?.data.error.code,
    "MALFORMED_MARKER",
  );
  const missText = events
    .filter(({ event, data }) => event === "section-chunk" && data.section === "miss_analysis")
    .map(({ data }) => data.text)
    .join("");
  assert.equal(missText, "Useful text  recovered.");
  assert.deepEqual(
    events.filter(({ event }) => event === "section-complete").map(({ data }) => data.section),
    ["short_feedback", "miss_analysis", "explanation", "memory_tip"],
  );
});

test("out-of-order markers emit errors for skipped required sections", async () => {
  const response = new FakeResponse();

  await streamSpellingCoach(
    createRequest(),
    response as unknown as ServerResponse,
    baseRequest,
    {
      precompute: async () => precomputed,
      runRuntimeStream: async () =>
        tokenStream([
          "[[SHORT_FEEDBACK]]Good try.[[END_SECTION]]",
          "[[EXPLANATION]]Explain.[[END_SECTION]]",
          "[[MEMORY_TIP]]Tip.[[END_SECTION]]",
        ]),
      requestId: () => "request-out-of-order",
    },
  );

  const events = parseEvents(response);
  assert.equal(events.find(({ event }) => event === "section-error")?.data.section, "miss_analysis");
  assert.equal(events.find(({ event }) => event === "section-error")?.data.error.code, "MISSING_SECTION_MARKER");
  assert.deepEqual(
    events.filter(({ event }) => event === "section-start").map(({ data }) => data.section),
    ["short_feedback", "explanation", "memory_tip"],
  );
});

test("missing final section emits section-error at stream end", async () => {
  const response = new FakeResponse();

  await streamSpellingCoach(
    createRequest(),
    response as unknown as ServerResponse,
    baseRequest,
    {
      precompute: async () => precomputed,
      runRuntimeStream: async () =>
        tokenStream([
          "[[SHORT_FEEDBACK]]Good try.[[END_SECTION]]",
          "[[MISS_ANALYSIS]]Miss.[[END_SECTION]]",
          "[[EXPLANATION]]Explain.[[END_SECTION]]",
        ]),
      requestId: () => "request-missing-final",
    },
  );

  const error = parseEvents(response).find(({ event }) => event === "section-error");
  assert.equal(error?.data.section, "memory_tip");
  assert.equal(error?.data.error.code, "MISSING_SECTION");
});

test("unmarked text before the first marker is never assigned to a section", async () => {
  const response = new FakeResponse();

  await streamSpellingCoach(
    createRequest(),
    response as unknown as ServerResponse,
    baseRequest,
    {
      precompute: async () => precomputed,
      runRuntimeStream: async () =>
        tokenStream([
          "loose preface",
          "[[SHORT_FEEDBACK]]Good try.[[END_SECTION]]",
          "[[MISS_ANALYSIS]]Miss.[[END_SECTION]]",
          "[[EXPLANATION]]Explain.[[END_SECTION]]",
          "[[MEMORY_TIP]]Tip.[[END_SECTION]]",
        ]),
      requestId: () => "request-unmarked-before",
    },
  );

  const events = parseEvents(response);
  assert.equal(events.find(({ event }) => event === "section-error")?.data.error.code, "UNMARKED_TEXT_BEFORE_FIRST_MARKER");
  const firstChunk = events.find(({ event }) => event === "section-chunk");
  assert.equal(firstChunk?.data.text, "Good try.");
});

test("unmarked text after a valid marker streams as section text", async () => {
  const response = new FakeResponse();

  await streamSpellingCoach(
    createRequest(),
    response as unknown as ServerResponse,
    baseRequest,
    {
      precompute: async () => precomputed,
      runRuntimeStream: async () =>
        tokenStream([
          "[[SHORT_FEEDBACK]]Good try.[[END_SECTION]]",
          "[[MISS_ANALYSIS]]Miss text",
          " keeps going[[END_SECTION]]",
          "[[EXPLANATION]]Explain.[[END_SECTION]]",
          "[[MEMORY_TIP]]Tip.[[END_SECTION]]",
        ]),
      requestId: () => "request-unmarked-after",
    },
  );

  const chunks = parseEvents(response)
    .filter(({ event, data }) => event === "section-chunk" && data.section === "miss_analysis")
    .map(({ data }) => data.text)
    .join("");
  assert.equal(chunks, "Miss text keeps going");
  assert.equal(
    parseEvents(response).some(({ event }) => event === "section-error"),
    false,
  );
});

test("runtime failure emits a section error and still finishes", async () => {
  const response = new FakeResponse();

  await streamSpellingCoach(
    createRequest(),
    response as unknown as ServerResponse,
    baseRequest,
    {
      precompute: async () => precomputed,
      runRuntime: async () => {
        throw new Error("model unavailable");
      },
      requestId: () => "request-failure",
    },
  );

  const events = parseEvents(response);
  const errors = events.filter(({ event }) => event === "section-error");
  assert.deepEqual(events.map(({ event }) => event), [
    "meta",
    "precomputed",
    "section-error",
    "done",
  ]);
  assert.equal(errors.length, 1);
  assert.equal(
    errors.every(
      ({ data }) => data.error.code === "SECTION_GENERATION_FAILED",
    ),
    true,
  );
  assert.equal(
    errors.every(
      ({ data }) =>
        data.error.message === "Runtime coaching section failed.",
    ),
    true,
  );
  for (const event of errors) {
    assert.deepEqual(Object.keys(event.data).sort(), [
      "error",
      "section",
      "timingMs",
    ]);
    assert.deepEqual(Object.keys(event.data.error).sort(), ["code", "message"]);
  }
  assert.equal(events.at(-1)?.event, "done");
});

test("precompute failure rejects before SSE headers are sent", async () => {
  const response = new FakeResponse();
  let runtimeCalls = 0;

  await assert.rejects(
    streamSpellingCoach(
      createRequest(),
      response as unknown as ServerResponse,
      { ...baseRequest, childAttempt: "about" },
      {
        precompute: async () => {
          throw new Error("precompute failed");
        },
        runRuntime: async () => {
          runtimeCalls += 1;
          return runtimeOutput;
        },
        requestId: () => "request-precompute-failure",
      },
    ),
    /precompute failed/,
  );

  assert.equal(runtimeCalls, 0);
  assert.equal(response.statusCode, 200);
  assert.equal(response.headers["Content-Type"], "text/event-stream");
  assert.equal(response.chunks.length, 1); // meta event was emitted
  assert.equal(response.flushCount, 1);
});

test("precompute timeout aborts stalled work before LLM starts", async () => {
  const response = new FakeResponse();
  let precomputeSignal: AbortSignal | undefined;

  await assert.rejects(
    streamSpellingCoach(
      createRequest(),
      response as unknown as ServerResponse,
      { ...baseRequest, childAttempt: "about" },
      {
        precompute: async (_input, signal) => {
          precomputeSignal = signal;
          return new Promise<WordTeachingPrecompute>(() => {});
        },
        requestId: () => "request-precompute-timeout",
        precomputeTimeoutMs: 5,
      },
    ),
    /Precompute timed out after 5ms/,
  );

  assert.equal(precomputeSignal?.aborted, true);
  assert.equal(response.statusCode, 200);
  assert.equal(response.headers["Content-Type"], "text/event-stream");
  assert.equal(response.chunks.length, 1); // meta event was emitted
  assert.equal(response.flushCount, 1);
});

test("first marker timeout emits a section error and aborts runtime", async () => {
  const response = new FakeResponse();
  let runtimeSignal: AbortSignal | undefined;

  await streamSpellingCoach(
    createRequest(),
    response as unknown as ServerResponse,
    baseRequest,
    {
      precompute: async () => precomputed,
      runRuntime: async (_input, signal) => {
        runtimeSignal = signal;
        return new Promise<SpellingCoachOutput>(() => {});
      },
      requestId: () => "request-timeout",
      firstMarkerTimeoutMs: 5,
      runtimeTimeoutMs: 50,
    },
  );

  const events = parseEvents(response);
  const errors = events.filter(({ event }) => event === "section-error");
  assert.deepEqual(events.map(({ event }) => event), [
    "meta",
    "precomputed",
    "section-error",
    "done",
  ]);
  assert.equal(errors.length, 1);
  assert.equal(errors[0].data.section, "short_feedback");
  assert.equal(errors[0].data.error.code, "FIRST_MARKER_TIMEOUT");
  assert.equal(runtimeSignal?.aborted, true);
  assert.equal(events.at(-1)?.event, "done");
});

test("section timeout aborts a stream that stalls inside an open section", async () => {
  const response = new FakeResponse();
  let runtimeSignal: AbortSignal | undefined;

  await streamSpellingCoach(
    createRequest(),
    response as unknown as ServerResponse,
    baseRequest,
    {
      precompute: async () => precomputed,
      runRuntimeStream: async (_input, _precomputed, signal) => {
        runtimeSignal = signal;
        return (async function* () {
          yield "[[SHORT_FEEDBACK]]Good try.[[END_SECTION]]";
          yield "[[MISS_ANALYSIS]]";
          await new Promise<void>(() => {});
        })();
      },
      requestId: () => "request-section-timeout",
      firstMarkerTimeoutMs: 50,
      sectionTimeoutMs: 5,
      runtimeTimeoutMs: 50,
    },
  );

  const events = parseEvents(response);
  assert.deepEqual(events.map(({ event }) => event), [
    "meta",
    "precomputed",
    "section-start",
    "section-chunk",
    "section-complete",
    "section-start",
    "section-error",
    "done",
  ]);
  const error = events.find(({ event }) => event === "section-error");
  assert.equal(error?.data.section, "miss_analysis");
  assert.equal(error?.data.error.code, "SECTION_TIMEOUT");
  assert.equal(runtimeSignal?.aborted, true);
});

test("runtime timeout aborts a stream before all required sections finish", async () => {
  const response = new FakeResponse();

  await streamSpellingCoach(
    createRequest(),
    response as unknown as ServerResponse,
    baseRequest,
    {
      precompute: async () => precomputed,
      runRuntimeStream: async () =>
        (async function* () {
          yield "[[SHORT_FEEDBACK]]Good try.[[END_SECTION]]";
          yield "[[MISS_ANALYSIS]]Some text";
          await new Promise<void>(() => {});
        })(),
      requestId: () => "request-runtime-timeout",
      firstMarkerTimeoutMs: 50,
      sectionTimeoutMs: 50,
      runtimeTimeoutMs: 5,
    },
  );

  const error = parseEvents(response).find(({ event }) => event === "section-error");
  assert.equal(error?.data.section, "miss_analysis");
  assert.equal(error?.data.error.code, "RUNTIME_TIMEOUT");
});

test("client disconnect aborts runtime and suppresses further events", async () => {
  const request = createRequest();
  const response = new FakeResponse();
  let runtimeSignal: AbortSignal | undefined;

  await streamSpellingCoach(
    request,
    response as unknown as ServerResponse,
    baseRequest,
    {
      precompute: async () => precomputed,
      runRuntime: async (_input, signal) => {
        runtimeSignal = signal;
        queueMicrotask(() => response.emit("close"));
        return new Promise<SpellingCoachOutput>((_resolve, reject) => {
          signal.addEventListener("abort", () => reject(signal.reason), {
            once: true,
          });
        });
      },
      requestId: () => "request-disconnect",
    },
  );

  assert.equal(runtimeSignal?.aborted, true);
  assert.deepEqual(parseEvents(response).map(({ event }) => event), [
    "meta",
    "precomputed",
  ]);
  assert.equal(response.listenerCount("close"), 0);
  assert.equal(response.listenerCount("error"), 0);
});

test("response errors cancel runtime and clean up connection listeners", async () => {
  const request = createRequest();
  const response = new FakeResponse();
  let runtimeSignal: AbortSignal | undefined;

  await streamSpellingCoach(
    request,
    response as unknown as ServerResponse,
    baseRequest,
    {
      precompute: async () => precomputed,
      runRuntime: async (_input, signal) => {
        runtimeSignal = signal;
        queueMicrotask(() => response.emit("error", new Error("socket failed")));
        return new Promise<SpellingCoachOutput>((_resolve, reject) => {
          signal.addEventListener("abort", () => reject(signal.reason), {
            once: true,
          });
        });
      },
      requestId: () => "request-response-error",
    },
  );

  assert.equal(runtimeSignal?.aborted, true);
  assert.deepEqual(parseEvents(response).map(({ event }) => event), [
    "meta",
    "precomputed",
  ]);
  assert.equal(request.listenerCount("aborted"), 0);
  assert.equal(response.listenerCount("close"), 0);
  assert.equal(response.listenerCount("error"), 0);
});

test("runtime cancellation signal reaches precompute and coaching model calls", async () => {
  const previousFlag = process.env.SPELLING_COACH_RUNTIME_CONCEPT_TEACHING;
  process.env.SPELLING_COACH_RUNTIME_CONCEPT_TEACHING = "on";
  const controller = new AbortController();
  const observedSignals: Array<AbortSignal | undefined> = [];
  const logLines: string[] = [];
  const originalConsoleLog = console.log;
  let invocation = 0;
  const directModel: DirectModelLike = {
    async invoke(_input, options?: { signal?: AbortSignal }) {
      observedSignals.push(options?.signal);
      invocation += 1;
      if (invocation === 1) {
        return JSON.stringify({
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
        });
      }
      return JSON.stringify({
        shortFeedback: "Good try.",
        sayAloudTip: "Say a-bout.",
      });
    },
  };

  console.log = (...args: unknown[]) => {
    logLines.push(args.map(String).join(" "));
  };
  try {
    await runSpellingCoachAgent(
      buildStreamingSpellingCoachInput(
        SpellingCoachStreamRequestSchema.parse(baseRequest),
      ),
      {
        directModel,
        runtime: "direct",
        signal: controller.signal,
        requestId: "request-runtime-timing",
      },
    );
  } finally {
    console.log = originalConsoleLog;
    if (previousFlag === undefined) {
      delete process.env.SPELLING_COACH_RUNTIME_CONCEPT_TEACHING;
    } else {
      process.env.SPELLING_COACH_RUNTIME_CONCEPT_TEACHING = previousFlag;
    }
  }

  assert.equal(observedSignals.length, 2);
  assert.equal(
    observedSignals.every((signal) => signal === controller.signal),
    true,
  );
  assert.equal(
    logLines.some((line) =>
      line.includes(
        '[spelling-coach timing] requestId=request-runtime-timing word="about"',
      ),
    ),
    true,
  );
});

test("split runtime timing logs include the stream request id", async () => {
  const previousFlag = process.env.SPELLING_COACH_RUNTIME_CONCEPT_TEACHING;
  process.env.SPELLING_COACH_RUNTIME_CONCEPT_TEACHING = "off";
  const logLines: string[] = [];
  const originalConsoleLog = console.log;
  const {
    wordTeaching: _wordTeaching,
    wordBreakdown: _wordBreakdown,
    conceptLabels: _conceptLabels,
    ...missOnlyOutput
  } = runtimeOutput;
  const directModel: DirectModelLike = {
    async invoke() {
      return JSON.stringify(missOnlyOutput);
    },
  };

  console.log = (...args: unknown[]) => {
    logLines.push(args.map(String).join(" "));
  };
  try {
    await runSplitSpellingCoachAgent(
      buildStreamingSpellingCoachInput(
        SpellingCoachStreamRequestSchema.parse({
          ...baseRequest,
          targetWord: "aardvark",
          childAttempt: "aardvarkk",
          level: 2,
        }),
      ),
      {
        directModel,
        runtime: "direct",
        requestId: "request-split-timing",
      },
    );
  } finally {
    console.log = originalConsoleLog;
    if (previousFlag === undefined) {
      delete process.env.SPELLING_COACH_RUNTIME_CONCEPT_TEACHING;
    } else {
      process.env.SPELLING_COACH_RUNTIME_CONCEPT_TEACHING = previousFlag;
    }
  }

  assert.equal(
    logLines.some((line) =>
      line.includes(
        '[spelling-coach split timing] requestId=request-split-timing word="aardvark"',
      ),
    ),
    true,
  );
});
