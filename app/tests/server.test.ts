import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import type { IncomingMessage, ServerResponse } from "node:http";
import test, { mock } from "node:test";

process.env.VERCEL = "1";
process.env.SUPABASE_URL = "https://server-test.supabase.co";
process.env.SUPABASE_PUBLISHABLE_KEY = "server-key";
delete process.env.LANGFUSE_PUBLIC_KEY;
delete process.env.LANGFUSE_SECRET_KEY;
delete process.env.SENTRY_DSN;

mock.module("@sentry/node", { namedExports: { init: () => undefined, captureException: () => undefined } });
mock.module("@supabase/supabase-js", {
  namedExports: { createClient: () => ({ from: () => { throw new Error("unexpected database call"); } }) },
});
mock.module("@opentelemetry/sdk-node", { namedExports: { NodeSDK: class { start() {} } } });
mock.module("@langfuse/otel", { namedExports: { LangfuseSpanProcessor: class {} } });
mock.module("openai", {
  defaultExport: class {
    audio = { speech: { create: async () => ({ arrayBuffer: async () => Uint8Array.from([1, 2]).buffer }) } };
  },
});

const { default: handler } = await import("../server.js");
const { default: apiHandler } = await import("../../api/index.js");

class Response extends EventEmitter {
  statusCode = 200;
  headersSent = false;
  writableEnded = false;
  destroyed = false;
  headers: Record<string, string | number> = {};
  body: Buffer<ArrayBufferLike> = Buffer.alloc(0);
  writeHead(status: number, headers: Record<string, string | number>) {
    this.statusCode = status;
    this.headers = headers;
    this.headersSent = true;
    return this;
  }
  end(value?: string | Buffer) {
    this.writableEnded = true;
    if (value) this.body = Buffer.isBuffer(value) ? value : Buffer.from(value);
    this.emit("finish");
    return this;
  }
}

function request(method?: string, url?: string, authorization?: string) {
  const req = new EventEmitter() as IncomingMessage;
  req.method = method;
  req.url = url;
  req.headers = authorization ? { authorization } : {};
  return req;
}

async function call(method?: string, url?: string, authorization?: string, body?: unknown) {
  const req = request(method, url, authorization);
  const response = new Response();
  const pending = handler(req, response as unknown as ServerResponse);
  if (body !== undefined) {
    req.emit("data", Buffer.from(JSON.stringify(body)));
    req.emit("end");
  }
  await pending;
  const json = response.body.length && String(response.headers["Content-Type"] ?? "").includes("application/json")
    ? JSON.parse(response.body.toString())
    : null;
  return { response, json };
}

test("server handles invalid requests, CORS preflight, health, and unknown routes", async () => {
  assert.equal(apiHandler, handler);
  assert.equal((await call()).response.statusCode, 400);
  const options = await call("OPTIONS", "/api/anything");
  assert.equal(options.response.statusCode, 204);
  assert.equal(options.response.headers["Access-Control-Allow-Origin"], "*");

  const health = await call("GET", "/api/health");
  assert.equal(health.response.statusCode, 200);
  assert.equal(health.json.ok, true);
  assert.equal(typeof health.json.featureFlags, "object");

  const missing = await call("DELETE", "/api/unknown");
  assert.equal(missing.response.statusCode, 404);
  assert.equal(missing.json.error, "Not found.");
});

test("server maps authentication failures to HTTP 401", async () => {
  const result = await call("GET", "/api/auth/me");
  assert.equal(result.response.statusCode, 401);
  assert.match(result.json.error, /missing bearer token/);

  const upload = await call("POST", "/api/words/import-file", undefined, {});
  assert.equal(upload.response.statusCode, 401);
  assert.match(upload.json.error, /missing bearer token/);

  const job = await call("GET", "/api/words/import-jobs/not-mine");
  assert.equal(job.response.statusCode, 401);
});

test("server exposes catalog, voice, and validation routes", async () => {
  const origins = await call("GET", "/api/foreign-origins");
  assert.equal(origins.response.statusCode, 200);
  assert.equal(Array.isArray(origins.json.origins), true);

  const unknownOrigin = await call("GET", "/api/foreign-origins/not-a-real-origin");
  assert.equal(unknownOrigin.response.statusCode, 404);
  const capabilities = await call("GET", "/api/voice/capabilities");
  assert.equal(capabilities.json.spellingBehavior.shouldAutoSubmit, false);

  const interpreted = await call("POST", "/api/voice/interpret", undefined, {
    targetWord: "friend", utterance: "f r i e n d",
  });
  assert.equal(interpreted.response.statusCode, 200);
  assert.equal(interpreted.json.intent, "spelling_attempt");

  const invalid = await call("POST", "/api/voice/interpret", undefined, {});
  assert.equal(invalid.response.statusCode, 400);

  // Secure pronunciation route: requires a valid bearer token resolved via Supabase.
  // No auth header is provided so authenticateRequest throws "Unauthorized: missing bearer token." → 401.
  const secureAudio = await call("GET", "/api/words/pronunciation?challengeId=test-id&sessionId=test-session");
  assert.equal(secureAudio.response.statusCode, 401);

  // The legacy GET /api/words/:word/pronunciation route has been removed.
  // It should now return 404.
  process.env.OPENAI_API_KEY = "mock-key";
  const legacyAudio = await call("GET", "/api/words/unitword/pronunciation");
  assert.equal(legacyAudio.response.statusCode, 404);
});
