import "dotenv/config";
import * as Sentry from "@sentry/node";
import { NodeSDK } from "@opentelemetry/sdk-node";
import { LangfuseSpanProcessor } from "@langfuse/otel";

if (process.env.SENTRY_DSN) {
  Sentry.init({
    dsn: process.env.SENTRY_DSN,
    sendDefaultPii: true,
  });
}

if (process.env.LANGFUSE_PUBLIC_KEY && process.env.LANGFUSE_SECRET_KEY) {
  console.log("[LANGFUSE OTEL] Initializing LangfuseSpanProcessor...");
  const langfuseSpanProcessor = new LangfuseSpanProcessor({
    publicKey: process.env.LANGFUSE_PUBLIC_KEY,
    secretKey: process.env.LANGFUSE_SECRET_KEY,
    baseUrl: process.env.LANGFUSE_BASE_URL || "https://us.cloud.langfuse.com",
  });

  const sdk = new NodeSDK({
    spanProcessor: langfuseSpanProcessor,
  });

  sdk.start();
  console.log("[LANGFUSE OTEL] OpenTelemetry SDK started.");
} else {
  console.warn("[LANGFUSE OTEL] Missing keys, OpenTelemetry SDK not started.");
}
