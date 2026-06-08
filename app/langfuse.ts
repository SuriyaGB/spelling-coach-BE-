import { propagateAttributes, startObservation } from "@langfuse/tracing";
import type { AuthenticatedUser } from "./auth.js";

export interface TraceOptions {
    input: any;
    output: any;
    latencyMs: number;
}

export async function recordSpellingCoachTrace(options: TraceOptions) {
    console.log("[LANGFUSE] recordSpellingCoachTrace called.");

    const { input, output, latencyMs } = options;

    try {
        const traceName = "spelling-coach-test";
        const userId = input.childProfile?.childId || "anonymous-child";

        // session ID grouping traces by child and mode
        const sessionId = input.childProfile?.childId
            ? `${input.childProfile.childId}-${input.sessionContext?.mode || "practice"}`
            : undefined;

        // Convert metadata values to strings as required by PropagateAttributesParams
        const metadata: Record<string, string> = {
            latencyMs: String(latencyMs),
        };
        if (input.childProfile?.age !== undefined) metadata.childAge = String(input.childProfile.age);
        if (input.childProfile?.grade !== undefined) metadata.childGrade = String(input.childProfile.grade);
        if (input.childProfile?.spellingLevel !== undefined) metadata.spellingLevel = String(input.childProfile.spellingLevel);

        console.log("[LANGFUSE] Sending trace via OpenTelemetry:", JSON.stringify({
            traceName,
            userId,
            sessionId,
        }));

        await propagateAttributes(
            {
                userId,
                sessionId,
                tags: [process.env.NODE_ENV || "development"],
                traceName,
                metadata,
            },
            async () => {
                const span = startObservation("spelling-coach-span", {
                    input: {
                        targetWord: input.targetWord,
                        childAttempt: input.childAttempt,
                        childProfile: input.childProfile,
                        sessionContext: input.sessionContext,
                    },
                    output: output,
                });

                // End the observation span immediately since we record post-execution
                span.end();
            }
        );

        console.log("[LANGFUSE] Manual trace recorded successfully!");
    } catch (err) {
        console.error("[LANGFUSE] Failed to record manual trace:", err);
    }
}

export interface ImportListTraceOptions {
    user: AuthenticatedUser;
    listName: string;
    wordCount: number;
    latencyMs: number;
    inputWords: string[];
    outputWords: any[];
}

export async function recordImportListTrace(options: ImportListTraceOptions) {
    console.log("[LANGFUSE] recordImportListTrace called.");
    const { user, listName, wordCount, latencyMs, inputWords, outputWords } = options;

    try {
        const traceName = "import-custom-list";
        const userId = user.email || user.id;

        // Convert metadata values to strings
        const metadata: Record<string, string> = {
            listName,
            wordCount: String(wordCount),
            latencyMs: String(latencyMs),
        };
        if (inputWords && inputWords.length > 0) {
            metadata.words = inputWords.join(", ");
        }

        console.log("[LANGFUSE] Sending import trace via OpenTelemetry:", JSON.stringify({
            traceName,
            userId,
        }));

        await propagateAttributes(
            {
                userId,
                tags: [process.env.NODE_ENV || "development"],
                traceName,
                metadata,
            },
            async () => {
                const span = startObservation("import-list-span", {
                    input: { listName, wordCount, words: inputWords },
                    output: { success: true, listName, wordCount, importedWords: outputWords },
                });

                span.end();
            }
        );

        console.log("[LANGFUSE] Import trace recorded successfully!");
    } catch (err) {
        console.error("[LANGFUSE] Failed to record import trace:", err);
    }
}

export function getLangfuseHandler(): undefined {
    return undefined;
}
