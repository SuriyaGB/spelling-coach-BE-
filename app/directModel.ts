import { SPELLING_COACH_SYSTEM_PROMPT } from "./prompt.js";
import { DEFAULT_MODEL_NAME, getConfiguredModelName, getOpenAITemperature } from "./modelConfig.js";

export type DirectModelLike = {
  invoke(input: unknown, options?: any): Promise<unknown>;
};

type ChatAnthropicCtor = typeof import("@langchain/anthropic")["ChatAnthropic"];
type ChatOpenAICtor = typeof import("@langchain/openai")["ChatOpenAI"];

let chatAnthropicCache: Promise<ChatAnthropicCtor> | null = null;
let chatOpenAICache: Promise<ChatOpenAICtor> | null = null;
const directModelCache = new Map<string, Promise<DirectModelLike>>();

async function getChatAnthropic(): Promise<ChatAnthropicCtor> {
  if (!chatAnthropicCache) {
    chatAnthropicCache = import("@langchain/anthropic").then(
      (anthropic) => anthropic.ChatAnthropic,
    );
  }

  return chatAnthropicCache;
}

async function getChatOpenAI(): Promise<ChatOpenAICtor> {
  if (!chatOpenAICache) {
    chatOpenAICache = import("@langchain/openai").then(
      (openai) => openai.ChatOpenAI,
    );
  }

  return chatOpenAICache;
}

export type CreateDirectSpellingCoachModelOptions = {
  model?: string | object;
};

async function resolveDirectModel(model?: string | object): Promise<DirectModelLike> {
  if (model && typeof model === "object") {
    return model as DirectModelLike;
  }

  const modelName = model ?? getConfiguredModelName();

  if (modelName.startsWith("openai:")) {
    const ChatOpenAI = await getChatOpenAI();
    const openAIModelName = modelName.slice("openai:".length);
    return new ChatOpenAI({
      apiKey: process.env.OPENAI_API_KEY,
      model: openAIModelName,
      temperature: getOpenAITemperature(modelName),
    });
  }

  const normalizedModelName = modelName.startsWith("anthropic:")
    ? modelName.slice("anthropic:".length)
    : modelName;

  const ChatAnthropic = await getChatAnthropic();
  return new ChatAnthropic({
    apiKey: process.env.ANTHROPIC_API_KEY,
    model: normalizedModelName,
    maxTokens: 4096,
    temperature: 0,
  });
}

export async function createDirectSpellingCoachModel(
  options: CreateDirectSpellingCoachModelOptions = {},
): Promise<DirectModelLike> {
  if (typeof options.model === "string" || options.model === undefined) {
    const cacheKey = options.model ?? getConfiguredModelName();
    const cachedModel = directModelCache.get(cacheKey);
    if (cachedModel) {
      return cachedModel;
    }

    const modelPromise = resolveDirectModel(options.model);
    directModelCache.set(cacheKey, modelPromise);
    return modelPromise;
  }

  return resolveDirectModel(options.model);
}

export function buildDirectRuntimeSystemPrompt(): string {
  return `${SPELLING_COACH_SYSTEM_PROMPT}

Architecture constraints:
- Use a single direct model response.
- No knowledge base lookups.
- No retrieval.
- No persistence.
- No session orchestration.
- Focus only on spelling miss analysis and coaching output.`;
}
