import { SPELLING_COACH_SYSTEM_PROMPT } from "./prompt.js";
import { DEFAULT_MODEL_NAME, getConfiguredModelName, getOpenAITemperature } from "./modelConfig.js";

export type DeepAgentLike = {
  invoke(input: unknown, options?: any): Promise<unknown>;
};

export type CreateSpellingCoachAgentOptions = {
  model?: string | object;
};

export type CreateDeepAgentFn = typeof import("deepagents")["createDeepAgent"];
type ChatAnthropicCtor = typeof import("@langchain/anthropic")["ChatAnthropic"];
type ChatOpenAICtor = typeof import("@langchain/openai")["ChatOpenAI"];

let createDeepAgentCache: Promise<CreateDeepAgentFn> | null = null;
let chatAnthropicCache: Promise<ChatAnthropicCtor> | null = null;
let chatOpenAICache: Promise<ChatOpenAICtor> | null = null;
const agentCache = new Map<string, Promise<DeepAgentLike>>();

export async function getCreateDeepAgent(): Promise<CreateDeepAgentFn> {
  if (!createDeepAgentCache) {
    createDeepAgentCache = import("deepagents").then(
      (deepagents) => deepagents.createDeepAgent,
    );
  }

  return createDeepAgentCache;
}

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

async function resolveModel(model?: string | object): Promise<string | object> {
  if (model && typeof model === "object") {
    return model;
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
  });
}

export async function createSpellingCoachAgent(
  options: CreateSpellingCoachAgentOptions = {},
): Promise<DeepAgentLike> {
  if (typeof options.model === "string" || options.model === undefined) {
    const cacheKey = options.model ?? getConfiguredModelName();
    const cachedAgent = agentCache.get(cacheKey);
    if (cachedAgent) {
      return cachedAgent;
    }

    const agentPromise = createSpellingCoachAgentInternal(options);
    agentCache.set(cacheKey, agentPromise);
    return agentPromise;
  }

  return createSpellingCoachAgentInternal(options);
}

async function createSpellingCoachAgentInternal(
  options: CreateSpellingCoachAgentOptions = {},
): Promise<DeepAgentLike> {
  const model = await resolveModel(options.model);
  const create_deep_agent = await getCreateDeepAgent();

  return create_deep_agent({
    model: model as any,
    tools: [],
    subagents: [],
    instructions: `${SPELLING_COACH_SYSTEM_PROMPT}

Architecture constraints:
- No knowledge base lookups.
- No retrieval.
- No persistence.
- No session orchestration.
- Focus only on spelling miss analysis and coaching output.`,
  }) as DeepAgentLike;
}