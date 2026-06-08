import { z } from "zod";
import {
  createDirectSpellingCoachModel,
  buildDirectRuntimeSystemPrompt,
  type DirectModelLike,
} from "./directModel.js";
import {
  type WordEntry,
  createCustomWordList,
  getBaseWordByText,
  getCustomWordListById,
  loadCustomWordLists,
  saveCustomWordLists,
  type CustomWordList,
} from "./wordCatalog.js";

export const CustomWordImportRequestSchema = z
  .object({
    listName: z.string().min(1),
    words: z.union([z.array(z.string().min(1)), z.string().min(1)]),
    overwriteList: z.boolean().optional(),
    listId: z.string().optional(),
  })
  .strict();

const GeneratedWordMetadataSchema = z.object({
  word: z.string(),
  definition: z.string(),
  origin: z.string(),
  exampleSentence: z.string(),
  partOfSpeech: z.string(),
});

const GeneratedWordMetadataListSchema = z.object({
  entries: z.array(GeneratedWordMetadataSchema),
});

export type CustomWordImportRequest = z.infer<typeof CustomWordImportRequestSchema>;
type GeneratedWordMetadata = z.infer<typeof GeneratedWordMetadataSchema>;

export type ImportCustomWordsResult = {
  list: {
    id: string;
    name: string;
    wordCount: number;
  };
  importedCount: number;
  skippedExistingCount: number;
  words: WordEntry[];
};

export type GenerateCustomWordMetadataFn = (
  words: string[],
  level: "custom",
) => Promise<GeneratedWordMetadata[]>;

export type ImportCustomWordsOptions = {
  model?: string | object;
  generateMetadata?: GenerateCustomWordMetadataFn;
  ownerUserId?: string;
  existingList?: CustomWordList;
  skipFileSave?: boolean;
};

function extractTextContent(content: unknown): string {
  if (typeof content === "string") {
    return content;
  }

  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (typeof part === "string") {
          return part;
        }

        if (
          part &&
          typeof part === "object" &&
          "type" in part &&
          (part as { type?: unknown }).type === "text" &&
          "text" in part
        ) {
          return String((part as { text: unknown }).text);
        }

        return "";
      })
      .join("");
  }

  return "";
}

function extractAssistantPayload(result: unknown): string {
  if (typeof result === "string") {
    return result.trim();
  }

  if (result && typeof result === "object") {
    const maybeContent = (result as { content?: unknown }).content;
    if (maybeContent !== undefined) {
      return extractTextContent(maybeContent).trim();
    }

    const messages = (result as { messages?: unknown }).messages;
    if (Array.isArray(messages) && messages.length > 0) {
      const lastMessage = messages[messages.length - 1] as { content?: unknown };
      return extractTextContent(lastMessage?.content).trim();
    }
  }

  throw new Error("Import response did not contain assistant text content.");
}

function parseStrictJson(payload: string): unknown {
  const trimmed = payload.trim();
  if (!trimmed.startsWith("{") || !trimmed.endsWith("}")) {
    throw new Error("Import output must be a single JSON object with no wrapper text.");
  }

  return JSON.parse(trimmed);
}

function normalizeImportedWords(words: string[] | string): string[] {
  const values = Array.isArray(words) ? words : words.split(/\r?\n|,/);
  const seen = new Set<string>();
  const normalized: string[] = [];

  for (const value of values) {
    const trimmed = value.trim();
    if (!trimmed) {
      continue;
    }

    const key = trimmed.toLowerCase();
    if (seen.has(key)) {
      continue;
    }

    seen.add(key);
    normalized.push(trimmed);
  }

  return normalized;
}

function getDefaultGradeBand(): string {
  return "custom";
}

function toWordEntry(
  metadata: GeneratedWordMetadata,
): WordEntry {
  return {
    word: metadata.word,
    level: "custom",
    grade_band: getDefaultGradeBand(),
    difficulty: "custom",
    origin: metadata.origin,
    definition: metadata.definition,
    example_sentence: metadata.exampleSentence,
    patterns: [],
    common_mistakes: [],
    coach_tip: "",
    part_of_speech: metadata.partOfSpeech,
  };
}

function buildCustomWordImportPrompt(words: string[]): string {
  return [
    "Generate child-friendly metadata for a custom spelling word list.",
    "Return one JSON object only.",
    "Do not add markdown or explanation.",
    "For each word, generate:",
    "- definition",
    "- origin",
    "- exampleSentence",
    "- partOfSpeech",
    "Requirements:",
    "- Definitions must be concise and clear for children.",
    "- Example sentences must use the word naturally.",
    "- Example sentences and definitions may contain the word; the backend will mask child-facing text later.",
    "- Origins can be broad but should be accurate enough for teaching, such as Latin, Greek, French, English, Sanskrit, Arabic, or place-name origin.",
    "- partOfSpeech should be a simple label like noun, verb, adjective, or adverb.",
    "- Preserve the original spelling and casing of each input word in the returned word field.",
    "- Include exactly one entry for each requested word.",
    "Output schema:",
    JSON.stringify(
      {
        entries: [
          {
            word: "string",
            definition: "string",
            origin: "string",
            exampleSentence: "string",
            partOfSpeech: "string",
          },
        ],
      },
      null,
      2,
    ),
    "Words:",
    JSON.stringify(words, null, 2),
  ].join("\n\n");
}

async function generateCustomWordMetadataWithModel(
  words: string[],
  model?: string | object,
): Promise<GeneratedWordMetadata[]> {
  const directModel: DirectModelLike = await createDirectSpellingCoachModel({
    model,
  });
  const response = await directModel.invoke([
    {
      role: "system",
      content: `${buildDirectRuntimeSystemPrompt()}

You are helping generate metadata for custom spelling words.
Focus only on accurate word metadata generation.`,
    },
    {
      role: "user",
      content: buildCustomWordImportPrompt(words),
    },
  ]);

  const payload = extractAssistantPayload(response);
  const parsed = parseStrictJson(payload);
  return GeneratedWordMetadataListSchema.parse(parsed).entries;
}

function toListSummary(list: CustomWordList) {
  return {
    id: list.id,
    name: list.name,
    wordCount: list.words.length,
  };
}

export async function importCustomWords(
  request: CustomWordImportRequest,
  options: ImportCustomWordsOptions = {},
): Promise<ImportCustomWordsResult> {
  const parsedRequest = CustomWordImportRequestSchema.parse(request);
  const words = normalizeImportedWords(parsedRequest.words);

  if (words.length === 0) {
    throw new Error("At least one custom word is required.");
  }

  const ownerUserId = options.ownerUserId?.trim() ?? "legacy";
  const existingList = options.existingList ?? (parsedRequest.listId
    ? getCustomWordListById(parsedRequest.listId, ownerUserId)
    : loadCustomWordLists().find(
        (list) =>
          list.owner_user_id === ownerUserId &&
          list.name.trim().toLowerCase() === parsedRequest.listName.trim().toLowerCase(),
      ));
  if (parsedRequest.listId && !existingList) {
    throw new Error(`Unknown custom list: ${parsedRequest.listId}`);
  }

  const currentListWords = existingList?.words ?? [];
  const existingByWord = new Map(
    currentListWords.map((entry) => [entry.word.toLowerCase(), entry]),
  );

  const wordsToGenerate =
    parsedRequest.overwriteList || !existingList
      ? words
      : words.filter((word) => !existingByWord.has(word.toLowerCase()));

  const reusedWordEntries: WordEntry[] = [];
  const wordsStillToGenerate: string[] = [];

  for (const word of wordsToGenerate) {
    const baseEntry = getBaseWordByText(word);
    if (baseEntry) {
      reusedWordEntries.push(baseEntry);
    } else {
      wordsStillToGenerate.push(word);
    }
  }

  const generatedEntries = wordsStillToGenerate.length === 0
    ? []
    : await (options.generateMetadata ?? ((items) =>
        generateCustomWordMetadataWithModel(items, options.model)))(
        wordsStillToGenerate,
        "custom",
      );

  const importedWordEntries = [
    ...reusedWordEntries,
    ...generatedEntries.map((entry) => toWordEntry(entry)),
  ];

  const mergedWords = parsedRequest.overwriteList || !existingList
    ? importedWordEntries
    : [
        ...currentListWords,
        ...importedWordEntries.filter(
          (entry) => !existingByWord.has(entry.word.toLowerCase()),
        ),
      ];

  const nextList = existingList
    ? {
        ...existingList,
        name: parsedRequest.listName,
        words: mergedWords,
      }
    : createCustomWordList(parsedRequest.listName, mergedWords, ownerUserId);

  if (!options.skipFileSave) {
    const allLists = loadCustomWordLists();
    const updatedLists = existingList
      ? allLists.map((list) => (list.id === nextList.id ? nextList : list))
      : [...allLists, nextList];
    saveCustomWordLists(updatedLists);
  }

  return {
    list: toListSummary(nextList),
    importedCount: importedWordEntries.length,
    skippedExistingCount:
      existingList && !parsedRequest.overwriteList
        ? words.length - wordsToGenerate.length
        : 0,
    words: importedWordEntries,
  };
}
