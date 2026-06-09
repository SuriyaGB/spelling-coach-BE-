import { createClient } from "@supabase/supabase-js";
// @ts-ignore
import ws from "ws";
import { type WordEntry } from "./wordCatalog.js";

// Polyfill WebSocket support globally for Node.js < 22
global.WebSocket = ws as any;

const supabaseUrl = process.env.SUPABASE_URL?.trim();
const supabaseKey = process.env.SUPABASE_PUBLISHABLE_KEY?.trim();

function checkSupabaseConfig() {
  if (!supabaseUrl || !supabaseKey) {
    throw new Error("Supabase is not configured. Set SUPABASE_URL and SUPABASE_PUBLISHABLE_KEY.");
  }
}

export const supabase = supabaseUrl && supabaseKey
  ? createClient(supabaseUrl, supabaseKey, {
      auth: {
        persistSession: false,
      },
    })
  : null as any;

/**
 * Creates a Supabase client that acts on behalf of the logged-in user.
 * This ensures that Row Level Security (RLS) is applied correctly.
 */
export function getSupabaseUserClient(authToken: string) {
  checkSupabaseConfig();
  const cleanToken = authToken.replace(/^bearer\s+/i, "").trim();
  return createClient(supabaseUrl!, supabaseKey!, {
    auth: {
      persistSession: false,
    },
    global: {
      headers: {
        Authorization: `Bearer ${cleanToken}`,
      },
    },
  });
}

export interface DBCustomList {
  id: string;
  name: string;
  owner_user_id: string;
  words: WordEntry[];
  created_at?: string;
}

/**
 * Fetch all custom word lists for a user from Supabase.
 */
export async function fetchCustomListsFromDB(authToken: string, userId: string) {
  const userClient = getSupabaseUserClient(authToken);
  const { data, error } = await userClient
    .from("custom_word_lists")
    .select("id, name, words")
    .eq("owner_user_id", userId);

  if (error) {
    throw error;
  }

  return (data || []).map((list: any) => ({
    id: list.id as string,
    name: list.name as string,
    wordCount: Array.isArray(list.words) ? list.words.length : 0,
  }));
}

/**
 * Fetch a specific custom word list by ID from Supabase.
 */
export async function fetchCustomListByIdFromDB(authToken: string, listId: string, userId: string) {
  const userClient = getSupabaseUserClient(authToken);
  const { data, error } = await userClient
    .from("custom_word_lists")
    .select("id, name, words, owner_user_id")
    .eq("id", listId)
    .eq("owner_user_id", userId)
    .maybeSingle();

  if (error) {
    throw error;
  }

  return data as DBCustomList | null;
}

/**
 * Upsert (save/update) a custom word list in Supabase.
 */
export async function saveCustomListToDB(
  authToken: string,
  userId: string,
  name: string,
  words: WordEntry[],
  listId?: string,
) {
  const userClient = getSupabaseUserClient(authToken);

  const payload: Partial<DBCustomList> = {
    name,
    owner_user_id: userId,
    words,
  };

  if (listId) {
    payload.id = listId;
  }

  const { data, error } = await userClient
    .from("custom_word_lists")
    .upsert(payload)
    .select("id, name, words")
    .single();

  if (error) {
    throw error;
  }

  return data as DBCustomList;
}
