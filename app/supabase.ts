import { createClient } from "@supabase/supabase-js";
// @ts-ignore
import ws from "ws";
import { type WordEntry } from "./wordCatalog.js";

// Polyfill WebSocket support globally for Node.js < 22
global.WebSocket = ws as any;

const supabaseUrl = process.env.SUPABASE_URL?.trim();
const supabaseKey = process.env.SUPABASE_PUBLISHABLE_KEY?.trim();

if (!supabaseUrl || !supabaseKey) {
  throw new Error("Supabase is not configured. Set SUPABASE_URL and SUPABASE_PUBLISHABLE_KEY.");
}

const customFetch = async (url: RequestInfo | URL, options?: RequestInit): Promise<Response> => {
  let retries = 3;
  while (retries > 0) {
    try {
      return await fetch(url, options);
    } catch (e: any) {
      retries--;
      if (retries === 0) throw e;
      await new Promise(r => setTimeout(r, 1000));
    }
  }
  throw new Error("Fetch failed");
};

export const supabase = createClient(supabaseUrl, supabaseKey, {
  auth: {
    persistSession: false,
  },
  global: {
    fetch: customFetch,
  },
});

/**
 * Creates a Supabase client that acts on behalf of the logged-in user.
 * This ensures that Row Level Security (RLS) is applied correctly.
 */
export function getSupabaseUserClient(authToken: string) {
  const cleanToken = authToken.replace(/^bearer\s+/i, "").trim();
  return createClient(supabaseUrl!, supabaseKey!, {
    auth: {
      persistSession: false,
    },
    global: {
      fetch: customFetch,
      headers: {
        Authorization: `Bearer ${cleanToken}`,
      },
    },
  });
}

/**
 * Normalizes any incoming frontend mode string to one of the 6 allowed database enum values:
 * 'standard_level_1', 'standard_level_2', 'standard_level_3', 'custom', 'foreign_origin', 'mock_bee'
 */
function normalizeStandardLevel(level?: number): 1 | 2 | 3 {
  if (level == null) {
    throw new Error("level is required when mode is standard");
  }

  if (level !== 1 && level !== 2 && level !== 3) {
    throw new Error("level must be 1, 2, or 3 when mode is standard");
  }

  return level;
}

export function normalizeMode(mode: string, level?: number): string {
  if (mode === "standard") {
    return `standard_level_${normalizeStandardLevel(level)}`;
  }
  if (mode.startsWith("standard_level_")) {
    const parsedLevel = Number(mode.replace("standard_level_", ""));
    normalizeStandardLevel(parsedLevel);
    return mode;
  }
  if (mode.startsWith("custom_list_") || mode === "custom") {
    return "custom";
  }
  if (mode.startsWith("foreign_origin_") || mode === "foreign_origin" || mode === "foreignOrigin") {
    return "foreign_origin";
  }
  if (mode === "mock_bee" || mode === "mock-bee") {
    return "mock_bee";
  }
  return mode;
}

type PracticeScope = {
  dbMode: string;
  modeKey: string;
  originLanguage: string | null;
  customListId: string | null;
};

function normalizeOptionalText(value?: string | null): string | null {
  const trimmed = value?.trim();
  return trimmed ? trimmed : null;
}

function buildModeKeyFromScope(scope: {
  dbMode: string;
  originLanguage?: string | null;
  customListId?: string | null;
}): string {
  if (scope.dbMode === "foreign_origin" && scope.originLanguage) {
    return `foreign_origin_${scope.originLanguage}`;
  }

  if (scope.dbMode === "custom" && scope.customListId) {
    return `custom_list_${scope.customListId}`;
  }

  return scope.dbMode;
}

function resolvePracticeScope(
  mode: string,
  level?: number,
  options?: {
    originLanguage?: string | null;
    customListId?: string | null;
  },
): PracticeScope {
  const dbMode = normalizeMode(mode, level);

  let originLanguage = normalizeOptionalText(options?.originLanguage);
  let customListId = normalizeOptionalText(options?.customListId);

  if (!originLanguage && mode.startsWith("foreign_origin_")) {
    originLanguage = normalizeOptionalText(mode.replace("foreign_origin_", ""));
  }

  if (!customListId && mode.startsWith("custom_list_")) {
    customListId = normalizeOptionalText(mode.replace("custom_list_", ""));
  }

  return {
    dbMode,
    modeKey: buildModeKeyFromScope({
      dbMode,
      originLanguage,
      customListId,
    }),
    originLanguage,
    customListId,
  };
}

function buildModeKeyFromSessionRow(session: {
  mode: string;
  origin_language?: string | null;
  custom_list_id?: string | null;
}) {
  return buildModeKeyFromScope({
    dbMode: session.mode,
    originLanguage: session.origin_language,
    customListId: session.custom_list_id,
  });
}

export interface DBCustomList {
  id: string;
  name: string;
  owner_user_id: string;
  words: WordEntry[];
  word_count?: number;
  created_at?: string;
}

/**
 * Fetch all custom word lists for a user from Supabase.
 */
export async function fetchCustomListsFromDB(authToken: string, userId: string) {
  const userClient = getSupabaseUserClient(authToken);
  const { data, error } = await userClient
    .from("custom_word_lists")
    .select("id, name, word_count")
    .eq("owner_user_id", userId);

  if (error) {
    throw error;
  }

  return (data || []).map((list: any) => ({
    id: list.id as string,
    name: list.name as string,
    wordCount: Number(list.word_count ?? (Array.isArray(list.words) ? list.words.length : 0)) || 0,
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
    word_count: words.length,
  };

  if (listId) {
    payload.id = listId;
  }

  const { data, error } = await userClient
    .from("custom_word_lists")
    .upsert(payload)
    .select("id, name, words, word_count")
    .single();

  if (error) {
    throw error;
  }

  return data as DBCustomList;
}

export interface UserProfile {
  id: string;
  email: string | null;
  full_name: string | null;
  theme_preference: string;
  audio_enabled: boolean;
  child_id: string | null;
  age: number | null;
  grade: string | null;
  spelling_level: string | null;
}

/**
 * Fetch the user's profile from the users table. If not found, insert a default row.
 */
export async function fetchUserProfileFromDB(authToken: string, userId: string, email?: string | null) {
  const userClient = getSupabaseUserClient(authToken);
  const { data, error } = await userClient
    .from("users")
    .select("id, email, full_name, theme_preference, audio_enabled, child_id, age, grade, spelling_level")
    .eq("id", userId)
    .maybeSingle();

  if (error) {
    throw error;
  }

  if (!data) {
    const { data: inserted, error: insertError } = await userClient
      .from("users")
      .insert({
        id: userId,
        email: email ?? null,
        theme_preference: "default",
        audio_enabled: true,
        child_id: "c1",
        age: 10,
        grade: "5",
        spelling_level: "competition",
      })
      .select("id, email, full_name, theme_preference, audio_enabled, child_id, age, grade, spelling_level")
      .maybeSingle();

    if (insertError) {
      throw insertError;
    }
    return inserted;
  }

  return data;
}

/**
 * Update the user's profile details.
 */
export async function updateUserProfileInDB(
  authToken: string,
  userId: string,
  updates: Partial<Omit<UserProfile, "id" | "email">>,
) {
  const userClient = getSupabaseUserClient(authToken);
  const { data, error } = await userClient
    .from("users")
    .update(updates)
    .eq("id", userId)
    .select("id, email, full_name, theme_preference, audio_enabled, child_id, age, grade, spelling_level")
    .single();

  if (error) {
    throw error;
  }
  return data;
}

/**
 * Start a new practice session in the DB.
 */
export type StartPracticeSessionResult =
  | {
      action: "created";
      sessionId: string;
    }
  | {
      action: "resume_existing";
      sessionId: string;
    }
  | {
      action: "active_session_conflict";
      activeSessionId: string;
      activeMode: string;
    };

export type PracticeSessionStatus = "active" | "completed" | "abandoned";

function calculateSessionDurationSeconds(sessionStartedAt: string, sessionEndedAt: string) {
  const startedAtMs = new Date(sessionStartedAt).getTime();
  const endedAtMs = new Date(sessionEndedAt).getTime();

  if (Number.isNaN(startedAtMs) || Number.isNaN(endedAtMs)) {
    return 0;
  }

  return Math.max(0, Math.round((endedAtMs - startedAtMs) / 1000));
}

async function forceCloseActiveSessionInDB(
  userClient: ReturnType<typeof getSupabaseUserClient>,
  userId: string,
  activeSession: {
    id: string;
    session_started_at: string;
    total_words_attempted: number | null;
    total_correct: number | null;
  },
  status: Extract<PracticeSessionStatus, "completed" | "abandoned">,
) {
  const endedAt = new Date().toISOString();
  const durationSeconds = calculateSessionDurationSeconds(
    activeSession.session_started_at,
    endedAt,
  );

  const { error: endError } = await userClient
    .from("practice_sessions")
    .update({
      status,
      session_ended_at: endedAt,
      duration_seconds: durationSeconds,
      total_words_attempted: activeSession.total_words_attempted || 0,
      total_correct: activeSession.total_correct || 0,
    })
    .eq("id", activeSession.id)
    .eq("user_id", userId);

  if (endError) {
    throw endError;
  }
}

export async function startPracticeSessionInDB(
  authToken: string,
  userId: string,
  mode: string,
  level?: number,
  forceCloseCurrent?: boolean,
  options?: {
    originLanguage?: string | null;
    customListId?: string | null;
  },
): Promise<StartPracticeSessionResult> {
  const userClient = getSupabaseUserClient(authToken);
  const scope = resolvePracticeScope(mode, level, options);

  const { data: activeSession, error: activeSessionError } = await userClient
    .from("practice_sessions")
    .select("id, mode, origin_language, custom_list_id, session_started_at, total_words_attempted, total_correct")
    .eq("user_id", userId)
    .eq("status", "active")
    .is("session_ended_at", null)
    .order("session_started_at", { ascending: false })
    .limit(1)
    .maybeSingle();

  if (activeSessionError) {
    throw activeSessionError;
  }

  if (activeSession) {
    const activeModeKey = buildModeKeyFromSessionRow(activeSession);

    if (activeModeKey === scope.modeKey) {
      return {
        action: "resume_existing",
        sessionId: activeSession.id as string,
      };
    }

    if (!forceCloseCurrent) {
      return {
        action: "active_session_conflict",
        activeSessionId: activeSession.id as string,
        activeMode: activeModeKey,
      };
    }

    await forceCloseActiveSessionInDB(userClient, userId, {
      id: activeSession.id as string,
      session_started_at: activeSession.session_started_at as string,
      total_words_attempted: activeSession.total_words_attempted as number | null,
      total_correct: activeSession.total_correct as number | null,
    }, "abandoned");
  }

  const insertPayload: Record<string, unknown> = {
    user_id: userId,
    mode: scope.dbMode,
    status: "active",
    origin_language: scope.originLanguage,
    custom_list_id: scope.customListId,
    session_started_at: new Date().toISOString(),
  };

  let { data, error } = await userClient
    .from("practice_sessions")
    .insert(insertPayload)
    .select("id")
    .single();

  if (error && (error.code === "PGRST204" || error.message?.includes("custom_list_name"))) {
    delete insertPayload.custom_list_name;
    const retry = await userClient
      .from("practice_sessions")
      .insert(insertPayload)
      .select("id")
      .single();
    data = retry.data;
    error = retry.error;
  }

  if (error) {
    throw error;
  }

  return {
    action: "created",
    sessionId: data!.id as string,
  };
}

/**
 * Stores a challengeId -> targetWord mapping inside the session's session_state JSONB.
 * Merges safely into any existing session_state without overwriting other keys.
 */
export async function addChallengeToSession(
  authToken: string,
  userId: string,
  sessionId: string,
  challengeId: string,
  targetWord: string,
): Promise<void> {
  const userClient = getSupabaseUserClient(authToken);

  // Read the current session_state first so we can merge without overwriting
  const { data, error: fetchError } = await userClient
    .from("practice_sessions")
    .select("session_state")
    .eq("id", sessionId)
    .eq("user_id", userId)
    .single();

  if (fetchError) throw fetchError;

  const current = (data?.session_state as Record<string, unknown>) ?? {};
  const activeChallenges = ((current.activeChallenges ?? {}) as Record<string, string>);

  const { error: updateError } = await userClient
    .from("practice_sessions")
    .update({
      session_state: {
        ...current,
        activeChallenges: {
          ...activeChallenges,
          [challengeId]: targetWord,
        },
      },
    })
    .eq("id", sessionId)
    .eq("user_id", userId);

  if (updateError) throw updateError;
}

/**
 * Read-only lookup of a challengeId in the session — does NOT consume/remove the entry.
 * Used by the pronunciation route so the challenge remains available for the stream route.
 * Returns null if the challenge is not found.
 */
export async function peekChallengeInSession(
  authToken: string,
  userId: string,
  sessionId: string,
  challengeId: string,
): Promise<string | null> {
  const userClient = getSupabaseUserClient(authToken);

  const { data, error } = await userClient
    .from("practice_sessions")
    .select("session_state")
    .eq("id", sessionId)
    .eq("user_id", userId)
    .single();

  if (error || !data) return null;

  const state = (data.session_state as Record<string, unknown>) ?? {};
  const activeChallenges = (state.activeChallenges ?? {}) as Record<string, string>;
  const completedChallenges = (state.completedChallenges ?? {}) as Record<string, string>;
  
  return activeChallenges[challengeId] ?? completedChallenges[challengeId] ?? null;
}

/**
 * Resolves a challengeId to the target word and removes it from the session in one operation
 * (consume-once pattern). Returns null if the challenge is not found.
 * This prevents entry accumulation and eliminates replay attacks.
 */
export async function getChallengeFromSession(
  authToken: string,
  userId: string,
  sessionId: string,
  challengeId: string,
): Promise<string | null> {
  const userClient = getSupabaseUserClient(authToken);

  const { data, error } = await userClient
    .from("practice_sessions")
    .select("session_state")
    .eq("id", sessionId)
    .eq("user_id", userId)
    .single();

  if (error || !data) return null;

  const state = (data.session_state as Record<string, unknown>) ?? {};
  const activeChallenges = { ...(state.activeChallenges ?? {}) } as Record<string, string>;
  const completedChallenges = { ...(state.completedChallenges ?? {}) } as Record<string, string>;
  
  let targetWord = activeChallenges[challengeId];
  let wasAlreadyCompleted = false;

  if (!targetWord) {
    targetWord = completedChallenges[challengeId];
    if (targetWord) {
      wasAlreadyCompleted = true;
    } else {
      return null;
    }
  }

  if (wasAlreadyCompleted) {
    return targetWord;
  }

  // Move the consumed challenge to completedChallenges to prevent replay on attempts
  // but still allow peeking for pronunciation audio and retrying.
  delete activeChallenges[challengeId];
  completedChallenges[challengeId] = targetWord;
  
  await userClient
    .from("practice_sessions")
    .update({
      session_state: {
        ...state,
        activeChallenges,
        completedChallenges,
      },
    })
    .eq("id", sessionId)
    .eq("user_id", userId);

  return targetWord;
}

export async function recordWordAttemptInDB(
  authToken: string,
  userId: string,
  sessionId: string,
  targetWord: string,
  childAttempt: string,
  isCorrect: boolean,
  mode: string,
  level?: number,
  definitionViewed?: boolean,
  exampleViewed?: boolean,
  originViewed?: boolean,
  partOfSpeechViewed?: boolean,
  repeatWordCount?: number,
  usedVoiceInput?: boolean,
  coachingResponse?: string,
) {
  const userClient = getSupabaseUserClient(authToken);
  const scope = resolvePracticeScope(mode, level);
  const { data, error } = await userClient
    .from("word_attempts")
    .insert({
      session_id: sessionId,
      user_id: userId,
      target_word: targetWord,
      child_attempt: childAttempt,
      is_correct: isCorrect,
      level,
      definition_viewed: definitionViewed,
      example_viewed: exampleViewed,
      origin_viewed: originViewed,
      part_of_speech_viewed: partOfSpeechViewed,
      repeat_word_count: repeatWordCount || 0,
      used_voice_input: usedVoiceInput,
      coaching_response: coachingResponse,
    })
    .select("id")
    .single();

  if (error) {
    throw error;
  }

  // Update practice_sessions table in real-time
  const { data: sessionData } = await userClient
    .from("practice_sessions")
    .select("total_words_attempted, total_correct")
    .eq("id", sessionId)
    .maybeSingle();

  if (sessionData) {
    const nextWords = (sessionData.total_words_attempted || 0) + 1;
    const nextCorrect = (sessionData.total_correct || 0) + (isCorrect ? 1 : 0);
    await userClient
      .from("practice_sessions")
      .update({
        total_words_attempted: nextWords,
        total_correct: nextCorrect,
      })
      .eq("id", sessionId);
  }

  // Fetch current user_statistics for the normalized practice scope.
  let query = userClient
    .from("user_statistics")
    .select("current_streak, best_streak, total_attempts, correct_attempts")
    .eq("user_id", userId)
    .eq("mode", scope.dbMode);

  if (scope.originLanguage) {
    query = query.eq("origin_language", scope.originLanguage);
  } else {
    query = query.is("origin_language", null);
  }

  if (scope.customListId) {
    query = query.eq("custom_list_id", scope.customListId);
  } else {
    query = query.is("custom_list_id", null);
  }

  const { data: stats } = await query.maybeSingle();

  const currentStreak = stats ? (stats.current_streak || 0) : 0;
  const bestStreak = stats ? (stats.best_streak || 0) : 0;
  const totalAttempts = stats ? (stats.total_attempts || 0) : 0;
  const correctAttempts = stats ? (stats.correct_attempts || 0) : 0;

  const nextStreak = isCorrect ? currentStreak + 1 : 0;
  const nextBestStreak = Math.max(bestStreak, nextStreak);
  const nextAttempts = totalAttempts + 1;
  const nextCorrectAttempts = isCorrect ? correctAttempts + 1 : correctAttempts;

  // Calculate earned badges based on stats rules
  const badges: string[] = [];
  if (nextBestStreak >= 3) badges.push("streak3");
  if (nextBestStreak >= 5) badges.push("streak5");
  if (nextBestStreak >= 10) badges.push("streak10");
  if (nextCorrectAttempts >= 25) badges.push("total25");
  if (nextCorrectAttempts >= 50) badges.push("total50");

  const { error: statsError } = await userClient
    .from("user_statistics")
    .upsert({
      user_id: userId,
      mode: scope.dbMode,
      origin_language: scope.originLanguage,
      custom_list_id: scope.customListId,
      total_attempts: nextAttempts,
      correct_attempts: nextCorrectAttempts,
      current_streak: nextStreak,
      best_streak: nextBestStreak,
      badges: badges,
      updated_at: new Date().toISOString(),
    }, {
      onConflict: "user_id,mode,origin_language,custom_list_id"
    });

  if (statsError) {
    console.error("Failed to update user_statistics:", statsError);
  }

  return data.id as string;
}

/**
 * End a practice session in the DB.
 */
export async function endPracticeSessionInDB(
  authToken: string,
  userId: string,
  sessionId: string,
  totalWordsAttempted: number,
  totalCorrect: number,
  durationSeconds: number,
) {
  const userClient = getSupabaseUserClient(authToken);

  const { error } = await userClient
    .from("practice_sessions")
    .update({
      status: "completed",
      session_ended_at: new Date().toISOString(),
      total_words_attempted: totalWordsAttempted,
      total_correct: totalCorrect,
      duration_seconds: durationSeconds,
    })
    .eq("id", sessionId)
    .eq("user_id", userId);

  if (error) {
    throw error;
  }
}

/**
 * Fetch all statistics for a user.
 */
export async function getUserStatisticsInDB(
  authToken: string,
  userId: string,
) {
  const userClient = getSupabaseUserClient(authToken);
  const { data, error } = await userClient
    .from("user_statistics")
    .select("mode, origin_language, custom_list_id, current_streak, best_streak, total_attempts, correct_attempts, badges")
    .eq("user_id", userId);

  if (error) {
    throw error;
  }
  return data;
}

/**
 * Fetch all word attempts for a specific practice session.
 */
export async function getSessionAttemptsFromDB(
  authToken: string,
  userId: string,
  sessionId: string,
) {
  const userClient = getSupabaseUserClient(authToken);
  const { data, error } = await userClient
    .from("word_attempts")
    .select("*")
    .eq("session_id", sessionId)
    .eq("user_id", userId)
    .order("created_at", { ascending: true });

  if (error) {
    throw error;
  }
  return data;
}

export async function getPracticeSessionFromDB(
  authToken: string,
  userId: string,
  sessionId: string,
) {
  const userClient = getSupabaseUserClient(authToken);
  const { data, error } = await userClient
    .from("practice_sessions")
    .select("id, mode, status, session_started_at, session_ended_at, origin_language, custom_list_id")
    .eq("id", sessionId)
    .eq("user_id", userId)
    .maybeSingle();

  if (error) {
    if (error.code === "PGRST204" || error.message?.includes("custom_list_name")) {
      const { data: fallbackData, error: fallbackError } = await userClient
        .from("practice_sessions")
        .select("id, mode, status, session_started_at, session_ended_at, origin_language, custom_list_id")
        .eq("id", sessionId)
        .eq("user_id", userId)
        .maybeSingle();
      if (fallbackError) throw fallbackError;
      return fallbackData;
    }
    throw error;
  }
  return data;
}

/**
 * Fetch a user's subscription record from the DB.
 */
export async function getUserSubscriptionFromDB(
  authToken: string,
  userId: string,
) {
  const userClient = getSupabaseUserClient(authToken);
  const { data, error } = await userClient
    .from("user_subscriptions")
    .select("*")
    .eq("user_id", userId)
    .maybeSingle();

  if (error) {
    throw error;
  }
  return data;
}

/**
 * Upsert a user's subscription record in the DB.
 */
export async function updateUserSubscriptionInDB(
  authToken: string,
  userId: string,
  subData: {
    stripeCustomerId: string | null;
    stripeSubscriptionId: string | null;
    status: string | null;
    currentPeriodEnd: string | null;
    cancelAtPeriodEnd: boolean;
    stripePriceId: string | null;
    priceUnitAmount: number | null;
    priceCurrency: string | null;
    billingInterval: string | null;
  },
) {
  const userClient = getSupabaseUserClient(authToken);
  const { data, error } = await userClient
    .from("user_subscriptions")
    .upsert({
      user_id: userId,
      stripe_customer_id: subData.stripeCustomerId,
      stripe_subscription_id: subData.stripeSubscriptionId,
      status: subData.status,
      current_period_end: subData.currentPeriodEnd,
      cancel_at_period_end: subData.cancelAtPeriodEnd,
      stripe_price_id: subData.stripePriceId,
      price_unit_amount: subData.priceUnitAmount,
      price_currency: subData.priceCurrency,
      billing_interval: subData.billingInterval,
      updated_at: new Date().toISOString(),
    })
    .select()
    .single();

  if (error) {
    throw error;
  }
  return data;
}

export interface DBMockBeeSessionRow {
  id: string;
  user_id: string;
  mode: "mock_bee";
  status: PracticeSessionStatus;
  session_started_at: string;
  session_ended_at: string | null;
  total_words_attempted: number | null;
  total_correct: number | null;
  duration_seconds: number | null;
  created_at: string;
  session_config: any;
  session_state: any;
}

export async function createMockBeeSessionInDB(
  authToken: string,
  userId: string,
  sessionConfig: unknown,
  sessionState: unknown,
) {
  const userClient = getSupabaseUserClient(authToken);

  const { data, error } = await userClient
    .from("practice_sessions")
    .insert({
      user_id: userId,
      mode: "mock_bee",
      status: "active",
      session_started_at: new Date().toISOString(),
      session_config: sessionConfig,
      session_state: sessionState,
      total_words_attempted: 0,
      total_correct: 0,
    })
    .select("*")
    .single();

  if (error) throw error;
  return data as DBMockBeeSessionRow;
}

export async function getMockBeeSessionFromDB(
  authToken: string,
  userId: string,
  sessionId: string,
) {
  const userClient = getSupabaseUserClient(authToken);

  const { data, error } = await userClient
    .from("practice_sessions")
    .select("*")
    .eq("id", sessionId)
    .eq("user_id", userId)
    .eq("mode", "mock_bee")
    .maybeSingle();

  if (error) throw error;
  return data as DBMockBeeSessionRow | null;
}

export async function updateMockBeeSessionInDB(
  authToken: string,
  userId: string,
  sessionId: string,
  updates: {
    session_state?: unknown;
    session_config?: unknown;
    status?: PracticeSessionStatus;
    total_words_attempted?: number;
    total_correct?: number;
    session_ended_at?: string | null;
    duration_seconds?: number;
  },
) {
  const userClient = getSupabaseUserClient(authToken);

  const { data, error } = await userClient
    .from("practice_sessions")
    .update(updates)
    .eq("id", sessionId)
    .eq("user_id", userId)
    .eq("mode", "mock_bee")
    .select("*")
    .single();

  if (error) throw error;
  return data as DBMockBeeSessionRow;
}

export async function startMockBeeSessionInDB(
  authToken: string,
  userId: string,
  sessionConfig: unknown,
  sessionState: unknown,
  forceCloseCurrent?: boolean,
): Promise<StartPracticeSessionResult> {
  const userClient = getSupabaseUserClient(authToken);

  const { data: activeSession, error: activeSessionError } = await userClient
    .from("practice_sessions")
    .select("id, mode, session_started_at, total_words_attempted, total_correct")
    .eq("user_id", userId)
    .eq("status", "active")
    .is("session_ended_at", null)
    .order("session_started_at", { ascending: false })
    .limit(1)
    .maybeSingle();

  if (activeSessionError) {
    throw activeSessionError;
  }

  if (activeSession) {
    if (activeSession.mode === "mock_bee" && !forceCloseCurrent) {
      return {
        action: "resume_existing",
        sessionId: activeSession.id as string,
      };
    }

    if (!forceCloseCurrent) {
      return {
        action: "active_session_conflict",
        activeSessionId: activeSession.id as string,
        activeMode: activeSession.mode as string,
      };
    }

    await forceCloseActiveSessionInDB(userClient, userId, {
      id: activeSession.id as string,
      session_started_at: activeSession.session_started_at as string,
      total_words_attempted: activeSession.total_words_attempted as number | null,
      total_correct: activeSession.total_correct as number | null,
    }, "abandoned");
  }

  const { data, error } = await userClient
    .from("practice_sessions")
    .insert({
      user_id: userId,
      mode: "mock_bee",
      status: "active",
      session_started_at: new Date().toISOString(),
      session_config: sessionConfig,
      session_state: sessionState,
      total_words_attempted: 0,
      total_correct: 0,
    })
    .select("id")
    .single();

  if (error) {
    throw error;
  }

  return {
    action: "created",
    sessionId: data.id as string,
  };
}
