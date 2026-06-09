import type { IncomingMessage } from "node:http";
import { logError } from "./logging.js";

export type AuthenticatedUser = {
  id: string;
  email: string | null;
};

function getBearerToken(request: IncomingMessage): string | null {
  const authorizationHeader = request.headers.authorization;
  if (!authorizationHeader) {
    return null;
  }

  const [scheme, token] = authorizationHeader.split(" ");
  if (scheme?.toLowerCase() !== "bearer" || !token) {
    return null;
  }

  return token.trim();
}

function getSupabaseConfig(): { url: string; publishableKey: string } {
  const url = process.env.SUPABASE_URL?.trim() ?? "";
  const publishableKey = process.env.SUPABASE_PUBLISHABLE_KEY?.trim() ?? "";

  if (!url || !publishableKey) {
    throw new Error(
      "Supabase auth is not configured. Set SUPABASE_URL and SUPABASE_PUBLISHABLE_KEY.",
    );
  }

  return { url, publishableKey };
}

export async function authenticateRequest(
  request: IncomingMessage,
): Promise<AuthenticatedUser> {
  const token = getBearerToken(request);
  if (!token) {
    throw new Error("Unauthorized: missing bearer token.");
  }

  const { url, publishableKey } = getSupabaseConfig();
  const response = await fetch(`${url}/auth/v1/user`, {
    method: "GET",
    headers: {
      apikey: publishableKey,
      Authorization: `Bearer ${token}`,
    },
  });

  if (!response.ok) {
    const errorBody = await response.text().catch(() => "Unknown error");
    logError(`Supabase auth request failed with status ${response.status}: ${errorBody}`);
    throw new Error("Unauthorized: invalid or expired token.");
  }

  const user = (await response.json()) as { id?: unknown; email?: unknown };
  if (!user?.id || typeof user.id !== "string") {
    throw new Error("Unauthorized: token did not resolve to a user.");
  }

  return {
    id: user.id,
    email: typeof user.email === "string" ? user.email : null,
  };
}
