/**
 * GET /api/actions - Returns action history for the History tab
 * Supports ?db=memora or ?db=ob1 parameter to select database
 * Supports ?limit=200 to control number of results
 */

interface Env {
  DB_MEMORA: D1Database;
  DB_OB1: D1Database;
  DEFAULT_DB?: string;
}

function getDatabase(env: Env, dbName: string | null): D1Database {
  const name = dbName || env.DEFAULT_DB || "memora";
  if (name === "ob1") return env.DB_OB1;
  return env.DB_MEMORA;
}

interface Action {
  id: number;
  memory_id: number | null;
  action: string;
  summary: string;
  timestamp: string;
}

export const onRequestGet: PagesFunction<Env> = async ({ env, request }) => {
  const url = new URL(request.url);
  const dbName = url.searchParams.get("db");
  const db = getDatabase(env, dbName);
  const limit = Math.min(parseInt(url.searchParams.get("limit") || "200", 10), 500);

  try {
    const result = await db.prepare(
      "SELECT id, memory_id, action, summary, timestamp FROM memories_actions ORDER BY id DESC LIMIT ?"
    ).bind(limit).all<Action>();

    return Response.json({ actions: result.results || [] });
  } catch {
    // Table may not exist yet
    return Response.json({ actions: [] });
  }
};
