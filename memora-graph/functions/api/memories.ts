/**
 * GET /api/memories - Returns all memories for timeline view
 * Supports ?db=memora or ?db=ob1 parameter to select database
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

interface Memory {
  id: number;
  content: string;
  metadata: string;
  tags: string;
  created_at: string;
  updated_at: string | null;
}

function parseJson<T>(str: string | null, defaultValue: T): T {
  if (!str) return defaultValue;
  try {
    return JSON.parse(str);
  } catch {
    return defaultValue;
  }
}

function expandR2Urls(metadata: Record<string, unknown> | null): Record<string, unknown> {
  if (!metadata) return {};

  const images = metadata.images as Array<{ src: string; caption?: string }> | undefined;
  if (images?.length) {
    metadata.images = images.map(img => {
      let src = img.src;
      // Convert r2:// URLs to our proxy path
      if (src?.startsWith("r2://")) {
        src = "/api/r2/" + src.replace("r2://", "");
      }
      return { ...img, src };
    });
  }

  return metadata;
}

export const onRequestGet: PagesFunction<Env> = async ({ env, request }) => {
  const url = new URL(request.url);
  const dbName = url.searchParams.get("db");
  const db = getDatabase(env, dbName);

  const limit = Math.min(Math.max(parseInt(url.searchParams.get("limit") || "50", 10) || 50, 1), 200);
  const offset = Math.max(parseInt(url.searchParams.get("offset") || "0", 10) || 0, 0);

  const countRow = await db.prepare("SELECT COUNT(*) as cnt FROM memories").first<{ cnt: number }>();
  const total = countRow?.cnt ?? 0;

  const result = await db.prepare(
    "SELECT id, content, metadata, tags, created_at, updated_at FROM memories ORDER BY created_at DESC LIMIT ? OFFSET ?"
  ).bind(limit, offset).all<Memory>();

  const memories = (result.results || []).map(m => {
    const meta = parseJson<Record<string, unknown>>(m.metadata, {});
    return {
      id: m.id,
      content: m.content,
      tags: parseJson<string[]>(m.tags, []),
      created: m.created_at || "",
      updated: m.updated_at,
      metadata: expandR2Urls(meta),
    };
  });

  return Response.json({ memories, total, limit, offset });
};
