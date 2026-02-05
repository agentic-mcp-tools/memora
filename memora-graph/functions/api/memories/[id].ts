/**
 * GET /api/memories/:id - Returns a single memory by ID
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

export const onRequestPatch: PagesFunction<Env> = async ({ env, params, request }) => {
  const url = new URL(request.url);
  const dbName = url.searchParams.get("db");
  const db = getDatabase(env, dbName);

  const id = parseInt(params.id as string, 10);
  if (isNaN(id)) {
    return Response.json({ error: "invalid_id" }, { status: 400 });
  }

  const body = await request.json<{ favorite?: boolean }>();
  const favorite = Boolean(body.favorite);

  // Read current metadata
  const row = await db.prepare(
    "SELECT metadata FROM memories WHERE id = ?"
  ).bind(id).first<{ metadata: string }>();

  if (!row) {
    return Response.json({ error: "not_found" }, { status: 404 });
  }

  const meta = parseJson<Record<string, unknown>>(row.metadata, {});
  if (favorite) {
    meta.favorite = true;
  } else {
    delete meta.favorite;
  }

  await db.prepare(
    "UPDATE memories SET metadata = ?, updated_at = datetime('now') WHERE id = ?"
  ).bind(JSON.stringify(meta), id).run();

  return Response.json({ ok: true });
};

export const onRequestGet: PagesFunction<Env> = async ({ env, params, request }) => {
  const url = new URL(request.url);
  const dbName = url.searchParams.get("db");
  const db = getDatabase(env, dbName);

  const id = parseInt(params.id as string, 10);

  if (isNaN(id)) {
    return Response.json({ error: "invalid_id" }, { status: 400 });
  }

  const result = await db.prepare(
    "SELECT id, content, metadata, tags, created_at, updated_at FROM memories WHERE id = ?"
  ).bind(id).first<Memory>();

  if (!result) {
    return Response.json({ error: "not_found" }, { status: 404 });
  }

  const meta = parseJson<Record<string, unknown>>(result.metadata, {});

  return Response.json({
    id: result.id,
    content: result.content,
    tags: parseJson<string[]>(result.tags, []),
    created: result.created_at || "",
    updated: result.updated_at,
    metadata: expandR2Urls(meta),
  });
};
