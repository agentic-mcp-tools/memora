/**
 * POST /api/chat - Chat about memories using LLM with RAG
 * Uses semantic search (embeddings) + keyword search for memory retrieval.
 * Requires OPENROUTER_API_KEY secret and optionally CHAT_MODEL env var.
 * Supports ?db=memora or ?db=ob1 parameter to select database.
 */

interface Env {
  DB_MEMORA: D1Database;
  DB_OB1: D1Database;
  DEFAULT_DB?: string;
  OPENROUTER_API_KEY?: string;
  CHAT_MODEL?: string;
  EMBEDDING_MODEL?: string;
}

function getDatabase(env: Env, dbName: string | null): D1Database {
  const name = dbName || env.DEFAULT_DB || "memora";
  if (name === "ob1") return env.DB_OB1;
  return env.DB_MEMORA;
}

interface MemoryRow {
  id: number;
  content: string;
  tags: string;
  created_at?: string;
}

interface ChatMessage {
  role: string;
  content: string;
}

interface ChatRequest {
  message: string;
  history?: ChatMessage[];
}

interface MemoryReference {
  id: number;
  score: number;
  preview: string;
}

function parseJson<T>(str: string | null, defaultValue: T): T {
  if (!str) return defaultValue;
  try {
    return JSON.parse(str);
  } catch {
    return defaultValue;
  }
}

/**
 * Get embedding vector for a query via OpenRouter/OpenAI embeddings API.
 */
async function getQueryEmbedding(
  query: string,
  apiKey: string,
  model: string
): Promise<number[] | null> {
  try {
    const response = await fetch(
      "https://openrouter.ai/api/v1/embeddings",
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${apiKey}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ input: query, model }),
      }
    );
    if (!response.ok) return null;
    const data = await response.json<{
      data: Array<{ embedding: number[] }>;
    }>();
    return data.data?.[0]?.embedding || null;
  } catch {
    return null;
  }
}

/**
 * Compute cosine similarity between two vectors.
 */
function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length || a.length === 0) return 0;
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

/**
 * Semantic search: embed the query, then compare against stored embeddings.
 */
async function semanticSearch(
  db: D1Database,
  queryEmbedding: number[],
  topK: number
): Promise<Array<{ memory: MemoryRow; score: number }>> {
  // Fetch all embeddings and memory content in one query
  const result = await db
    .prepare(
      `SELECT m.id, m.content, m.tags, m.created_at, e.embedding
       FROM memories m
       JOIN memories_embeddings e ON e.memory_id = m.id`
    )
    .all<MemoryRow & { embedding: string }>();

  if (!result.results || result.results.length === 0) return [];

  // Compute recency boost: newer memories get a small score bonus
  const now = Date.now();
  function recencyBoost(createdAt: string | undefined): number {
    if (!createdAt) return 0;
    const age = now - new Date(createdAt).getTime();
    const days = age / (1000 * 60 * 60 * 24);
    // Boost up to 0.05 for very recent (today), decaying over 90 days
    return Math.max(0, 0.05 * (1 - days / 90));
  }

  // Score each memory by cosine similarity + recency
  const scored: Array<{ memory: MemoryRow; score: number }> = [];
  for (const row of result.results) {
    // Stored format: array of [string_index, float_value] pairs
    const pairs = parseJson<Array<[string, number]>>(row.embedding, []);
    if (pairs.length === 0) continue;

    // Convert to dense array
    const dense = new Array(queryEmbedding.length).fill(0);
    for (const [k, v] of pairs) {
      const idx = parseInt(k, 10);
      if (idx < dense.length) {
        dense[idx] = v;
      }
    }

    const similarity = cosineSimilarity(queryEmbedding, dense);
    const boost = recencyBoost(row.created_at);
    const score = similarity + boost;

    if (similarity > 0.1) {
      scored.push({
        memory: { id: row.id, content: row.content, tags: row.tags, created_at: row.created_at },
        score,
      });
    }
  }

  // Sort by score descending, return top K
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, topK);
}

/**
 * Keyword fallback search using LIKE matching on content and tags.
 */
async function keywordSearch(
  db: D1Database,
  query: string,
  topK: number
): Promise<Array<{ memory: MemoryRow; score: number }>> {
  const keywords = query
    .toLowerCase()
    .split(/\s+/)
    .map(w => w.replace(/[^a-z0-9-]/g, ""))
    .filter(w => w.length >= 3);

  if (keywords.length === 0) {
    const result = await db
      .prepare(
        "SELECT id, content, tags FROM memories ORDER BY created_at DESC LIMIT ?"
      )
      .bind(topK)
      .all<MemoryRow>();
    return (result.results || []).map(m => ({ memory: m, score: 0.1 }));
  }

  const conditions = keywords.map(
    () => "(LOWER(content) LIKE ? OR LOWER(tags) LIKE ?)"
  );
  const params: string[] = [];
  for (const k of keywords) {
    params.push(`%${k}%`, `%${k}%`);
  }

  const sql = `
    SELECT id, content, tags
    FROM memories
    WHERE ${conditions.join(" OR ")}
    ORDER BY created_at DESC
    LIMIT ?
  `;

  const result = await db
    .prepare(sql)
    .bind(...params, topK)
    .all<MemoryRow>();

  return (result.results || []).map(m => ({
    memory: { id: m.id, content: m.content, tags: m.tags },
    score: 0.3,
  }));
}

/**
 * Combined search: try semantic first, fall back to keyword, then recent.
 * Returns results with a `method` field indicating which search was used.
 */
async function searchMemories(
  db: D1Database,
  query: string,
  apiKey: string,
  embeddingModel: string,
  topK: number = 8
): Promise<{ results: Array<{ memory: MemoryRow; score: number }>; method: string }> {
  // Try semantic search first
  const queryEmbedding = await getQueryEmbedding(query, apiKey, embeddingModel);
  if (queryEmbedding) {
    const results = await semanticSearch(db, queryEmbedding, topK);
    if (results.length > 0) return { results, method: "semantic" };
  }

  // Try keyword search
  const kwResults = await keywordSearch(db, query, topK);
  if (kwResults.length > 0) return { results: kwResults, method: "keyword" };

  // Final fallback: return recent memories
  const result = await db
    .prepare(
      "SELECT id, content, tags FROM memories ORDER BY created_at DESC LIMIT ?"
    )
    .bind(topK)
    .all<MemoryRow>();
  return {
    results: (result.results || []).map(m => ({ memory: m, score: 0.1 })),
    method: "recent",
  };
}

export const onRequestPost: PagesFunction<Env> = async ({
  env,
  request,
}) => {
  const url = new URL(request.url);
  const dbName = url.searchParams.get("db");
  const db = getDatabase(env, dbName);

  const apiKey = env.OPENROUTER_API_KEY;
  if (!apiKey) {
    return Response.json(
      {
        error: "llm_not_configured",
        message:
          "LLM not configured. Set OPENROUTER_API_KEY secret in Cloudflare dashboard.",
      },
      { status: 503 }
    );
  }

  let body: ChatRequest;
  try {
    body = await request.json<ChatRequest>();
  } catch {
    return Response.json({ error: "invalid_json" }, { status: 400 });
  }

  const message = (body.message || "").trim();
  if (!message) {
    return Response.json({ error: "empty_message" }, { status: 400 });
  }

  const history = body.history || [];
  const model = env.CHAT_MODEL || "openai/gpt-4o-mini";
  const embeddingModel = env.EMBEDDING_MODEL || "openai/text-embedding-3-small";

  // Search for relevant memories using semantic + keyword search
  const { results: searchResults, method: searchMethod } = await searchMemories(
    db,
    message,
    apiKey,
    embeddingModel,
    8
  );

  // Build references and context
  const references: (MemoryReference & { method?: string })[] = [];
  const contextParts: string[] = [];

  for (const r of searchResults) {
    const mem = r.memory;
    const tags = parseJson<string[]>(mem.tags, []);
    references.push({
      id: mem.id,
      score: Math.round(r.score * 1000) / 1000,
      preview: mem.content.slice(0, 100).replace(/\n/g, " "),
    });
    const tagsStr = tags.join(", ");
    const dateStr = mem.created_at ? ` [${mem.created_at.split(" ")[0]}]` : "";
    const contentTruncated = mem.content.slice(0, 1500);
    contextParts.push(
      `Memory #${mem.id} (tags: ${tagsStr})${dateStr}:\n${contentTruncated}`
    );
  }

  const contextBlock =
    contextParts.length > 0
      ? contextParts.join("\n\n---\n\n")
      : "No relevant memories found.";

  const systemMsg: ChatMessage = {
    role: "system",
    content: [
      "You are a helpful assistant answering questions about the user's personal knowledge base.",
      "Use the following memories as context. When referencing a memory, cite it as [Memory #<id>].",
      "If the memories don't contain relevant information, say so honestly.",
      "",
      "## Relevant Memories",
      "",
      contextBlock,
    ].join("\n"),
  };

  // Build messages: system + last 20 history + current
  const trimmedHistory = history.slice(-20);
  const messages = [
    systemMsg,
    ...trimmedHistory,
    { role: "user", content: message },
  ];

  // Stream response from OpenRouter
  const llmResponse = await fetch(
    "https://openrouter.ai/api/v1/chat/completions",
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
        "HTTP-Referer": url.origin,
      },
      body: JSON.stringify({
        model,
        messages,
        stream: true,
        temperature: 0.7,
        max_tokens: 2000,
      }),
    }
  );

  if (!llmResponse.ok || !llmResponse.body) {
    const errText = await llmResponse.text().catch(() => "Unknown error");
    return Response.json(
      { error: "llm_error", message: errText.slice(0, 200) },
      { status: 502 }
    );
  }

  // Transform the OpenAI-format SSE stream into our chat SSE format
  const encoder = new TextEncoder();
  const decoder = new TextDecoder();
  let buffer = "";

  const { readable, writable } = new TransformStream();
  const writer = writable.getWriter();

  const writeSSE = async (event: string, data: string) => {
    // JSON-encode token data to preserve newlines in SSE transport
    const encoded = event === "token" ? JSON.stringify(data) : data;
    await writer.write(encoder.encode(`event: ${event}\ndata: ${encoded}\n\n`));
  };

  // Process the stream in the background
  const processStream = async () => {
    try {
      // Emit references with search method info
      if (references.length > 0) references[0].method = searchMethod;
      await writeSSE("references", JSON.stringify(references));

      const reader = llmResponse.body!.getReader();
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const data = line.slice(6).trim();
          if (data === "[DONE]") continue;

          try {
            const chunk = JSON.parse(data);
            const content = chunk.choices?.[0]?.delta?.content;
            if (content) {
              await writeSSE("token", content);
            }
          } catch {
            // Skip malformed chunks
          }
        }
      }

      await writeSSE("done", "");
    } catch (e: unknown) {
      const errMsg = e instanceof Error ? e.message : String(e);
      await writeSSE("error", errMsg.slice(0, 200));
    } finally {
      await writer.close();
    }
  };

  // Start processing without awaiting (runs concurrently with response streaming)
  processStream();

  return new Response(readable, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    },
  });
};
