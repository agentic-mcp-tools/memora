"""HTTP server and routes for graph visualization."""

import asyncio
import os
import socket
import sys
import threading
from importlib.metadata import version as get_version
from typing import TYPE_CHECKING

from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response
from sse_starlette.sse import EventSourceResponse

import json
from importlib.resources import files as _pkg_files

from .data import export_graph_html, get_graph_data, get_memory_for_api
from ..storage import connect, update_memory

def _get_memora_version() -> str:
    try:
        return get_version("memora")
    except Exception:
        return ""

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP


def _normalize_host_for_connect(host: str) -> str:
    """Convert wildcard bind addresses to connectable localhost."""
    if host in ("0.0.0.0", "::", ""):
        return "127.0.0.1"
    return host


def _check_port_status(host: str, port: int) -> str:
    """Check port status and identify what's running.

    Returns:
        "free" - port is available
        "memora" - our graph server is running
        "other" - something else is using the port
    """
    connect_host = _normalize_host_for_connect(host)

    # First, quick check if port is in use
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        try:
            s.connect((connect_host, port))
        except (OSError, socket.timeout):
            return "free"

    # Port is in use - verify it's our graph server
    try:
        import urllib.request
        url = f"http://{connect_host}:{port}/api/graph"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = resp.read().decode()
            # Check for our specific response structure
            if '"nodes"' in data or '"count"' in data:
                return "memora"
    except Exception:
        pass

    return "other"


def register_graph_routes(mcp: "FastMCP") -> None:
    """Register graph-related routes on the MCP server.

    Args:
        mcp: The FastMCP server instance.
    """

    @mcp.custom_route("/graph", methods=["GET"])
    async def serve_graph(request: Request):
        """Serve the knowledge graph visualization via HTTP."""
        try:
            min_score = float(request.query_params.get("min_score", 0.25))
            result = export_graph_html(output_path=None, min_score=min_score)
            if "error" in result:
                return JSONResponse(result, status_code=404)
            return HTMLResponse(result["html"])
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)


def start_graph_server(host: str, port: int) -> None:
    """Start background HTTP server for graph visualization.

    This server provides:
    - /graph: SPA HTML page
    - /api/graph: Graph data API
    - /api/memories/{id}: Individual memory API
    - /r2/{path}: R2 image proxy

    Args:
        host: Host to bind to
        port: Port to bind to
    """
    port_status = _check_port_status(host, port)
    if port_status == "memora":
        print(f"Graph server already running on port {port}, reusing existing", file=sys.stderr)
        return
    elif port_status == "other":
        print(f"Port {port} is in use by another service, skipping graph server", file=sys.stderr)
        return

    from starlette.applications import Starlette
    from starlette.routing import Route

    def _load_spa_html(version: str) -> str:
        html = _pkg_files("memora.graph").joinpath("index.html").read_text("utf-8")
        config = json.dumps({
            "version": version,
            "r2Prefix": "/r2/",
            "dbSelector": False,
            "wsUrl": None,
            "sseUrl": "/api/events",
        })
        return html.replace(
            "</head>",
            f"<script>window.MEMORA_CONFIG={config};</script>\n</head>",
        )

    GRAPH_HTML = _load_spa_html(version=_get_memora_version())

    async def graph_handler(request: Request):
        """Serve the static graph SPA."""
        return HTMLResponse(GRAPH_HTML)

    async def api_graph(request: Request):
        """API endpoint: Get graph nodes and edges."""
        try:
            min_score = float(request.query_params.get("min_score", 0.25))
            rebuild = request.query_params.get("rebuild", "").lower() == "true"
            result = get_graph_data(min_score, rebuild=rebuild)
            return JSONResponse(result)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    async def api_memory(request: Request):
        """API endpoint: Get a single memory by ID."""
        try:
            memory_id = int(request.path_params.get("id"))
            result = get_memory_for_api(memory_id)
            if result.get("error") == "not_found":
                return JSONResponse(result, status_code=404)
            return JSONResponse(result)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    async def api_memories_list(request: Request):
        """API endpoint: Get memories for timeline with pagination."""
        try:
            limit = max(1, min(int(request.query_params.get("limit", "50")), 200))
            offset = max(0, int(request.query_params.get("offset", "0")))

            conn = connect()
            total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            rows = conn.execute(
                """SELECT id, content, created_at, updated_at, tags, metadata
                   FROM memories ORDER BY created_at DESC
                   LIMIT ? OFFSET ?""",
                (limit, offset),
            ).fetchall()
            conn.close()
            memories = []
            for row in rows:
                memories.append({
                    "id": row["id"],
                    "content": row["content"],
                    "created": row["created_at"].split(" ")[0] if row["created_at"] else "",
                    "updated": row["updated_at"].split(" ")[0] if row["updated_at"] else None,
                    "tags": json.loads(row["tags"]) if row["tags"] else [],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                })
            return JSONResponse({
                "memories": memories,
                "total": total,
                "limit": limit,
                "offset": offset,
            })
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    async def api_actions(request: Request):
        """API endpoint: Get action history."""
        try:
            from ..storage import get_action_history
            limit = int(request.query_params.get("limit", "200"))
            conn = connect()
            actions = get_action_history(conn, limit=limit)
            conn.close()
            return JSONResponse({"actions": actions})
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    async def graph_events(request: Request):
        """SSE endpoint for graph update notifications."""
        async def event_generator():
            last_count = None
            last_modified = None
            while True:
                try:
                    conn = connect()
                    row = conn.execute(
                        """SELECT COUNT(*) as cnt,
                           MAX(COALESCE(updated_at, created_at)) as latest
                           FROM memories"""
                    ).fetchone()
                    conn.close()

                    current_count = row["cnt"] if row else 0
                    current_modified = row["latest"] if row else None

                    # Detect changes (create, update, or delete)
                    if last_count is not None and (
                        current_count != last_count or current_modified != last_modified
                    ):
                        yield {"event": "graph-updated", "data": "refresh"}

                    last_count = current_count
                    last_modified = current_modified
                except Exception:
                    pass

                await asyncio.sleep(2)  # Check every 2 seconds

        return EventSourceResponse(event_generator())

    async def api_memory_patch(request: Request):
        """API endpoint: Toggle favorite on a memory."""
        import json
        try:
            memory_id = int(request.path_params.get("id"))
            body = await request.json()
            favorite = bool(body.get("favorite", False))

            conn = connect()
            row = conn.execute(
                "SELECT metadata FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
            if not row:
                conn.close()
                return JSONResponse({"error": "not_found"}, status_code=404)

            metadata = json.loads(row["metadata"]) if row["metadata"] else {}
            if favorite:
                metadata["favorite"] = True
            else:
                metadata.pop("favorite", None)

            update_memory(conn, memory_id, metadata=metadata)
            conn.close()
            return JSONResponse({"ok": True})
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    async def r2_image_proxy(request: Request):
        """Proxy images from R2 storage."""
        try:
            from ..image_storage import get_image_storage_instance

            image_storage = get_image_storage_instance()
            if not image_storage:
                return JSONResponse({"error": "R2 not configured"}, status_code=503)

            key = request.path_params.get("path", "")
            if not key:
                return JSONResponse({"error": "No path provided"}, status_code=400)

            try:
                response = image_storage.s3_client.get_object(
                    Bucket=image_storage.bucket,
                    Key=key,
                )
                image_data = response["Body"].read()
                content_type = response.get("ContentType", "image/jpeg")

                return Response(
                    content=image_data,
                    media_type=content_type,
                    headers={"Cache-Control": "public, max-age=86400"},
                )
            except Exception as e:
                return JSONResponse({"error": f"Image not found: {e}"}, status_code=404)

        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    app = Starlette(
        routes=[
            Route("/graph", graph_handler),
            Route("/api/graph", api_graph),
            Route("/api/events", graph_events),
            Route("/api/memories", api_memories_list),
            Route("/api/memories/{id:int}", api_memory),
            Route("/api/memories/{id:int}/favorite", api_memory_patch, methods=["PATCH"]),
            Route("/api/actions", api_actions),
            Route("/r2/{path:path}", r2_image_proxy),
        ]
    )

    def run_server():
        import uvicorn

        config = uvicorn.Config(app, host=host, port=port, log_level="warning")
        server = uvicorn.Server(config)
        # SO_REUSEADDR is set by default in uvicorn, but we ensure quick restart
        server.run()

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()

    # Get bucket name for unique URL
    bucket_name = ""
    try:
        from ..storage import STORAGE_BACKEND
        if hasattr(STORAGE_BACKEND, 'bucket'):
            bucket_name = STORAGE_BACKEND.bucket
    except Exception:
        pass

    bucket_param = f"?bucket={bucket_name}" if bucket_name else ""
    print(f"Graph visualization available at http://{host}:{port}/graph{bucket_param}", file=sys.stderr)
