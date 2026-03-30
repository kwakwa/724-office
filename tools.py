"""
Tool Registry - All LLM-callable tool definitions + implementations

## Adding a new tool:
1. Write a function at the bottom of this file
2. Decorate with @tool

No other files need to be modified.

## Decorator usage:
@tool("tool_name", "description", {parameter_schema}, ["required_params"])
def my_tool(args, ctx):
    return "result string"

args: dict - parameters passed by LLM
ctx: dict - runtime context {"owner_id", "workspace", "session_key"}
"""

import json
import logging
import os
import subprocess
import time
import urllib.request
import urllib.parse

log = logging.getLogger("agent")

# ============================================================
#  Tool Registry
# ============================================================

_registry = {}  # name -> {"fn", "definition"}


def tool(name, description, properties, required=None):
    """Decorator: register an LLM tool"""
    def decorator(fn):
        _registry[name] = {
            "fn": fn,
            "definition": {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        **({"required": required} if required else {}),
                    },
                },
            },
        }
        return fn
    return decorator


def get_definitions():
    """Return all tool definitions in OpenAI function calling format"""
    return [entry["definition"] for entry in _registry.values()]


def execute(name, args, ctx):
    """Execute a tool, return result string"""
    log.info(f"[tool] {name}({json.dumps(args, ensure_ascii=False)[:200]})")
    entry = _registry.get(name)
    if not entry:
        return f"[error] unknown tool: {name}"
    try:
        return entry["fn"](args, ctx)
    except Exception as e:
        log.error(f"[tool] {name} error: {e}", exc_info=True)
        return f"[error] {e}"


# ============================================================
#  Tool Implementations
# ============================================================

# Lazy imports to avoid circular dependencies
import messaging
import scheduler


def _resolve_path(path, workspace):
    if os.path.isabs(path):
        return path
    return os.path.join(workspace, path)


def _split_message(text, max_bytes=1800):
    if len(text.encode("utf-8")) <= max_bytes:
        return [text]
    chunks, current = [], ""
    for line in text.split("\n"):
        test = current + "\n" + line if current else line
        if len(test.encode("utf-8")) > max_bytes:
            if current:
                chunks.append(current)
            current = line
        else:
            current = test
    if current:
        chunks.append(current)
    return chunks


# --- Core Tools ---

@tool("exec", "Execute a shell command on the server. "
      "Default timeout 60s. Set timeout to 300 for slow operations (installs, downloads).",
      {"command": {"type": "string", "description": "Shell command to execute"},
       "timeout": {"type": "integer", "description": "Timeout in seconds, default 60, max 300"}},
      ["command"])
def tool_exec(args, ctx):
    timeout = min(args.get("timeout", 60), 300)
    try:
        result = subprocess.run(
            args["command"], shell=True, capture_output=True, text=True,
            timeout=timeout, cwd=ctx["workspace"]
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += ("\n[stderr] " + result.stderr) if output else result.stderr
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return "[error] command timed out (%ds)" % timeout


@tool("message", "Send a text message to the owner via messaging platform. "
      "Used for scheduled task notifications. Normal conversation replies don't need this tool.",
      {"content": {"type": "string", "description": "Message content"}},
      ["content"])
def tool_message(args, ctx):
    owner_id = ctx["owner_id"]
    chunks = _split_message(args["content"], 1800)
    for i, chunk in enumerate(chunks):
        messaging.send_text(owner_id, chunk)
        if i < len(chunks) - 1:
            time.sleep(0.5)
    return f"Sent to owner ({len(chunks)} messages)"


# --- File Tools ---

@tool("read_file", "Read file content. Path relative to workspace directory.",
      {"path": {"type": "string", "description": "File path (relative to workspace or absolute)"}},
      ["path"])
def tool_read_file(args, ctx):
    fpath = _resolve_path(args["path"], ctx["workspace"])
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()
        if len(content) > 10000:
            content = content[:10000] + f"\n... (truncated, total {len(content)} chars)"
        return content or "(empty file)"
    except FileNotFoundError:
        return f"[error] file not found: {fpath}"
    except Exception as e:
        return f"[error] {e}"


@tool("write_file", "Write file (overwrite). Path relative to workspace directory.",
      {"path": {"type": "string", "description": "File path"},
       "content": {"type": "string", "description": "File content"}},
      ["path", "content"])
def tool_write_file(args, ctx):
    fpath = _resolve_path(args["path"], ctx["workspace"])
    try:
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(args["content"])
        return f"Written to {fpath} ({len(args['content'])} chars)"
    except Exception as e:
        return f"[error] {e}"


@tool("edit_file", "Edit file: replace old text with new text. "
      "For appending, use the end of file as old and old+new content as new.",
      {"path": {"type": "string", "description": "File path"},
       "old": {"type": "string", "description": "Original text to replace"},
       "new": {"type": "string", "description": "Replacement text"}},
      ["path", "old", "new"])
def tool_edit_file(args, ctx):
    fpath = _resolve_path(args["path"], ctx["workspace"])
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()
        if args["old"] not in content:
            return f"[error] old string not found in {fpath}"
        content = content.replace(args["old"], args["new"], 1)
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Edited {fpath}"
    except FileNotFoundError:
        return f"[error] file not found: {fpath}"
    except Exception as e:
        return f"[error] {e}"


@tool("list_files", "List received and saved files. Filter by type (image/video/file/voice/gif) or list all.",
      {"type": {"type": "string", "description": "File type filter (image/video/file/voice/gif), empty for all"},
       "limit": {"type": "integer", "description": "Number of results (default 20)"}})
def tool_list_files(args, ctx):
    index_path = os.path.join(ctx["workspace"], "files", "index.json")
    if not os.path.exists(index_path):
        return "No files received yet."
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
    except Exception:
        return "File index read failed."

    file_type = args.get("type", "")
    if file_type:
        index = [e for e in index if e.get("type") == file_type]

    limit = args.get("limit", 20)
    recent = index[-limit:]
    recent.reverse()

    if not recent:
        return f"No files of type '{file_type}' found." if file_type else "No files received yet."

    lines = [f"Total {len(index)} files" + (f" (type: {file_type})" if file_type else "") + f", showing {len(recent)} most recent:"]
    for e in recent:
        size_kb = e.get("size", 0) / 1024
        size_str = f"{size_kb/1024:.1f}MB" if size_kb > 1024 else f"{size_kb:.0f}KB"
        lines.append(f"  - [{e.get('type', '?')}] {e.get('filename', '?')} ({size_str}) {e.get('time', '?')}")
        lines.append(f"    Path: {e.get('path', '?')}")
    return "\n".join(lines)


# --- Scheduler Tools ---

@tool("schedule", "Create a scheduled task. One-shot tasks use delay_seconds, recurring tasks use cron_expr. "
      "On trigger, the message is sent to LLM as a user message for processing.",
      {"name": {"type": "string", "description": "Task name (unique identifier)"},
       "message": {"type": "string", "description": "Message sent to LLM on trigger (should include instructions like 'send to owner via message tool')"},
       "delay_seconds": {"type": "integer", "description": "Delay in seconds (one-shot task)"},
       "cron_expr": {"type": "string", "description": "Cron expression (recurring task, e.g. '0 9 * * *')"},
       "once": {"type": "boolean", "description": "Execute only once (default true, only for cron_expr)"}},
      ["name", "message"])
def tool_schedule(args, ctx):
    return scheduler.add(args)


@tool("list_schedules", "List all scheduled tasks", {})
def tool_list_schedules(args, ctx):
    return scheduler.list_all()


@tool("remove_schedule", "Delete a scheduled task",
      {"name": {"type": "string", "description": "Task name"}},
      ["name"])
def tool_remove_schedule(args, ctx):
    return scheduler.remove(args["name"])


# --- Media Send Tools ---

@tool("send_image", "Send an image to the owner. Supports HTTP URL or local file path.",
      {"path": {"type": "string", "description": "Image URL (http/https) or local file path"},
       "caption": {"type": "string", "description": "Optional text caption"}},
      ["path"])
def tool_send_image(args, ctx):
    result = messaging.upload_and_send(ctx["owner_id"], args["path"], args.get("caption", ""), ctx["workspace"])
    return "Image sent to owner" if result.get("code") == 0 else f"[error] Send failed: {result.get('msg', '?')}"


@tool("send_file", "Send a file to the owner (PDF, Excel, Word, ZIP, etc.). Supports HTTP URL or local path.",
      {"path": {"type": "string", "description": "File URL (http/https) or local file path"},
       "caption": {"type": "string", "description": "Optional text caption"}},
      ["path"])
def tool_send_file(args, ctx):
    result = messaging.upload_and_send(ctx["owner_id"], args["path"], args.get("caption", ""), ctx["workspace"])
    return "File sent to owner" if result.get("code") == 0 else f"[error] Send failed: {result.get('msg', '?')}"


@tool("send_video", "Send a video to the owner. Supports HTTP URL or local MP4 file path.",
      {"path": {"type": "string", "description": "Video URL (http/https) or local file path"},
       "caption": {"type": "string", "description": "Optional text caption"}},
      ["path"])
def tool_send_video(args, ctx):
    result = messaging.upload_and_send(ctx["owner_id"], args["path"], args.get("caption", ""), ctx["workspace"])
    return "Video sent to owner" if result.get("code") == 0 else f"[error] Send failed: {result.get('msg', '?')}"


@tool("send_link", "Send a rich link card to the owner. Displays as a card with title, description, and icon.",
      {"title": {"type": "string", "description": "Card title"},
       "desc": {"type": "string", "description": "Card description"},
       "link_url": {"type": "string", "description": "Click-through URL"},
       "icon_url": {"type": "string", "description": "Card icon URL (optional)"}},
      ["title", "desc", "link_url"])
def tool_send_link(args, ctx):
    result = messaging.send_link(ctx["owner_id"], args["title"], args["desc"], args["link_url"], args.get("icon_url", ""))
    return f"Link card sent: {args['title']}" if result.get("code") == 0 else f"[error] Send failed: {result.get('msg', '?')}"


# --- Video Processing Tools ---

def _ensure_local(path, workspace, label="file"):
    """If path is URL, download to /tmp/ and return local path; otherwise return as-is"""
    if path.startswith("http://") or path.startswith("https://"):
        ext = os.path.splitext(urllib.parse.urlparse(path).path)[1] or ".mp4"
        local = "/tmp/agent_%s_%d%s" % (label, int(time.time()), ext)
        log.info("[video] downloading %s -> %s" % (path[:80], local))
        urllib.request.urlretrieve(path, local)
        return local
    return path


def _video_output_path(workspace):
    """Generate output path under workspace/files/YYYY-MM/"""
    from datetime import datetime, timezone, timedelta
    now = datetime.now(timezone(timedelta(hours=8)))
    out_dir = os.path.join(workspace, "files", now.strftime("%Y-%m"))
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, "video_%s.mp4" % now.strftime("%Y%m%d_%H%M%S"))


@tool("trim_video", "Trim video: extract a time segment. Uses -c copy (no re-encoding), millisecond-fast even for large files.",
      {"input_path": {"type": "string", "description": "Video file path (local or URL)"},
       "start": {"type": "string", "description": "Start time, format HH:MM:SS or seconds"},
       "end": {"type": "string", "description": "End time (optional, omit for until end)"},
       "send_to": {"type": "string", "description": "Send to whom after trimming (optional)"}},
      ["input_path", "start"])
def tool_trim_video(args, ctx):
    input_path = _ensure_local(args["input_path"], ctx["workspace"], "trim_in")
    output_path = _video_output_path(ctx["workspace"])

    cmd = ["ffmpeg", "-y", "-ss", str(args["start"])]
    if args.get("end"):
        cmd += ["-to", str(args["end"])]
    cmd += ["-i", input_path, "-c:v", "copy", "-c:a", "copy",
            "-avoid_negative_ts", "make_zero", output_path]

    log.info("[video] trim: %s" % " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        return "[error] ffmpeg trim failed: %s" % result.stderr[-500:]

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    msg = "Trim complete: %s (%.1fMB)" % (output_path, size_mb)

    if args.get("send_to"):
        send_result = messaging.upload_and_send(args["send_to"], output_path, "", ctx["workspace"])
        if send_result.get("code") == 0:
            msg += ", sent"
        else:
            msg += ", send failed: %s" % send_result.get("msg", "?")

    log.info("[video] %s" % msg)
    return msg


@tool("add_bgm", "Add background music to video. Video stream not re-encoded (-c:v copy), only mixes audio tracks.",
      {"video_path": {"type": "string", "description": "Video file path (local or URL)"},
       "audio_path": {"type": "string", "description": "Audio file path (mp3/wav/aac, local or URL)"},
       "volume": {"type": "number", "description": "BGM volume ratio, default 0.3 (30%), to not overpower original audio"},
       "send_to": {"type": "string", "description": "Send to whom after processing (optional)"}},
      ["video_path", "audio_path"])
def tool_add_bgm(args, ctx):
    video = _ensure_local(args["video_path"], ctx["workspace"], "bgm_video")
    audio = _ensure_local(args["audio_path"], ctx["workspace"], "bgm_audio")
    output_path = _video_output_path(ctx["workspace"])
    vol = args.get("volume", 0.3)

    # Video stream copy, only mix audio: original + BGM (volume adjusted)
    filter_complex = "[1:a]volume=%.2f[bgm];[0:a][bgm]amix=inputs=2:duration=first[a]" % vol
    cmd = ["ffmpeg", "-y", "-i", video, "-i", audio,
           "-filter_complex", filter_complex,
           "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac",
           output_path]

    log.info("[video] add_bgm: %s" % " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        return "[error] ffmpeg BGM failed: %s" % result.stderr[-500:]

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    msg = "BGM added: %s (%.1fMB)" % (output_path, size_mb)

    if args.get("send_to"):
        send_result = messaging.upload_and_send(args["send_to"], output_path, "", ctx["workspace"])
        if send_result.get("code") == 0:
            msg += ", sent"
        else:
            msg += ", send failed: %s" % send_result.get("msg", "?")

    log.info("[video] %s" % msg)
    return msg


@tool("generate_video", "Generate video from text description (AI video generation API). "
      "Async task, typically takes 2-5 minutes.",
      {"prompt": {"type": "string", "description": "Video content description (the more detailed, the better)"},
       "size": {"type": "string", "description": "Video resolution, default 1280x720"},
       "send_to": {"type": "string", "description": "Send to whom after generation (optional)"}},
      ["prompt"])
def tool_generate_video(args, ctx):
    # Read video_api config
    video_cfg = _extra_config.get("video_api", {})
    api_key = video_cfg.get("api_key", "")
    if not api_key:
        return "[error] video_api.api_key not configured in config.json"
    api_base = video_cfg.get("api_base", "https://api.video-generation.example.com/v1")
    model = video_cfg.get("model", "video-generation-model")

    # 1. Submit video generation task
    body = json.dumps({
        "model": model,
        "prompt": args["prompt"],
        "size": args.get("size", "1280x720"),
    }).encode("utf-8")
    req = urllib.request.Request(
        "%s/videos/generations" % api_base, data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer %s" % api_key},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            task = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            err_body = json.loads(e.read())
            err_msg = err_body.get("error", {}).get("message", str(err_body)[:200])
        except Exception:
            err_msg = "HTTP %d" % e.code
        return "[error] Video generation submit failed: %s" % err_msg
    except Exception as e:
        return "[error] Video generation submit failed: %s" % e

    task_id = task.get("id", "")
    if not task_id:
        return "[error] No task_id returned: %s" % json.dumps(task, ensure_ascii=False)[:300]
    log.info("[video] generate task submitted: %s (model=%s)" % (task_id, model))

    # 2. Poll for result (3s intervals, up to 300s)
    poll_url = "%s/async-result/%s" % (api_base, task_id)
    for i in range(100):  # 100 * 3s = 300s
        time.sleep(3)
        try:
            poll_req = urllib.request.Request(poll_url, headers={
                "Authorization": "Bearer %s" % api_key,
            })
            with urllib.request.urlopen(poll_req, timeout=15) as resp:
                status = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            try:
                status = json.loads(e.read())
            except Exception:
                log.warning("[video] poll HTTP %d (retry)" % e.code)
                continue
        except Exception as e:
            log.warning("[video] poll error (retry): %s" % e)
            continue

        task_status = status.get("task_status", "")
        if task_status == "SUCCESS":
            # 3. Download generated video
            video_results = status.get("video_result", [])
            if not video_results:
                return "[error] Task succeeded but no video result: %s" % json.dumps(status, ensure_ascii=False)[:300]

            video_url = video_results[0].get("url", "")
            if not video_url:
                return "[error] Video result missing url"

            output_path = _video_output_path(ctx["workspace"])
            urllib.request.urlretrieve(video_url, output_path)
            size_mb = os.path.getsize(output_path) / 1024 / 1024
            msg = "Video generated: %s (%.1fMB, model=%s)" % (output_path, size_mb, model)

            if args.get("send_to"):
                send_result = messaging.upload_and_send(args["send_to"], output_path, "", ctx["workspace"])
                if send_result.get("code") == 0:
                    msg += ", sent"
                else:
                    msg += ", send failed: %s" % send_result.get("msg", "?")

            log.info("[video] %s" % msg)
            return msg

        elif task_status == "FAIL":
            err = status.get("error", {})
            err_msg = err.get("message", json.dumps(status, ensure_ascii=False)[:300])
            return "[error] Video generation failed: %s" % err_msg

        # PROCESSING - continue waiting

    return "[error] Video generation timed out (300s), task_id=%s" % task_id


# --- Web Search Tools ---

_extra_config = {}

# Plugin directory: plugins/ next to tools.py
_plugins_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plugins")


def init_extra(config):
    """Called by xiaowang.py to pass extra config (search API keys, etc.)"""
    global _extra_config
    _extra_config = config
    _load_plugins()
    _load_mcp_servers(config)


# ============================================================
#  Plugin System - Self-Extension Capability
# ============================================================

def _exec_plugin(code, source="<plugin>"):
    """Execute plugin code in controlled environment, plugins can use @tool to register tools"""
    exec(compile(code, source, "exec"), {
        "__builtins__": __builtins__,
        "tool": tool,
        "log": log,
    })


def _load_plugins():
    """On startup, scan plugins/ directory, load all custom tools"""
    if not os.path.isdir(_plugins_dir):
        return
    loaded = 0
    for fname in sorted(os.listdir(_plugins_dir)):
        if not fname.endswith(".py"):
            continue
        fpath = os.path.join(_plugins_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                code = f.read()
            _exec_plugin(code, fname)
            loaded += 1
        except Exception as e:
            log.error("[plugins] failed to load %s: %s" % (fname, e))
    if loaded:
        log.info("[plugins] loaded %d custom tools" % loaded)



def _tavily_search(query, count=5):
    """Tavily API search - high quality for English content, returns summaries and links"""
    api_key = _extra_config.get("tavily_api_key", "")
    if not api_key or api_key.startswith("YOUR_"):
        return "[error] Tavily API key not configured"

    url = "https://api.tavily.com/search"
    body = json.dumps({
        "api_key": api_key,
        "query": query,
        "search_depth": "advanced",
        "include_answer": True,
        "max_results": count,
    }).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={
        "Content-Type": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        return f"[error] Tavily search failed: {e}"

    parts = []
    answer = data.get("answer")
    if answer:
        parts.append("== AI Summary ==\n" + answer)

    results = data.get("results", [])
    if not results:
        return answer or "No relevant results found."

    items = []
    for i, item in enumerate(results[:count], 1):
        title = item.get("title", "")
        content = item.get("content", "")[:300]
        link = item.get("url", "")
        score = item.get("score", 0)
        items.append(f"{i}. {title} (relevance: {score:.2f})\n   {content}\n   Link: {link}")
    parts.append("\n\n".join(items))
    return "\n\n".join(parts)


def _web_search(query, count=5):
    """DuckDuckGo search - free, no API key required."""
    try:
        from ddgs import DDGS
    except ImportError:
        return "[error] ddgs package not installed (pip install ddgs)"

    try:
        results = list(DDGS().text(query, max_results=count))
    except Exception as e:
        return f"[error] DuckDuckGo search failed: {e}"

    if not results:
        return "No relevant results found."

    lines = []
    for i, item in enumerate(results, 1):
        title = item.get("title", "")
        body = item.get("body", "")[:300]
        link = item.get("href", "")
        lines.append(f"{i}. {title}\n   {body}\n   Link: {link}")
    return "\n\n".join(lines)


def _github_search(query, count=5):
    """GitHub public API search: search repos first, then code, merge and deduplicate"""
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "724-office-agent",
    }
    results = []

    # 1. Search repositories (name + description + README)
    encoded = urllib.parse.quote(query)
    url = "https://api.github.com/search/repositories?q=%s&sort=stars&per_page=%d" % (encoded, count)
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        for item in data.get("items", [])[:count]:
            name = item.get("full_name", "")
            desc = (item.get("description") or "")[:150]
            stars = item.get("stargazers_count", 0)
            link = item.get("html_url", "")
            lang = item.get("language", "")
            updated = (item.get("updated_at") or "")[:10]
            line = "%s stars:%d" % (name, stars)
            if lang:
                line += " [%s]" % lang
            if updated:
                line += " (updated %s)" % updated
            line += "\n   %s\n   Link: %s" % (desc, link)
            results.append(line)
    except Exception as e:
        results.append("[repo search error: %s]" % e)

    # 2. If few repo results, supplement with code search
    if len(results) < 2:
        try:
            code_url = "https://api.github.com/search/code?q=%s&per_page=%d" % (encoded, count)
            req = urllib.request.Request(code_url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as resp:
                code_data = json.loads(resp.read())
            seen_repos = set()
            for item in code_data.get("items", []):
                repo = item.get("repository", {})
                repo_name = repo.get("full_name", "")
                if repo_name and repo_name not in seen_repos:
                    seen_repos.add(repo_name)
                    desc = (repo.get("description") or "")[:150]
                    link = repo.get("html_url", "")
                    results.append("%s (from code search)\n   %s\n   Link: %s" % (repo_name, desc, link))
                    if len(seen_repos) >= count:
                        break
        except Exception:
            pass  # Code search is supplementary, failure is OK

    if not results:
        return "No relevant projects found on GitHub."
    return "\n\n".join("%d. %s" % (i, r) for i, r in enumerate(results, 1))


def _huggingface_search(query, count=5):
    """HuggingFace API search for models"""
    encoded = urllib.parse.quote(query)
    url = f"https://huggingface.co/api/models?search={encoded}&sort=downloads&direction=-1&limit={count}"
    req = urllib.request.Request(url, headers={"User-Agent": "724-office-agent"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        # Fallback to web search
        return _web_search(f"huggingface {query}", count)

    if not data:
        return "No relevant models found on HuggingFace."
    results = []
    for i, item in enumerate(data[:count], 1):
        model_id = item.get("modelId", item.get("id", ""))
        downloads = item.get("downloads", 0)
        likes = item.get("likes", 0)
        pipeline = item.get("pipeline_tag", "")
        results.append(f"{i}. {model_id} (downloads: {downloads}, likes: {likes})" +
                       (f" [{pipeline}]" if pipeline else "") +
                       f"\n   Link: https://huggingface.co/{model_id}")
    return "\n\n".join(results)


@tool("web_search", "Web search. Supports multiple search sources. "
      "source=auto uses dual-engine (Tavily + web) by default, specific keywords route to specialized sources. "
      "source=github searches GitHub. source=all searches all sources.",
      {"query": {"type": "string", "description": "Search keywords"},
       "source": {"type": "string", "description": "Search source: auto/web/tavily/github/huggingface/all",
                  "enum": ["auto", "web", "tavily", "github", "huggingface", "all"]},
       "count": {"type": "integer", "description": "Number of results (default 5)"}},
      ["query"])
def tool_web_search(args, ctx):
    query = args["query"]
    source = args.get("source", "auto")
    count = args.get("count", 5)

    if source == "auto":
        ql = query.lower()
        if any(kw in ql for kw in ["huggingface", "hugging face", "hf model"]):
            source = "huggingface"
        elif any(kw in ql for kw in ["github.com", "github repo"]):
            source = "github"
        elif any(kw in ql for kw in ["verify", "exist", "skill", "plugin", "mcp", "tool",
                                      "open source", "repo"]):
            source = "all"
        else:
            source = "web+tavily"

    if source == "github":
        return _github_search(query, count)
    elif source == "tavily":
        return _tavily_search(query, count)
    elif source == "web+tavily":
        # Dual-engine: Tavily + web search
        parts = []
        tav = _tavily_search(query, count)
        if tav and "[error]" not in tav:
            parts.append("== Tavily ==\n" + tav)
        web = _web_search(query, count)
        if web and "[error]" not in web:
            parts.append("== Web ==\n" + web)
        return "\n\n".join(parts) if parts else "No results found."
    elif source == "huggingface":
        return _huggingface_search(query, count)
    elif source == "all":
        # Multi-engine: Tavily + GitHub + web
        parts = []
        tav = _tavily_search(query, max(count // 2, 3))
        if tav and "[error]" not in tav:
            parts.append("== Tavily ==\n" + tav)
        gh = _github_search(query, max(count // 2, 3))
        if gh and "[error]" not in gh:
            parts.append("== GitHub ==\n" + gh)
        web = _web_search(query, max(count // 2, 3))
        if web and "[error]" not in web:
            parts.append("== Web ==\n" + web)
        return "\n\n".join(parts) if parts else "No results from any source."
    else:
        return _web_search(query, count)



# --- Memory Search Tool ---

@tool("search_memory", "Search memory files. Uses keyword search in workspace/memory/ directory.",
      {"query": {"type": "string", "description": "Search keywords (space-separated)"},
       "scope": {"type": "string", "description": "Search scope: all (default), long (MEMORY.md only), daily (daily logs only)"}},
      ["query"])
def tool_search_memory(args, ctx):
    query = args["query"]
    scope = args.get("scope", "all")
    memory_dir = os.path.join(ctx["workspace"], "memory")

    if not os.path.isdir(memory_dir):
        return "Memory directory does not exist."

    grep_args = ["grep", "-r", "-i", "-n", "--include=*.md"]
    if scope == "long":
        target = os.path.join(memory_dir, "MEMORY.md")
        if not os.path.exists(target):
            return "MEMORY.md does not exist."
        grep_args = ["grep", "-i", "-n", "--", query, target]
    elif scope == "daily":
        grep_args.extend(["--include=2*.md", "--", query, memory_dir])
    else:
        grep_args.extend(["--", query, memory_dir])

    try:
        result = subprocess.run(grep_args, capture_output=True, text=True, timeout=10)
        output = result.stdout.strip()
        if not output:
            return "No memories found containing '%s'." % query

        lines = output.split("\n")
        if len(lines) > 30:
            return "\n".join(lines[:30]) + ("\n... %d total matches, showing first 30" % len(lines))
        return "%d matches:\n%s" % (len(lines), "\n".join(lines))
    except Exception as e:
        return "[error] Search failed: %s" % e



# --- Semantic Memory Retrieval Tool ---

@tool('recall', 'Semantic search in long-term memory. Use when the user asks about previous conversations or needs to recall historical information. '
      'More intelligent than search_memory (vector semantic matching vs keyword matching).',
      {'query': {'type': 'string', 'description': 'Search keywords or question'}},
      ['query'])
def tool_recall(args, ctx):
    import memory as mem_mod
    result = mem_mod.retrieve(args['query'], ctx['session_key'], top_k=10)
    return result or 'No relevant memories found.'


# --- Self-Check Tool ---

@tool("self_check", "System self-check: collect today's conversation stats, system health, "
      "error logs, scheduled task status, etc. Used to generate daily self-check reports.", {})
def tool_self_check(args, ctx):
    from datetime import datetime, timezone, timedelta

    CST = timezone(timedelta(hours=8))
    now = datetime.now(CST)
    today = now.strftime("%Y-%m-%d")
    report = []

    # 1. Today's active sessions
    sessions_dir = os.path.join(os.path.dirname(ctx["workspace"]), "sessions")
    active_sessions = 0
    total_user_msgs = 0
    total_assistant_msgs = 0
    total_tool_calls = 0
    if os.path.isdir(sessions_dir):
        for fname in os.listdir(sessions_dir):
            fpath = os.path.join(sessions_dir, fname)
            mtime = datetime.fromtimestamp(os.path.getmtime(fpath), CST)
            if mtime.strftime("%Y-%m-%d") == today:
                active_sessions += 1
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        msgs = json.load(f)
                    for m in msgs:
                        if m.get("role") == "user":
                            total_user_msgs += 1
                        elif m.get("role") == "assistant":
                            total_assistant_msgs += 1
                            if m.get("tool_calls"):
                                total_tool_calls += len(m["tool_calls"])
                except Exception:
                    pass
    report.append("== Today's Conversations (%s) ==" % today)
    report.append("Active sessions: %d" % active_sessions)
    report.append("User messages: %d, Assistant replies: %d, Tool calls: %d" % (total_user_msgs, total_assistant_msgs, total_tool_calls))

    # 2. Today's error logs
    try:
        err_cmd = 'journalctl -u agent --since today --no-pager | grep -ci "error"'
        err_result = subprocess.run(["bash", "-c", err_cmd], capture_output=True, text=True, timeout=10)
        err_count = err_result.stdout.strip() or "0"
        report.append("\n== Error Logs ==")
        report.append("Today's errors: %s" % err_count)
        if int(err_count) > 0:
            last_cmd = 'journalctl -u agent --since today --no-pager | grep -i "error" | tail -5'
            last_errs = subprocess.run(["bash", "-c", last_cmd], capture_output=True, text=True, timeout=10).stdout.strip()
            if last_errs:
                report.append("Last 5:\n" + last_errs)
    except Exception as e:
        report.append("\n== Error Logs ==\nRead failed: %s" % e)

    # 3. Service uptime
    try:
        uptime_result = subprocess.run(
            ["systemctl", "show", "agent", "--property=ActiveEnterTimestamp", "--value"],
            capture_output=True, text=True, timeout=5
        )
        report.append("\n== System Status ==")
        report.append("Service start time: %s" % uptime_result.stdout.strip())
    except Exception:
        pass

    # 4. Memory and disk
    try:
        mem = subprocess.run(["bash", "-c", "free -h | grep Mem"], capture_output=True, text=True, timeout=5).stdout.strip()
        disk = subprocess.run(["bash", "-c", "df -h /data | tail -1"], capture_output=True, text=True, timeout=5).stdout.strip()
        report.append("Memory: %s" % mem)
        report.append("Disk: %s" % disk)
    except Exception:
        pass

    # 5. Scheduled task status
    try:
        jobs_file = os.path.join(os.path.dirname(ctx["workspace"]), "jobs.json")
        with open(jobs_file, "r", encoding="utf-8") as f:
            jobs = json.load(f)
        report.append("\n== Scheduled Tasks (%d) ==" % len(jobs))
        for j in jobs:
            cron = j.get("cron_expr", "")
            last = j.get("last_run")
            last_str = datetime.fromtimestamp(last, CST).strftime("%H:%M") if last else "never"
            report.append("  - %s (%s) last: %s" % (j["name"], cron, last_str))
    except Exception as e:
        report.append("\n== Scheduled Tasks ==\nRead failed: %s" % e)

    # 6. Memory file status
    memory_dir = os.path.join(ctx["workspace"], "memory")
    memory_md = os.path.join(memory_dir, "MEMORY.md")
    today_log = os.path.join(memory_dir, "%s.md" % today)
    report.append("\n== Memory Files ==")
    if os.path.exists(memory_md):
        mtime = datetime.fromtimestamp(os.path.getmtime(memory_md), CST)
        size_kb = os.path.getsize(memory_md) / 1024
        report.append("MEMORY.md: %.1fKB, last updated %s" % (size_kb, mtime.strftime("%Y-%m-%d %H:%M")))
    if os.path.exists(today_log):
        size_kb = os.path.getsize(today_log) / 1024
        report.append("Today's log: %.1fKB" % size_kb)
    else:
        report.append("Today's log: not created")

    return "\n".join(report)


# --- Diagnostics Tool ---

@tool("diagnose", "Diagnose system problems. Check session file health, MCP server connection status, "
      "recent error log details. Call this first when encountering 400 errors, MCP tool unavailability, or any anomaly.",
      {"target": {"type": "string", "description": "Diagnosis target: 'session', 'mcp', 'errors', 'all'"}},
      ["target"])
def tool_diagnose(args, ctx):
    target = args.get("target", "all")
    report = []

    if target in ("session", "all"):
        report.append("== Session File Health Check ==")
        sessions_dir = os.path.join(os.path.dirname(ctx["workspace"]), "sessions")
        if os.path.isdir(sessions_dir):
            for fname in sorted(os.listdir(sessions_dir)):
                if not fname.endswith(".json"):
                    continue
                fpath = os.path.join(sessions_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        msgs = json.load(f)
                    issues = []
                    if msgs and msgs[0].get("role") == "tool":
                        issues.append("Starts with orphan tool message (causes LLM 400)")
                    if msgs and msgs[0].get("role") == "assistant" and msgs[0].get("tool_calls"):
                        issues.append("Starts with assistant+tool_calls (missing tool results, causes 400)")
                    # Check tool messages have matching tool_call_id
                    tc_ids = set()
                    for m in msgs:
                        for tc in m.get("tool_calls", []):
                            tc_ids.add(tc.get("id", ""))
                    orphan_tools = 0
                    for m in msgs:
                        if m.get("role") == "tool" and m.get("tool_call_id") not in tc_ids:
                            orphan_tools += 1
                    if orphan_tools:
                        issues.append("%d tool messages with no matching tool_call_id" % orphan_tools)
                    total_bytes = sum(len(json.dumps(m)) for m in msgs)
                    status = "ISSUES" if issues else "OK"
                    report.append("  %s: %d msgs, %d bytes, %s" % (fname, len(msgs), total_bytes, status))
                    for issue in issues:
                        report.append("    WARNING: %s" % issue)
                        report.append("    Fix: use edit_file/write_file to clean session, or delete to let system rebuild")
                except Exception as e:
                    report.append("  %s: read failed (%s)" % (fname, e))
        else:
            report.append("  sessions directory does not exist")

    if target in ("mcp", "all"):
        report.append("\n== MCP Server Status ==")
        try:
            import mcp_client
            config_path = os.environ.get("AGENT_CONFIG",
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json"))
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            configured = config.get("mcp_servers", {})
            connected = set(mcp_client._servers.keys())

            for name, srv_config in configured.items():
                if name in connected:
                    server = mcp_client._servers[name]
                    tools_count = len(server._tools)
                    alive = "process running" if (server._proc and server._proc.poll() is None) else "no process (HTTP)" if server.transport == "http" else "process exited"
                    report.append("  %s: connected, %d tools, %s" % (name, tools_count, alive))
                else:
                    transport = srv_config.get("transport", "stdio")
                    cmd = srv_config.get("command", "")
                    srv_args = srv_config.get("args", [])
                    report.append("  %s: NOT connected!" % name)
                    report.append("    Config: %s %s %s" % (transport, cmd, " ".join(str(a) for a in srv_args)))
                    if transport == "stdio":
                        report.append("    Debug steps:")
                        report.append("      1. exec: which %s  -- confirm command exists" % cmd)
                        report.append("      2. exec: timeout 5 %s %s 2>&1 | head -5  -- check startup errors" % (cmd, " ".join(str(a) for a in srv_args)))

            if not configured:
                report.append("  No mcp_servers configured in config.json")
        except ImportError:
            report.append("  mcp_client module not loaded")
        except Exception as e:
            report.append("  Check failed: %s" % e)

    if target in ("errors", "all"):
        report.append("\n== Recent Error Details ==")
        try:
            cmd = 'journalctl -u agent --no-pager -n 500 --since "1 hour ago" | grep -B 1 -A 2 "ERROR\\|400\\|Bad Request" | tail -30'
            result = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=10)
            errors = result.stdout.strip()
            if errors:
                report.append(errors)
            else:
                report.append("  No errors in last hour")
        except Exception as e:
            report.append("  Read failed: %s" % e)

    return "\n".join(report)


# --- Plugin Management Tools (Self-Extension) ---

@tool("create_tool", "Create a new custom tool plugin. Code is saved to plugins/ directory and hot-loaded immediately. "
      "Persists across restarts. Use @tool decorator in code to register tools. "
      "Available variables: tool (decorator), log (logger).",
      {"name": {"type": "string", "description": "Tool name (used as filename, e.g. 'weather' creates plugins/weather.py)"},
       "code": {"type": "string", "description": "Complete Python plugin code with imports, @tool decorator, and function definition"}},
      ["name", "code"])
def tool_create_tool(args, ctx):
    name = args["name"]
    code = args["code"]

    # Validate tool name
    if not name.replace("_", "").isalnum():
        return "[error] Tool name can only contain letters, digits, and underscores"

    # Protect built-in tools: only allow overwriting existing plugins
    plugin_path = os.path.join(_plugins_dir, "%s.py" % name)
    if name in _registry and not os.path.exists(plugin_path):
        return "[error] Cannot overwrite built-in tool '%s'" % name

    # Try loading first to validate code
    try:
        _exec_plugin(code, "%s.py" % name)
    except Exception as e:
        return "[error] Code execution failed: %s" % e

    # Validation passed, persist to disk
    os.makedirs(_plugins_dir, exist_ok=True)
    with open(plugin_path, "w", encoding="utf-8") as f:
        f.write(code)

    log.info("[plugins] created: %s.py" % name)
    return "Created and loaded custom tool '%s', saved at plugins/%s.py" % (name, name)


@tool("list_custom_tools", "List all custom tool plugins (in plugins/ directory)", {})
def tool_list_custom_tools(args, ctx):
    if not os.path.isdir(_plugins_dir):
        return "No custom tools yet. plugins/ directory does not exist."
    plugins = [f for f in sorted(os.listdir(_plugins_dir)) if f.endswith(".py")]
    if not plugins:
        return "No custom tools yet."
    lines = ["Custom tools (%d):" % len(plugins)]
    for fname in plugins:
        tool_name = fname[:-3]
        fpath = os.path.join(_plugins_dir, fname)
        size = os.path.getsize(fpath)
        status = "loaded" if tool_name in _registry else "not loaded"
        lines.append("  - %s (%s, %d bytes)" % (tool_name, status, size))
    return "\n".join(lines)


@tool("remove_tool", "Delete a custom tool plugin. Can only delete plugins/ tools, not built-in tools.",
      {"name": {"type": "string", "description": "Tool name to delete"}},
      ["name"])
def tool_remove_tool(args, ctx):
    name = args["name"]
    plugin_path = os.path.join(_plugins_dir, "%s.py" % name)

    if not os.path.exists(plugin_path):
        return "[error] Custom tool '%s' does not exist (can only delete plugins/ tools)" % name

    os.remove(plugin_path)
    # Remove from registry
    if name in _registry:
        del _registry[name]
    log.info("[plugins] removed: %s" % name)
    return "Deleted custom tool '%s'" % name


# ============================================================
#  MCP Hot-Reload Tool
# ============================================================

@tool('reload_mcp', 'Hot-reload MCP servers: re-read mcp_servers config from config.json, '
      'connect new servers, disconnect removed ones. No restart needed.', {})
def _reload_mcp(args, ctx):
    import mcp_client
    import os as _os
    config_path = _os.environ.get('AGENT_CONFIG', _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'config.json'))
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Remove old MCP tools from registry
    old_mcp_keys = [k for k in _registry if '__' in k and k.split('__')[0] in mcp_client._servers]
    for k in old_mcp_keys:
        del _registry[k]

    # Hot-reload
    added, removed, total = mcp_client.reload(config)

    # Re-register new tools
    for tool_def in mcp_client.get_all_tool_defs():
        name = tool_def['function']['name']
        _registry[name] = {
            'fn': lambda a, c, _name=name: mcp_client.execute(_name, a),
            'definition': tool_def,
        }

    parts = []
    if added:
        parts.append('Added servers: %s' % ', '.join(added))
    if removed:
        parts.append('Removed servers: %s' % ', '.join(removed))
    tools_count = len(mcp_client.get_all_tool_defs())
    parts.append('Current MCP tools: %d (from %d servers)' % (tools_count, total))
    result = "\n".join(parts)
    log.info('[mcp] reloaded: %s' % result)
    return result


# ============================================================
#  MCP Server Tool Loading
# ============================================================

def _load_mcp_servers(config):
    """Connect MCP servers, register their tools into _registry"""
    if not config.get("mcp_servers"):
        return
    import mcp_client
    mcp_client.init(config)
    for tool_def in mcp_client.get_all_tool_defs():
        name = tool_def["function"]["name"]
        _registry[name] = {
            "fn": lambda args, ctx, _name=name: mcp_client.execute(_name, args),
            "definition": tool_def,
        }
    count = len(mcp_client.get_all_tool_defs())
    if count:
        log.info("[mcp] registered %d MCP tools" % count)
