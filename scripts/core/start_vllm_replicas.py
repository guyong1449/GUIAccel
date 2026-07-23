#!/usr/bin/env python3
"""Start N vLLM replicas (one per GPU) with a round-robin proxy on front-port.

Architecture
------------
  GPU 0  →  vLLM  :base_port+0  ─┐
  GPU 1  →  vLLM  :base_port+1  ─┤  round-robin proxy  :front_port
  ...                             ┤  (OpenAI-compatible /v1/*)
  GPU N  →  vLLM  :base_port+N  ─┘

Usage
-----
  python scripts/core/start_vllm_replicas.py \\
      --gpus 0,1,2,3,4,5,6,7 \\
      --model /path/to/MAI-UI-8B \\
      --served-model-name MAI-UI-8B \\
      --front-port 8000 --base-port 8100 \\
      --gpu-memory-utilization 0.90 \\
      --max-num-batched-tokens 32768 \\
      --max-num-seqs 32 \\
      --max-num-partial-prefills 4
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import os
import signal
import subprocess
import sys
import time
from typing import Sequence


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _model_path_from_config() -> str:
    """Read base_model_path from configs/androidcontrol/default.json."""
    import json
    from pathlib import Path

    for _parent in Path(__file__).resolve().parents:
        _lib = _parent / "_lib"
        if (_lib / "repo_path.py").is_file():
            sys.path.insert(0, str(_lib))
            break
    from repo_path import repo_root

    repo_root_path = repo_root(Path(__file__))
    cfg_path = repo_root_path / "configs" / "androidcontrol" / "default.json"
    try:
        cfg = json.loads(cfg_path.read_text())
        return cfg["paths"]["base_model_path"]
    except Exception:
        return ""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Launch vLLM replica pool with proxy.")
    p.add_argument("--gpus", default="",
                   help="Comma-separated GPU indices, e.g. 0,1,2,3")
    p.add_argument("--model", default=None,
                   help="Model path (default: read from configs/androidcontrol/default.json)")
    p.add_argument("--served-model-name", default="MAI-UI-8B",
                   help="Model name exposed by the API")
    p.add_argument("--front-port", type=int, default=8000,
                   help="Port for the round-robin proxy")
    p.add_argument("--base-port", type=int, default=8100,
                   help="First backend port; subsequent replicas use base+1, base+2 …")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--max-num-batched-tokens", type=int, default=32768)
    p.add_argument("--max-num-seqs", type=int, default=32)
    p.add_argument("--max-model-len", type=int, default=None,
                   help="Max sequence length (KV cache); omit to use model default")
    p.add_argument("--dtype", default="bfloat16",
                   help="Model dtype (bfloat16 / float16 / auto)")
    p.add_argument("--trust-remote-code", action="store_true",
                   help="Pass --trust-remote-code to vLLM")
    p.add_argument("--host", default="127.0.0.1",
                   help="Host vLLM replicas bind to (use 0.0.0.0 for multi-node)")
    p.add_argument("--extra-backends", nargs="*", default=[],
                   metavar="URL",
                   help="Additional remote backend URLs to include in the proxy "
                        "(e.g. http://10.0.0.2:8100 … for multi-node inference)")
    p.add_argument("--enforce-eager", action="store_true",
                   help="Disable torch.compile and CUDA graphs (vLLM stability)")
    p.add_argument("--proxy-only", action="store_true",
                   help="Skip launching vLLM replicas; only run the proxy over "
                        "--extra-backends (all backends must already be running)")
    p.add_argument("--log-dir", default="",
                   help="If set, write each replica's stdout/stderr to "
                        "replica_<i>.out / replica_<i>.err under this directory "
                        "(keeps EngineCore crash traces from being lost in a mix)")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Launch vLLM replicas
# ---------------------------------------------------------------------------

def launch_replicas(args: argparse.Namespace) -> list[subprocess.Popen]:
    from pathlib import Path

    gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip()]
    procs: list[subprocess.Popen] = []
    # Keep log file handles alive for the process lifetime (avoid GC close).
    log_handles: list[object] = []
    log_dir = Path(args.log_dir).expanduser() if str(args.log_dir or "").strip() else None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)

    for i, gpu_id in enumerate(gpu_ids):
        port = args.base_port + i
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu_id}
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", args.model,
            "--served-model-name", args.served_model_name,
            "--port", str(port),
            "--host", args.host,
            "--gpu-memory-utilization", str(args.gpu_memory_utilization),
            "--max-num-batched-tokens", str(args.max_num_batched_tokens),
            "--max-num-seqs", str(args.max_num_seqs),
            *([] if args.max_model_len is None else ["--max-model-len", str(args.max_model_len)]),
            "--dtype", args.dtype,
            "--disable-log-requests",
        ]
        if args.trust_remote_code:
            cmd.append("--trust-remote-code")
        if args.enforce_eager:
            cmd.append("--enforce-eager")

        popen_kwargs: dict = {"env": env}
        if log_dir is not None:
            out_path = log_dir / f"replica_{i}_gpu{gpu_id}_port{port}.out"
            err_path = log_dir / f"replica_{i}_gpu{gpu_id}_port{port}.err"
            # Line-buffered text logs so EngineCore deaths remain inspectable.
            out_fh = open(out_path, "w", buffering=1)
            err_fh = open(err_path, "w", buffering=1)
            log_handles.extend((out_fh, err_fh))
            popen_kwargs["stdout"] = out_fh
            popen_kwargs["stderr"] = err_fh
            print(
                f"[replica {i}]  GPU={gpu_id}  port={port}  "
                f"logs={out_path.name}|{err_path.name}  cmd={' '.join(cmd[:6])} …",
                flush=True,
            )
        else:
            print(
                f"[replica {i}]  GPU={gpu_id}  port={port}  cmd={' '.join(cmd[:6])} …",
                flush=True,
            )
        proc = subprocess.Popen(cmd, **popen_kwargs)
        # Stash handles on the Popen object so they outlive this function.
        setattr(proc, "_skillreuse_log_handles", log_handles)
        print(f"[replica {i}]  pid={proc.pid}", flush=True)
        procs.append(proc)

    return procs


def _replica_early_exits(procs: Sequence[subprocess.Popen]) -> list[str]:
    dead: list[str] = []
    for i, proc in enumerate(procs):
        code = proc.poll()
        if code is not None:
            dead.append(f"replica[{i}] pid={proc.pid} exit={code}")
    return dead


# ---------------------------------------------------------------------------
# Wait until all replicas are healthy
# ---------------------------------------------------------------------------

def wait_for_replicas(
    ports: list[int],
    timeout: int = 600,
    procs: Sequence[subprocess.Popen] | None = None,
) -> None:
    import urllib.request
    deadline = time.time() + timeout
    remaining = set(ports)
    print(f"Waiting for {len(ports)} replicas to become healthy …", flush=True)
    while remaining and time.time() < deadline:
        if procs is not None:
            dead = _replica_early_exits(procs)
            if dead:
                raise RuntimeError(
                    "Replica process(es) exited before becoming healthy: "
                    + "; ".join(dead)
                    + ". Inspect --log-dir replica_*.err if set."
                )
        for port in list(remaining):
            try:
                url = f"http://127.0.0.1:{port}/health"
                urllib.request.urlopen(url, timeout=2)
                remaining.discard(port)
                print(f"  ✓ replica on :{port} ready", flush=True)
            except Exception:
                pass
        if remaining:
            time.sleep(3)
    if remaining:
        dead = _replica_early_exits(procs) if procs is not None else []
        detail = f" Early exits: {'; '.join(dead)}." if dead else ""
        raise RuntimeError(
            f"Replicas on ports {sorted(remaining)} did not become healthy "
            f"within {timeout}s.{detail}"
        )
    print("All replicas healthy.", flush=True)


# ---------------------------------------------------------------------------
# Round-robin proxy (aiohttp)
# ---------------------------------------------------------------------------

async def run_proxy(front_port: int, backend_urls: list[str]) -> None:
    try:
        import aiohttp
        from aiohttp import web
    except ImportError:
        sys.exit("aiohttp is required for the proxy — pip install aiohttp")

    counter = itertools.count()

    async def handle(request: web.Request) -> web.StreamResponse:
        # Health endpoint handled by the proxy itself
        if request.path == "/healthz":
            return web.Response(text="ok")

        backend = backend_urls[next(counter) % len(backend_urls)]
        target = backend + str(request.rel_url)

        body = await request.read()
        req_headers = {
            k: v for k, v in request.headers.items()
            if k.lower() not in ("host", "transfer-encoding")
        }

        async with aiohttp.ClientSession() as session:
            async with session.request(
                request.method,
                target,
                headers=req_headers,
                data=body,
                allow_redirects=False,
            ) as resp:
                # Stream the response back
                response = web.StreamResponse(
                    status=resp.status,
                    headers={
                        k: v for k, v in resp.headers.items()
                        if k.lower() not in ("transfer-encoding", "content-encoding")
                    },
                )
                await response.prepare(request)
                async for chunk in resp.content.iter_chunked(32768):
                    await response.write(chunk)
                await response.write_eof()
                return response

    app = web.Application(client_max_size=100 * 1024 * 1024)  # 100MB，适配 base64 截图
    app.router.add_route("*", "/{path_info:.*}", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", front_port)
    await site.start()
    print(f"Proxy listening on :{front_port}  →  {backend_urls}", flush=True)

    # Keep running until SIGTERM / SIGINT
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)
    await stop_event.wait()
    await runner.cleanup()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # 未指定 --model 时从 config 自动读取
    if not args.model:
        args.model = _model_path_from_config()
    if not args.model:
        sys.exit("ERROR: 无法确定模型路径，请用 --model 指定或检查 configs/androidcontrol/default.json")
    print(f"Model: {args.model}", flush=True)

    gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip()]
    ports = [args.base_port + i for i in range(len(gpu_ids))]
    local_host = args.host if args.host != "0.0.0.0" else "127.0.0.1"
    backend_urls = [f"http://{local_host}:{p}" for p in ports]
    # Append any additional remote backends (multi-node distributed inference)
    backend_urls = backend_urls + list(args.extra_backends)

    # validate
    if not args.proxy_only and not args.gpus:
        sys.exit("--gpus is required unless --proxy-only is set")

    # proxy-only mode: skip replica launch, run proxy over --extra-backends only
    if args.proxy_only:
        proxy_backends = list(args.extra_backends)
        if not proxy_backends:
            sys.exit("--proxy-only requires at least one --extra-backends URL")
        print(f"Proxy-only mode over {len(proxy_backends)} backends", flush=True)
        asyncio.run(run_proxy(args.front_port, proxy_backends))
        return

    # Launch all replicas
    procs = launch_replicas(args)

    def _shutdown_replicas() -> int:
        """Terminate replicas and return the worst non-zero exit code seen."""
        print("\nShutting down replicas …", flush=True)
        worst = 0
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for i, p in enumerate(procs):
            try:
                code = p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
                code = p.wait(timeout=5)
            if code not in (0, None, -signal.SIGTERM, -15):
                print(
                    f"[replica {i}] exited with code={code} (pid={p.pid})",
                    flush=True,
                )
                worst = code if worst == 0 else worst
        dead_before = _replica_early_exits(procs)
        if dead_before:
            # Already-dead replicas are reported above via poll/wait codes.
            print(
                "Replica exit summary: " + "; ".join(dead_before),
                flush=True,
            )
        return int(worst or 0)

    def _sigterm_handler(signum=None, frame=None):
        # SIGTERM: 完整退出，停代理 + 杀副本；保留非零退出码便于 Slurm/上层检测
        worst = _shutdown_replicas()
        sys.exit(1 if worst not in (0, -signal.SIGTERM, -15) else 0)

    def _sigint_handler(signum=None, frame=None):
        # SIGINT (Ctrl+C / terminal Ctrl+C): 只停代理，副本继续运行
        # nohup 不防 SIGINT，用 setsid 启动可彻底避免此问题
        print("\nSIGINT: proxy stopped, replicas still running.", flush=True)
        pids = " ".join(str(p.pid) for p in procs)
        print(f"Replica PIDs: {pids}", flush=True)
        print(f"To stop all:  kill {pids}", flush=True)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _sigterm_handler)
    signal.signal(signal.SIGINT, _sigint_handler)

    # Wait for all replicas to be healthy before opening the proxy
    try:
        wait_for_replicas(ports, timeout=600, procs=procs)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", flush=True)
        _shutdown_replicas()
        sys.exit(1)

    # Start the round-robin proxy (blocks until signal)
    # Note: asyncio内部信号覆盖不影响副本，proxy退出时副本保持运行
    asyncio.run(run_proxy(args.front_port, backend_urls))
    # If proxy returns without signal handler exit, still tear down and surface errors.
    worst = _shutdown_replicas()
    if worst not in (0, -signal.SIGTERM, -15):
        sys.exit(1)


if __name__ == "__main__":
    main()
