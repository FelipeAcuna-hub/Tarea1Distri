import os
import sys
import math
import time
import random
import asyncio
import statistics
from typing import List, Optional

import aiohttp
import asyncpg

# =======================
# Configuración (ENV)
# =======================
API_URL        = os.getenv("API_URL", "http://app:8000/answer")  # en docker: http://app:8000/answer
DB_URL         = os.getenv("DB_URL", "postgresql://user:password@db:5432/yahoo_answers_db")
QPM            = int(os.getenv("QPM", "10"))
TOTAL_QUERIES  = int(os.getenv("TOTAL_QUERIES", "200"))
DISTRIBUTION   = os.getenv("DISTRIBUTION", "poisson")  # poisson | pareto | deterministic
RUN_ID         = os.getenv("RUN_ID", None)             # si no viene, se autogenera
CACHE_TTL      = int(os.getenv("CACHE_TTL", "3600"))
CACHE_MB       = int(os.getenv("CACHE_MB", "256"))

TIMEOUT_TOTAL  = 120  # s para requests de begin/end y /answer


# =======================
# FIJAR SEMILLA: Asegura que la selección aleatoria de preguntas sea determinística
# Para obtener HITs en la segunda corrida, la selección debe ser la misma.
# =======================
random.seed(42)


# =======================
# Helpers
# =======================
def interarrival_times(dist: str, qpm: int, n: int) -> List[float]:
    """Devuelve tiempos de espera (segundos) entre consultas."""
    lam = max(qpm, 1) / 60.0  # eventos/seg
    if dist == "poisson":
        return [random.expovariate(lam) for _ in range(n)]
    elif dist == "pareto":
        # Pareto bursty: alpha=1.5, normalizado a media ~ 1/lam
        alpha = 1.5
        raw = [(1.0 / max(random.random(), 1e-12)) ** (1 / alpha) for _ in range(n)]
        scale = (statistics.mean(raw) or 1.0) / (1 / lam)
        return [x / max(scale, 1e-9) for x in raw]
    else:
        # determinística
        return [1.0 / lam] * n


async def fetch_random_titles(conn: asyncpg.Connection, k: int) -> List[str]:
    """
    Obtiene k títulos al azar sin traer toda la tabla.
    Usa TABLESAMPLE y fallback a ORDER BY random().
    """
    rows = await conn.fetch("""
        SELECT question_title
        FROM yahoo_dataset
        TABLESAMPLE SYSTEM (0.1)
        LIMIT $1
    """, k)
    if len(rows) < k:
        # fallback, más costoso, pero acotado
        rows = await conn.fetch("""
            SELECT question_title
            FROM yahoo_dataset
            ORDER BY random()
            LIMIT $1
        """, k)
    return [r["question_title"] for r in rows]


async def call_answer(session: aiohttp.ClientSession, question: str):
    t0 = time.perf_counter()
    try:
        async with session.get(API_URL, params={"q": question}) as resp:
            txt = await resp.text()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            cache_status = resp.headers.get("X-Cache", "UNKNOWN")
            ok = resp.status == 200
            return ok, dt_ms, resp.status, cache_status, txt
    except Exception as e:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return False, dt_ms, 599, "ERR", str(e)


async def metrics_begin(session: aiohttp.ClientSession, run_id: str):
    base_url = API_URL.rsplit("/answer", 1)[0]
    payload = {
        "run_id": run_id,
        "distribution": DISTRIBUTION,
        "qpm": QPM,
        "total_q": TOTAL_QUERIES,
        "ttl_seconds": CACHE_TTL,
        "cache_size_mb": CACHE_MB,
        "notes": "auto begin from traffic_gen"
    }
    try:
        async with session.post(f"{base_url}/metrics/begin", json=payload) as r:
            _ = await r.text()
            if r.status != 200:
                print(f"[metrics_begin] HTTP {r.status}", file=sys.stderr)
    except Exception as e:
        print(f"[metrics_begin] error: {e}", file=sys.stderr)


async def metrics_end(session: aiohttp.ClientSession, run_id: str):
    base_url = API_URL.rsplit("/answer", 1)[0]
    try:
        async with session.post(f"{base_url}/metrics/end", json={"run_id": run_id}) as r:
            txt = await r.text()
            print("[metrics_end]", r.status, txt.strip())
    except Exception as e:
        print(f"[metrics_end] error: {e}", file=sys.stderr)


# =======================
# Main
# =======================
async def main():
    run_id = RUN_ID or f"{DISTRIBUTION}-qpm{QPM}-n{TOTAL_QUERIES}-{int(time.time())}"
    print(f"[RUN] {run_id}")
    print(f"[CFG] API_URL={API_URL}  DB_URL={DB_URL}")
    print(f"[CFG] distribution={DISTRIBUTION}  qpm={QPM}  total={TOTAL_QUERIES}")

    # pool DB
    pool = None
    for i in range(8):
        try:
            pool = await asyncpg.create_pool(DB_URL, min_size=1, max_size=2)
            break
        except Exception as e:
            print(f"[DB] esperando ({i+1}/8): {e}")
            await asyncio.sleep(2)
    if pool is None:
        print("[FATAL] No se pudo conectar a la DB.")
        return

    timeout = aiohttp.ClientTimeout(total=TIMEOUT_TOTAL)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # BEGIN (baseline en Redis, no persiste)
        await metrics_begin(session, run_id)

        ok_count = 0
        lat_ms: List[float] = []

        try:
            async with pool.acquire() as conn:
                # obtener N preguntas para esta corrida
                titles = await fetch_random_titles(conn, TOTAL_QUERIES)
                if not titles:
                    print("[FATAL] No se pudieron obtener preguntas.")
                    return

            waits = interarrival_times(DISTRIBUTION, QPM, len(titles))
            print(f"[INFO] Enviando {len(titles)} consultas...")

            for i, (q, wait_s) in enumerate(zip(titles, waits), 1):
                await asyncio.sleep(max(0.0, wait_s))
                ok, dt, code, cache, _ = await call_answer(session, q)
                lat_ms.append(dt)
                status = "OK" if ok else f"FAIL({code})"
                print(f"[{i:04d}/{len(titles)}|{status}|{cache}] {dt:.1f} ms  q='{q[:60]}...'")
                if ok:
                    ok_count += 1

        finally:
            # END (PERSISTE en Postgres)
            await metrics_end(session, run_id)

    # resumen local
    if lat_ms:
        lat_ms_sorted = sorted(lat_ms)
        p50 = lat_ms_sorted[len(lat_ms_sorted)//2]
        p90 = lat_ms_sorted[int(len(lat_ms_sorted)*0.9)]
    else:
        p50 = p90 = 0.0
    print(f"[SUMMARY] run_id={run_id}  ok={ok_count}/{len(lat_ms)}  p50={p50:.1f}ms  p90={p90:.1f}ms")

    if pool:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())