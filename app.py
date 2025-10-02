import os
import asyncio
import json
import hashlib
import re
import time
import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Response, Depends
from pydantic import BaseModel, Field
import asyncpg
import aiohttp
from unidecode import unidecode
import redis.asyncio as redis  # usar SIEMPRE el cliente async

# =======================
# Configuración de entorno
# =======================
DB_USER  = os.getenv("POSTGRES_USER", "user")
DB_PASS  = os.getenv("POSTGRES_PASSWORD", "password")
DB_HOST  = os.getenv("DB_HOST", "db")
DB_PORT  = os.getenv("DB_PORT", "5432")
DB_NAME  = os.getenv("POSTGRES_DB", "yahoo_answers_db")
DB_URL   = os.getenv("DB_URL", f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

REDIS_URL = os.getenv("REDIS_URL", "redis://cache:6379/0")
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))

LLM_PROVIDER   = os.getenv("LLM_PROVIDER", "gemini")
LLM_MODEL      = os.getenv("LLM_MODEL", "gemini-2.5-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_BASE    = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com")

app: FastAPI = FastAPI(
    title="Servicio Principal: Cache, LLM, Score",
    description="Consultas con caché Redis, búsqueda en Postgres, Gemini y scoring.",
)

# Globals
r: Optional[redis.Redis] = None  # Usamos 'r' como el cliente global de Redis
pool: Optional[asyncpg.Pool] = None

# =======================
# Utilidades
# =======================
def norm_query(q: str) -> str:
    # Ya incluye .strip() y .lower()
    q = unidecode(q).strip().lower()
    return re.sub(r"\s+", " ", q)

def cache_key_for_qid(q: str) -> str:
    h = hashlib.sha256(norm_query(q).encode("utf-8")).hexdigest()
    return f"qid:v1:{h}"

def cache_key_for_answer(q: str, qid: int) -> str:
    raw = f"{norm_query(q)}|{LLM_MODEL}|v2|{qid}"
    h = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"ans:v2:{h}"

# ... (Las funciones get_db_pool, get_db_connection, fts_top_id, etc., permanecen igual) ...

async def get_db_pool() -> asyncpg.Pool:
    global pool
    if pool is None:
        print("Intentando crear pool de conexiones DB...")
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                pool = await asyncpg.create_pool(DB_URL, min_size=1, max_size=10, timeout=15)
                print("Pool de conexiones DB creado exitosamente.")
                break
            except Exception as e:
                if attempt < max_attempts - 1:
                    wait = 2 ** attempt
                    print(f"ERROR DB (intento {attempt+1}/{max_attempts}). Reintentando en {wait}s. Detalle: {e}")
                    await asyncio.sleep(wait)
                else:
                    print(f"FATAL DB: no se pudo crear pool tras {max_attempts} intentos: {e}")
                    raise HTTPException(503, f"Servicio DB no disponible: {e}")
    return pool

async def get_db_connection():
    db_pool = await get_db_pool()
    conn = await db_pool.acquire()
    try:
        yield conn
    finally:
        await db_pool.release(conn)

async def get_redis_client() -> redis.Redis:
    global r
    if r is None:
        # CORRECCIÓN: Usar la inicialización ASÍNCRONA de redis.asyncio
        r = redis.from_url(REDIS_URL, decode_responses=True)
        try:
            await r.ping()
        except Exception as e:
            print(f"FATAL Redis: {e}")
            raise HTTPException(503, f"Servicio Cache no disponible: {e}")
    return r

# =======================
# Búsqueda (trigram + unaccent)
# =======================
async def fts_top_id(conn: asyncpg.Connection, qtext: str) -> Optional[int]:
    # Buscar por título usando índice GIN trigram con immutable_unaccent
    sql_title = """
    WITH qs AS (SELECT immutable_unaccent(lower($1)) AS q)
    SELECT id
    FROM yahoo_dataset
    WHERE immutable_unaccent(lower(question_title)) % (SELECT q FROM qs)
    ORDER BY similarity(immutable_unaccent(lower(question_title)), (SELECT q FROM qs)) DESC
    LIMIT 1;
    """
    row = await conn.fetchrow(sql_title, qtext)
    if row:
        return int(row["id"])

    # Fallback: buscar en contenido
    sql_content = """
    WITH qs AS (SELECT immutable_unaccent(lower($1)) AS q)
    SELECT id
    FROM yahoo_dataset
    WHERE immutable_unaccent(lower(question_content)) % (SELECT q FROM qs)
    ORDER BY similarity(immutable_unaccent(lower(question_content)), (SELECT q FROM qs)) DESC
    LIMIT 1;
    """
    row = await conn.fetchrow(sql_content, qtext)
    return int(row["id"]) if row else None

async def get_best_answer(conn: asyncpg.Connection, qid: int) -> Optional[str]:
    try:
        row = await conn.fetchrow("SELECT best_answer FROM yahoo_dataset WHERE id=$1", qid)
        if row and row["best_answer"]:
            return row["best_answer"]
    except Exception as e:
        print(f"Error al obtener best_answer (QID={qid}): {e}")
    return None

# =======================
# Persistencia de inferencias
# =======================
async def upsert_inference(conn: asyncpg.Connection, qid: int, model: str, answer: str, score: float = None):
    try:
        await conn.execute("""
            INSERT INTO inferences(question_id, llm_model, llm_answer, score, last_asked_at)
            VALUES($1, $2, $3, $4, now())
        """, qid, model, answer, score)
    except Exception as e:
        print(f"Error al insertar inferencia: {e}")

async def increment_hits(conn: asyncpg.Connection, qid: int, model: str):
    try:
        await conn.execute("""
            UPDATE inferences
            SET hits = hits + 1, last_asked_at = now()
            WHERE id = (
              SELECT id FROM inferences
              WHERE question_id = $1 AND llm_model = $2
              ORDER BY id DESC
              LIMIT 1
            );
        """, qid, model)
    except Exception as e:
        print(f"Error al actualizar hits: {e}")

# =======================
# Scoring (ROUGE-L F1)
# =======================
def _lcs_len(a_words, b_words):
    n, m = len(a_words), len(b_words)
    dp = [0]*(m+1)
    for i in range(1, n+1):
        prev = 0
        for j in range(1, m+1):
            tmp = dp[j]
            if a_words[i-1] == b_words[j-1]:
                dp[j] = max(dp[j], prev + 1)
            else:
                dp[j] = max(dp[j], dp[j-1])
            prev = tmp
    return dp[m]

def rouge_l_f1(pred: str, ref: str) -> float:
    if not pred or not ref:
        return 0.0
    A, B = pred.split(), ref.split()
    lcs = _lcs_len(A, B)
    if lcs == 0:
        return 0.0
    prec = lcs / max(1, len(A))
    rec  = lcs / max(1, len(B))
    beta2 = 1.2**2
    denom = (rec + beta2 * prec)
    if denom == 0:
        return 0.0
    return (1 + beta2) * prec * rec / denom

# =======================
# LLM (Gemini)
# =======================
async def gemini_generate(prompt: str) -> str:
    if not GEMINI_API_KEY:
        raise HTTPException(500, "GEMINI_API_KEY no configurada.")

    url = f"{GEMINI_BASE}/v1beta/models/{LLM_MODEL}:generateContent"
    headers = {
        "x-goog-api-key": GEMINI_API_KEY,
        "Content-Type": "application/json",
    }
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "systemInstruction": {
            "parts": [{"text": "You are an expert question answering system. Provide a concise, direct, and factual answer based on the user's question. Do not add introductory or concluding remarks."}]
        }
    }
    timeout = aiohttp.ClientTimeout(total=60)

    for attempt in range(3):
        try:
            async with aiohttp.ClientSession(timeout=timeout) as s:
                async with s.post(url, headers=headers, json=body) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        try:
                            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
                        except (KeyError, IndexError, TypeError):
                            print(f"Advertencia: Respuesta vacía o mal formada de Gemini: {data}")
                            return ""
                    elif resp.status == 429:
                        wait_time = 2 ** attempt
                        print(f"Rate Limit 429. Reintento en {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        txt = await resp.text()
                        print(f"Error {resp.status} de Gemini: {txt}")
                        raise HTTPException(resp.status, f"Gemini error: {txt}")
        except asyncio.TimeoutError:
            print(f"Timeout en intento {attempt+1}")
        except aiohttp.ClientError as e:
            print(f"HTTP client error intento {attempt+1}: {e}")

    raise HTTPException(500, "Fallo en Gemini tras 3 intentos.")

# =======================
# Lifecycle & health
# =======================
@app.on_event("startup")
async def _startup():
    print("Iniciando conexiones de DB y Redis...")
    try:
        await get_redis_client()
        print("Redis OK")
    except HTTPException:
        print("Advertencia: Redis no disponible en arranque.")
    try:
        await get_db_pool()
        print("DB OK")
    except HTTPException as e:
        print(f"ERROR FATAL DB en arranque: {e}")

@app.on_event("shutdown")
async def _shutdown():
    if pool:
        await pool.close()
    if r:
        await r.close()

@app.get("/healthz", tags=["Monitoreo"])
async def health():
    status = {"db": False, "redis": False}
    # Redis
    try:
        rd = await get_redis_client()
        status["redis"] = bool(await rd.ping())
    except Exception:
        pass
    # DB
    try:
        p = await get_db_pool()
        async with p.acquire() as conn:
            status["db"] = (await conn.fetchval("SELECT 1")) == 1
    except Exception:
        pass

    if status["db"] and status["redis"]:
        return {"status": "OK", **status}
    raise HTTPException(503, detail={"status": "DEGRADED", **status})

# =======================
# /db-status
# =======================
@app.get("/db-status", tags=["Monitoreo"])
async def get_total_question_count(conn: asyncpg.Connection = Depends(get_db_connection)):
    try:
        count = await conn.fetchval("SELECT COUNT(*) FROM yahoo_dataset")
        return {"message": "Total de preguntas cargadas en la BD", "count": count}
    except asyncpg.exceptions.UndefinedTableError:
        raise HTTPException(500, "La tabla 'yahoo_dataset' no existe.")
    except Exception as e:
        print(f"ERROR DB /db-status: {e}")
        raise HTTPException(500, f"Fallo al contar registros: {e}")

# =======================
# /answer
# =======================
@app.get("/answer", tags=["API Principal"])
async def answer(q: str, response: Response):
    if not q or not q.strip():
        raise HTTPException(400, "El parámetro 'q' es obligatorio.")

    rd = await get_redis_client()
    db_pool = await get_db_pool()

    # 1) QID desde caché o DB
    # Nota: cache_key_for_qid ya usa norm_query, lo que incluye .lower().strip()
    key_qid = cache_key_for_qid(q)
    qid_cached = await rd.get(key_qid)
    if qid_cached:
        qid = int(qid_cached)
    else:
        async with db_pool.acquire() as conn:
            qid = await fts_top_id(conn, q)
        if qid is None:
            raise HTTPException(404, "No se encontró una pregunta candidata en el dataset de Yahoo!")
        await rd.set(key_qid, str(qid), ex=CACHE_TTL)

    # 2) Respuesta desde caché
    # Nota: cache_key_for_answer ya usa norm_query.
    key_ans = cache_key_for_answer(q, qid)
    cached = await rd.get(key_ans)
    if cached:
        response.headers["X-Cache"] = "HIT"
        data = json.loads(cached)
        qid_for_hit = int(data.get("question_id", 0))
        if qid_for_hit:
            async with db_pool.acquire() as conn:
                await increment_hits(conn, qid_for_hit, LLM_MODEL)
        return data

    # 3) MISS → generar con LLM + score
    response.headers["X-Cache"] = "MISS"
    async with db_pool.acquire() as conn:
        ref = await get_best_answer(conn, qid)

    prompt = f"Responde clara y brevemente a la siguiente pregunta de Yahoo! Answers:\n\nPregunta: {q}\n\nRespuesta de IA:"
    llm_ans = await gemini_generate(prompt)
    score = rouge_l_f1(llm_ans, ref or "")

    async with db_pool.acquire() as conn:
        await upsert_inference(conn, qid, LLM_MODEL, llm_ans, score)

    payload = {
        "question_id": qid,
        "model": LLM_MODEL,
        "answer": llm_ans,
        "score": round(score, 4),
        "best_answer_ref_exists": bool(ref),
        "ref_answer_length": len(ref.split()) if ref else 0
    }
    await rd.set(key_ans, json.dumps(payload, ensure_ascii=False), ex=CACHE_TTL)
    return payload

# =======================
# Métricas de cache (experimentos)
# =======================
class BeginPayload(BaseModel):
    run_id: str = Field(..., description="Identificador de la corrida, ej: lfu-poisson-30qpm-200")
    distribution: str | None = Field(None, description="poisson | pareto | deterministic ...")
    qpm: int | None = None
    total_q: int | None = None
    ttl_seconds: int | None = None
    cache_size_mb: int | None = None
    notes: str | None = None

class EndPayload(BaseModel):
    run_id: str

async def _redis_hits_misses_policy():
    # Usamos la conexión global 'r'
    info = await r.info("stats")
    hits = int(info.get("keyspace_hits", 0))
    miss = int(info.get("keyspace_misses", 0))
    pol = (await r.config_get("maxmemory-policy")).get("maxmemory-policy", "unknown")
    return hits, miss, pol

@app.post("/metrics/begin")
async def metrics_begin(p: BeginPayload):
    # Aseguramos que 'r' esté disponible
    await get_redis_client()
    hits, miss, pol = await _redis_hits_misses_policy()
    started_at = datetime.datetime.utcnow().isoformat()

    key = f"metrics:baseline:{p.run_id}"
    await r.hset(key, mapping={
        "hits": hits, "misses": miss, "policy": pol, "started_at": started_at,
        "distribution": p.distribution or "", "qpm": p.qpm or 0,
        "total_q": p.total_q or 0, "ttl_seconds": p.ttl_seconds or 0,
        "cache_size_mb": p.cache_size_mb or 0, "notes": p.notes or ""
    })
    await r.expire(key, 24*3600)
    return {"ok": True, "run_id": p.run_id, "policy": pol, "baseline": {"hits": hits, "misses": miss}, "started_at": started_at}

@app.post("/metrics/end")
async def metrics_end(p: EndPayload):
    # Aseguramos que 'r' esté disponible
    await get_redis_client()
    key = f"metrics:baseline:{p.run_id}"
    base = await r.hgetall(key)
    if not base:
        return {"ok": False, "error": f"No baseline for run_id={p.run_id}. Llama a /metrics/begin primero."}

    h2, m2, pol2 = await _redis_hits_misses_policy()
    ended_at = datetime.datetime.utcnow().isoformat()

    h1 = int(base.get("hits", 0))
    m1 = int(base.get("misses", 0))
    dh = max(h2 - h1, 0)
    dm = max(m2 - m1, 0)
    tot = dh + dm
    hr = round(dh / tot, 4) if tot > 0 else 0.0

    conn = await asyncpg.connect(DB_URL)
    try:
        await conn.execute("""
          INSERT INTO cache_experiments
          (run_id, policy, distribution, qpm, total_q, ttl_seconds, cache_size_mb,
           hits, misses, hit_rate, started_at, ended_at, notes)
          VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
        """,
        p.run_id,
        base.get("policy", pol2),
        (base.get("distribution") or None),
        int(base.get("qpm", 0)) or None,
        int(base.get("total_q", 0)) or None,
        int(base.get("ttl_seconds", 0)) or None,
        int(base.get("cache_size_mb", 0)) or None,
        dh, dm, hr,
        datetime.datetime.fromisoformat(base["started_at"]),
        datetime.datetime.fromisoformat(ended_at),
        (base.get("notes") or None)
        )
    finally:
        await conn.close()

    # opcional: limpiar baseline
    # await r.delete(key)

    return {
        "ok": True,
        "run_id": p.run_id,
        "policy": base.get("policy", pol2),
        "hits": dh, "misses": dm, "hit_rate": hr,
        "started_at": base["started_at"], "ended_at": ended_at
    }
