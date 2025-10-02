-- Extensiones
CREATE EXTENSION IF NOT EXISTS unaccent;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Función inmutable para índices/búsqueda
CREATE OR REPLACE FUNCTION immutable_unaccent(text)
RETURNS text
LANGUAGE sql
IMMUTABLE
PARALLEL SAFE
AS $$
  SELECT unaccent('public.unaccent', $1)
$$;

-- Índices trigram normalizados
CREATE INDEX IF NOT EXISTS idx_yahoo_title_trgm
  ON yahoo_dataset
  USING gin ( (immutable_unaccent(lower(question_title))) gin_trgm_ops );

CREATE INDEX IF NOT EXISTS idx_yahoo_content_trgm
  ON yahoo_dataset
  USING gin ( (immutable_unaccent(lower(question_content))) gin_trgm_ops );

CREATE TABLE IF NOT EXISTS cache_experiments (
  id            SERIAL PRIMARY KEY,
  run_id        TEXT NOT NULL,
  policy        TEXT NOT NULL,
  distribution  TEXT,
  qpm           INT,
  total_q       INT,
  ttl_seconds   INT,
  cache_size_mb INT,
  hits          BIGINT NOT NULL,
  misses        BIGINT NOT NULL,
  hit_rate      NUMERIC(6,4) NOT NULL,
  started_at    TIMESTAMPTZ NOT NULL,
  ended_at      TIMESTAMPTZ NOT NULL,
  notes         TEXT
);

