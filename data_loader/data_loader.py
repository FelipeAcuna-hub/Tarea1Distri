import pandas as pd
import asyncpg
import asyncio
import os
import time

# --- Configuración del Entorno ---
DB_USER     = os.getenv("POSTGRES_USER", "user")
DB_PASS     = os.getenv("POSTGRES_PASSWORD", "password")
DB_HOST     = os.getenv("DB_HOST", "db")
DB_PORT     = os.getenv("DB_PORT", "5432")
DB_NAME     = os.getenv("POSTGRES_DB", "yahoo_answers_db")
DB_URL      = os.getenv("DB_URL", f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

DATA_FILE_PATH = os.getenv("DATA_FILE_PATH", "/data/train.csv")
FORCE_RELOAD   = os.getenv("FORCE_RELOAD", "0") in ("1", "true", "True", "yes", "YES")

# ----------------------------------------------------
# SQL
# ----------------------------------------------------
CREATE_YAHOO_DATASET_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS yahoo_dataset (
    id SERIAL PRIMARY KEY,
    class_index INTEGER,
    question_title TEXT NOT NULL,
    question_content TEXT,
    best_answer TEXT
);
"""

CREATE_INFERENCES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS inferences (
    id SERIAL PRIMARY KEY,
    question_id INTEGER REFERENCES yahoo_dataset(id) ON DELETE CASCADE,
    llm_model VARCHAR(50) NOT NULL,
    llm_answer TEXT NOT NULL,
    score REAL,
    hits INTEGER DEFAULT 1,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_asked_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
"""

async def initialize_db(conn: asyncpg.Connection):
    print(">>> [DB INIT] Creando tablas...")
    await conn.execute(CREATE_YAHOO_DATASET_TABLE_SQL)
    await conn.execute(CREATE_INFERENCES_TABLE_SQL)
    print(">>> [DB INIT] Tablas creadas/verificadas exitosamente.")

async def check_data_exists(conn: asyncpg.Connection) -> bool:
    try:
        count = await conn.fetchval("SELECT COUNT(*) FROM yahoo_dataset")
        return (count or 0) > 0
    except asyncpg.exceptions.UndefinedTableError:
        return False
    except Exception as e:
        print(f"Error al verificar datos: {e}")
        return False

async def table_requires_class_index(conn: asyncpg.Connection) -> bool:
    """True si la tabla tiene columna class_index y es NOT NULL."""
    q = """
    SELECT is_nullable
    FROM information_schema.columns
    WHERE table_schema='public' AND table_name='yahoo_dataset' AND column_name='class_index'
    """
    res = await conn.fetchrow(q)
    if not res:
        return False
    return (res["is_nullable"] == "NO")

def _coerce_to_str(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        df[col] = df[col].astype(str).where(df[col].notna(), '')
    return df.fillna('')

def _read_csv_tolerant(path: str) -> pd.DataFrame:
    """
    Lee CSV con o sin encabezados.
      - 4 cols => [class_index, title, content, answer]
      - 3 cols => [title, content, answer]
      - Si ya trae nombres, los usa.
    """
    expected3 = {'question_title', 'question_content', 'best_answer'}
    expected4 = {'class_index', 'question_title', 'question_content', 'best_answer'}

    # Intento con encabezado
    df = pd.read_csv(path, encoding='latin-1', engine='python', sep=None, on_bad_lines='skip')
    print(f">>> [DATA LOAD] Columnas detectadas (intento con encabezado): {df.columns.tolist()}")

    cols = set(df.columns)
    if expected4.issubset(cols):
        return df[['class_index', 'question_title', 'question_content', 'best_answer']]
    if expected3.issubset(cols):
        return df[['question_title', 'question_content', 'best_answer']]

    # Reintento sin encabezado
    df2 = pd.read_csv(path, encoding='latin-1', engine='python', sep=None, header=None, on_bad_lines='skip')
    ncols = df2.shape[1]
    print(f">>> [DATA LOAD] CSV sin header. ncols={ncols}")

    if ncols >= 4:
        df2 = df2.iloc[:, :4]
        df2.columns = ['class_index', 'question_title', 'question_content', 'best_answer']
        return df2[['class_index', 'question_title', 'question_content', 'best_answer']]
    elif ncols == 3:
        df2.columns = ['question_title', 'question_content', 'best_answer']
        return df2
    else:
        raise ValueError(f"Formato inesperado: se esperaban 3 o 4 columnas, pero se encontraron {ncols}")

async def truncate_if_requested(conn: asyncpg.Connection):
    if FORCE_RELOAD:
        print(">>> [DATA LOAD] FORCE_RELOAD=1 → TRUNCATE yahoo_dataset;")
        await conn.execute("TRUNCATE yahoo_dataset RESTART IDENTITY;")

async def load_data(conn: asyncpg.Connection):
    print(f">>> [DATA LOAD] Intentando leer el archivo: {DATA_FILE_PATH}")

    if not os.path.exists(DATA_FILE_PATH):
        print(f"ERROR CRÍTICO: Archivo no encontrado en la ruta: {DATA_FILE_PATH}. Verifica tus volúmenes.")
        return

    try:
        df = _read_csv_tolerant(DATA_FILE_PATH)
        df = _coerce_to_str(df)

        # Normaliza class_index a entero si existe
        if 'class_index' in df.columns:
            df['class_index'] = pd.to_numeric(df['class_index'], errors='coerce').fillna(0).astype(int)

        print(f">>> [DATA LOAD] Columnas finales usadas: {df.columns.tolist()}")

        # ¿La tabla exige class_index NOT NULL?
        requires_ci = await table_requires_class_index(conn)
        print(f">>> [DATA LOAD] Tabla exige class_index NOT NULL: {requires_ci}")

        if requires_ci:
            if 'class_index' not in df.columns:
                df['class_index'] = 0
            data_to_insert = df[['class_index', 'question_title', 'question_content', 'best_answer']]
            columns = ('class_index', 'question_title', 'question_content', 'best_answer')
        else:
            if not {'question_title','question_content','best_answer'}.issubset(df.columns):
                raise ValueError("Faltan columnas requeridas: ['question_title','question_content','best_answer']")
            data_to_insert = df[['question_title', 'question_content', 'best_answer']]
            columns = ('question_title', 'question_content', 'best_answer')

        records = data_to_insert.to_records(index=False).tolist()
        print(f">>> [DATA LOAD] Registros a insertar: {len(records)}")
        if not records:
            print(">>> [DATA LOAD] No hay registros para insertar. Abortando copia.")
            return

        start = time.time()
        result = await conn.copy_records_to_table('yahoo_dataset', records=records, columns=columns)
        elapsed = time.time() - start
        print(f">>> [DATA LOAD] Carga masiva exitosa. Resultado: {result}")
        print(f">>> [DATA LOAD] Tiempo de carga: {elapsed:.2f} s.")

    except Exception as e:
        print(f"ERROR CRÍTICO DURANTE LA CARGA DE DATOS: {e}")

async def main():
    conn = None
    try:
        print(">>> [START] Conectándose a PostgreSQL...")
        for i in range(5):
            try:
                conn = await asyncpg.connect(DB_URL)
                print(">>> [START] Conexión a DB exitosa.")
                break
            except Exception as e:
                wait = 2**i
                print(f">>> [START] Esperando DB. Reintento en {wait}s... Error: {e}")
                await asyncio.sleep(wait)

        if conn is None:
            print(">>> [FATAL] Fallo al conectar a PostgreSQL después de varios intentos.")
            return

        await initialize_db(conn)

        if FORCE_RELOAD:
            await truncate_if_requested(conn)

        if not await check_data_exists(conn):
            print(">>> [DATA CHECK] La tabla está vacía. Procediendo a cargar datos.")
            await load_data(conn)
        else:
            count = await conn.fetchval("SELECT COUNT(*) FROM yahoo_dataset")
            print(f">>> [DATA CHECK] Datos ya existen. Total de registros: {count}. Omitiendo carga.")

    except Exception as e:
        print(f"ERROR INESPERADO EN main(): {e}")
    finally:
        if conn:
            await conn.close()
            print(">>> [END] Conexión a PostgreSQL cerrada.")
        else:
            print(">>> [END] No hubo conexión que cerrar.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"ERROR FATAL DE INICIO DEL SCRIPT (Runtime): {e}")
