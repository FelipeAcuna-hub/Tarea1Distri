1) Preparar Base de Datos (PostgreSQL)

createdb yahoo_answers_db
psql -h localhost -U $(whoami) -d yahoo_answers_db -c \
"CREATE TABLE IF NOT EXISTS yahoo_dataset (id SERIAL PRIMARY KEY, class_index INT, question_title TEXT, question_content TEXT, best_answer TEXT);"
psql -h localhost -U $(whoami) -d yahoo_answers_db -c \
"\copy yahoo_dataset(class_index, question_title, question_content, best_answer) FROM '$(pwd)/data/train.csv' DELIMITER ',' CSV HEADER;"
psql -h localhost -U $(whoami) -d yahoo_answers_db -c \
"CREATE TABLE IF NOT EXISTS inferences (id BIGSERIAL PRIMARY KEY, question_id BIGINT NOT NULL, llm_model TEXT NOT NULL, llm_answer TEXT, score DOUBLE PRECISION, hits INT NOT NULL DEFAULT 0, created_at TIMESTAMPTZ NOT NULL DEFAULT now(), last_asked_at TIMESTAMPTZ);"

2) Entorno Gemini y Cache

docker compose --env-file .env.gemini -f docker-compose.yml up --build -d redis api
