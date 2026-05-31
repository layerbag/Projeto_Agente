import os
import psycopg
from dotenv import load_dotenv

load_dotenv()

db_url = os.getenv("DATABASE_URL", "postgresql+psycopg://postgres:my_secure_password@localhost:5432/orchestration_db")
raw_conn_string = db_url.replace("postgresql+psycopg://", "postgresql://")

print("Conectando ao banco de dados...")
try:
    with psycopg.connect(raw_conn_string) as conn:
        with conn.cursor() as cur:
            print("Removendo tabelas antigas de embeddings...")
            cur.execute("DROP TABLE IF EXISTS langchain_pg_embedding CASCADE;")
            cur.execute("DROP TABLE IF EXISTS langchain_pg_collection CASCADE;")
            conn.commit()
            print("Sucesso! As tabelas do PGVector foram removidas.")
            print("Ao indexar novos documentos, as tabelas serão recriadas automaticamente com as novas dimensões.")
except Exception as e:
    print(f"Erro ao resetar o banco de dados: {e}")
