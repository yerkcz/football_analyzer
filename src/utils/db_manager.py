# src/utils/db_manager.py

import os
import psycopg2
from dotenv import load_dotenv

# Cargamos las variables de entorno desde el archivo .env
load_dotenv()

def get_db_connection():
    """
    Establece y devuelve una conexión a la base de datos PostgreSQL en Neon.
    Utiliza la URI de la base de datos desde las variables de entorno.
    """
    try:
        conn = psycopg2.connect(os.getenv("DATABASE_URI"))
        return conn
    except Exception as e:
        print(f"Error Crítico: No se pudo conectar a la base de datos.")
        print(f"Detalle: {e}")
        exit()

# --- Bloque de prueba ---
if __name__ == '__main__':
    print("Realizando prueba de conexión a la base de datos (Neon.tech)...")
    connection = get_db_connection()
    if connection:
        print("\n==========================")
        print("   ¡¡CONEXIÓN EXITOSA!!")
        print("==========================\n")
        cursor = connection.cursor()
        # Una consulta simple para verificar que estamos conectados a la BD correcta
        cursor.execute("SELECT version();")
        db_version = cursor.fetchone()
        print(f"Conectado a: {db_version[0]}")
        
        cursor.close()
        connection.close()
        print("\nConexión cerrada correctamente.")
