# src/utils/api_client.py

import os
import requests
from dotenv import load_dotenv

# Cargamos las variables de entorno para acceder a la API Key
load_dotenv()

# Definimos constantes para la configuración de la API
BASE_URL = "https://api.football-data.org/v4/"
API_KEY = os.getenv("API_KEY")

# Los headers son necesarios para autenticarnos en cada petición
HEADERS = {"X-Auth-Token": API_KEY}

def get_data(endpoint):
    """
    Función genérica para hacer una petición GET a un endpoint de la API.

    Args:
        endpoint (str): El endpoint específico al que se quiere llamar (ej. 'competitions').

    Returns:
        dict: Los datos en formato JSON si la petición es exitosa, None si falla.
    """

    # Asegurarnos de que la API Key se cargó correctamente
    if not API_KEY:
        print("Error Crítico: La variable de entorno API_KEY no está definida.")
        print("Asegúrate de que está en tu archivo .env y que el archivo se está cargando.")
        return None

    url = BASE_URL + endpoint
    try:
        response = requests.get(url, headers=HEADERS)
        # Lanza un error si la respuesta no fue exitosa (ej. 404, 403)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error al llamar a la API: {e}")
        return None

# --- Bloque de prueba ---
if __name__ == '__main__':
    print("Realizando prueba de conexión a la API de football-data.org...")
    
    data = get_data("competitions")
    
    if data:
        print("\n==========================")
        print("   ¡¡CONEXIÓN API EXITOSA!!")
        print("==========================\n")
        
        count = data.get('count', 0)
        competitions = data.get('competitions', [])
        
        print(f"Se encontraron {count} competiciones en total.")
        print("Mostrando las primeras 5:")
        for competition in competitions[:5]:
            print(f"- {competition['name']} (Código: {competition.get('code', 'N/A')})")
    else:
        print("\n--- CONEXIÓN API FALLIDA ---")
        print("Verifica que tu API_KEY en el archivo .env sea correcta.")
