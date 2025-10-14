"""
Configuración del proyecto Blockchain Analytics
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Configuración de la base de datos MySQL
DB_CONFIG = {
    'host': 'localhost',
    'user': 'blockchainuser',
    'password': '1234',
    'database': 'blockchain_analytics',
    'port': 3306,
    'charset': 'utf8mb4'
}

# Configuración de Corda
CORDA_CONFIG = {
    'testnet_url': 'https://testnet.corda.network',
    'api_endpoint': '/api/v1',
    'timeout': 30
}

# Configuración de archivos
DATA_DIR = 'data'
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
CSV_DIR = os.path.join(DATA_DIR, 'csv')

# Crear directorios si no existen
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, CSV_DIR]:
    os.makedirs(directory, exist_ok=True)

# Configuración de Power BI
POWERBI_CONFIG = {
    'workspace_id': 'your_workspace_id',  # Configurar según tu workspace
    'dataset_id': 'your_dataset_id'       # Se configurará después de crear el dataset
}
