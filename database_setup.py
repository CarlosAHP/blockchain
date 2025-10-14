"""
Configuración y creación de la base de datos MySQL para el proyecto Blockchain Analytics
"""
import mysql.connector
from mysql.connector import Error
from config import DB_CONFIG
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_database():
    """Crear la base de datos si no existe"""
    try:
        # Conectar sin especificar base de datos
        connection = mysql.connector.connect(
            host=DB_CONFIG['host'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            port=DB_CONFIG['port']
        )
        
        cursor = connection.cursor()
        
        # Crear base de datos
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_CONFIG['database']} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        logger.info(f"Base de datos '{DB_CONFIG['database']}' creada o ya existe")
        
        cursor.close()
        connection.close()
        
    except Error as e:
        logger.error(f"Error creando base de datos: {e}")
        raise

def create_tables():
    """Crear las tablas necesarias para almacenar datos de blockchain"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        cursor = connection.cursor()
        
        # Tabla de transacciones
        transactions_table = """
        CREATE TABLE IF NOT EXISTS transactions (
            id VARCHAR(255) PRIMARY KEY,
            block_hash VARCHAR(255),
            block_number BIGINT,
            transaction_index INT,
            from_address VARCHAR(255),
            to_address VARCHAR(255),
            value DECIMAL(20,8),
            gas_used BIGINT,
            gas_price DECIMAL(20,8),
            timestamp DATETIME,
            status VARCHAR(50),
            contract_address VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            INDEX idx_block_number (block_number),
            INDEX idx_timestamp (timestamp),
            INDEX idx_from_address (from_address),
            INDEX idx_to_address (to_address)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """
        
        # Tabla de bloques
        blocks_table = """
        CREATE TABLE IF NOT EXISTS blocks (
            block_number BIGINT PRIMARY KEY,
            block_hash VARCHAR(255) UNIQUE,
            parent_hash VARCHAR(255),
            timestamp DATETIME,
            gas_limit BIGINT,
            gas_used BIGINT,
            transaction_count INT,
            size_bytes INT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_timestamp (timestamp)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """
        
        # Tabla de contratos inteligentes
        contracts_table = """
        CREATE TABLE IF NOT EXISTS smart_contracts (
            address VARCHAR(255) PRIMARY KEY,
            name VARCHAR(255),
            symbol VARCHAR(50),
            decimals INT,
            total_supply DECIMAL(20,8),
            contract_type VARCHAR(100),
            deployment_tx VARCHAR(255),
            deployment_block BIGINT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            INDEX idx_contract_type (contract_type),
            INDEX idx_deployment_block (deployment_block)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """
        
        # Tabla de direcciones/wallets
        addresses_table = """
        CREATE TABLE IF NOT EXISTS addresses (
            address VARCHAR(255) PRIMARY KEY,
            address_type ENUM('wallet', 'contract', 'exchange', 'unknown') DEFAULT 'unknown',
            first_seen_block BIGINT,
            first_seen_tx VARCHAR(255),
            transaction_count INT DEFAULT 0,
            total_received DECIMAL(20,8) DEFAULT 0,
            total_sent DECIMAL(20,8) DEFAULT 0,
            balance DECIMAL(20,8) DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            INDEX idx_address_type (address_type),
            INDEX idx_transaction_count (transaction_count)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """
        
        # Tabla de métricas diarias
        daily_metrics_table = """
        CREATE TABLE IF NOT EXISTS daily_metrics (
            date DATE PRIMARY KEY,
            total_transactions INT,
            total_volume DECIMAL(20,8),
            avg_gas_price DECIMAL(20,8),
            total_gas_used BIGINT,
            unique_addresses INT,
            new_contracts INT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """
        
        # Ejecutar creación de tablas
        tables = [
            ("transactions", transactions_table),
            ("blocks", blocks_table),
            ("smart_contracts", contracts_table),
            ("addresses", addresses_table),
            ("daily_metrics", daily_metrics_table)
        ]
        
        for table_name, table_sql in tables:
            cursor.execute(table_sql)
            logger.info(f"Tabla '{table_name}' creada o ya existe")
        
        connection.commit()
        cursor.close()
        connection.close()
        
        logger.info("Todas las tablas han sido creadas exitosamente")
        
    except Error as e:
        logger.error(f"Error creando tablas: {e}")
        raise

def test_connection():
    """Probar la conexión a la base de datos"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            db_info = connection.get_server_info()
            logger.info(f"Conectado a MySQL Server versión {db_info}")
            
            cursor = connection.cursor()
            cursor.execute("SELECT DATABASE();")
            database_name = cursor.fetchone()
            logger.info(f"Base de datos actual: {database_name[0]}")
            
            cursor.close()
            connection.close()
            return True
    except Error as e:
        logger.error(f"Error conectando a MySQL: {e}")
        return False

if __name__ == "__main__":
    logger.info("Configurando base de datos...")
    create_database()
    create_tables()
    test_connection()
    logger.info("Configuración de base de datos completada")