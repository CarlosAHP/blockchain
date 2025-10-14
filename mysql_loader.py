"""
Cargador de datos procesados a MySQL
"""
import pandas as pd
import mysql.connector
from mysql.connector import Error
from sqlalchemy import create_engine, text
import logging
from config import DB_CONFIG, CSV_DIR
import os

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MySQLDataLoader:
    """Clase para cargar datos procesados en MySQL"""
    
    def __init__(self):
        self.engine = None
        self.connection = None
        self._connect()
    
    def _connect(self):
        """Establecer conexión con MySQL"""
        try:
            # Crear engine de SQLAlchemy
            connection_string = f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
            self.engine = create_engine(connection_string, echo=False)
            
            # Probar conexión
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            logger.info("Conexión a MySQL establecida exitosamente")
            
        except Error as e:
            logger.error(f"Error conectando a MySQL: {e}")
            raise
    
    def load_csv_to_mysql(self, csv_file: str, table_name: str, if_exists: str = 'replace'):
        """
        Cargar archivo CSV a tabla MySQL
        
        Args:
            csv_file: Ruta del archivo CSV
            table_name: Nombre de la tabla en MySQL
            if_exists: Qué hacer si la tabla existe ('replace', 'append', 'fail')
        """
        try:
            if not os.path.exists(csv_file):
                logger.warning(f"Archivo CSV no encontrado: {csv_file}")
                return False
            
            # Leer CSV
            df = pd.read_csv(csv_file)
            
            if df.empty:
                logger.warning(f"DataFrame vacío para {csv_file}")
                return False
            
            # Cargar a MySQL
            df.to_sql(
                name=table_name,
                con=self.engine,
                if_exists=if_exists,
                index=False,
                method='multi',
                chunksize=1000
            )
            
            logger.info(f"Datos cargados en tabla '{table_name}': {len(df)} registros")
            return True
            
        except Exception as e:
            logger.error(f"Error cargando {csv_file} a {table_name}: {e}")
            return False
    
    def load_transactions(self):
        """Cargar datos de transacciones"""
        csv_file = os.path.join(CSV_DIR, 'transactions_processed.csv')
        return self.load_csv_to_mysql(csv_file, 'transactions')
    
    def load_blocks(self):
        """Cargar datos de bloques"""
        csv_file = os.path.join(CSV_DIR, 'blocks_processed.csv')
        return self.load_csv_to_mysql(csv_file, 'blocks')
    
    def load_contracts(self):
        """Cargar datos de contratos inteligentes"""
        csv_file = os.path.join(CSV_DIR, 'contracts_processed.csv')
        return self.load_csv_to_mysql(csv_file, 'smart_contracts')
    
    def load_addresses(self):
        """Cargar datos de direcciones"""
        csv_file = os.path.join(CSV_DIR, 'addresses_processed.csv')
        return self.load_csv_to_mysql(csv_file, 'addresses')
    
    def load_daily_metrics(self):
        """Cargar métricas diarias"""
        csv_file = os.path.join(CSV_DIR, 'daily_metrics_processed.csv')
        return self.load_csv_to_mysql(csv_file, 'daily_metrics')
    
    def load_all_data(self):
        """Cargar todos los datos procesados"""
        logger.info("Iniciando carga de datos a MySQL...")
        
        success_count = 0
        total_tables = 5
        
        # Cargar cada tabla
        if self.load_transactions():
            success_count += 1
        
        if self.load_blocks():
            success_count += 1
        
        if self.load_contracts():
            success_count += 1
        
        if self.load_addresses():
            success_count += 1
        
        if self.load_daily_metrics():
            success_count += 1
        
        logger.info(f"Carga completada: {success_count}/{total_tables} tablas cargadas exitosamente")
        
        if success_count == total_tables:
            self.create_indexes()
            self.create_views()
            logger.info("Todos los datos han sido cargados exitosamente")
        else:
            logger.warning("Algunas tablas no pudieron ser cargadas")
    
    def create_indexes(self):
        """Crear índices adicionales para optimizar consultas"""
        try:
            with self.engine.connect() as conn:
                # Índices para transacciones
                indexes = [
                    "CREATE INDEX IF NOT EXISTS idx_transactions_date ON transactions(date)",
                    "CREATE INDEX IF NOT EXISTS idx_transactions_value ON transactions(value)",
                    "CREATE INDEX IF NOT EXISTS idx_transactions_gas_cost ON transactions(gas_cost)",
                    "CREATE INDEX IF NOT EXISTS idx_transactions_type ON transactions(transaction_type)",
                    
                    # Índices para bloques
                    "CREATE INDEX IF NOT EXISTS idx_blocks_date ON blocks(date)",
                    "CREATE INDEX IF NOT EXISTS idx_blocks_gas_utilization ON blocks(gas_utilization)",
                    
                    # Índices para direcciones
                    "CREATE INDEX IF NOT EXISTS idx_addresses_balance ON addresses(balance)",
                    "CREATE INDEX IF NOT EXISTS idx_addresses_tx_count ON addresses(transaction_count)",
                    
                    # Índices para métricas diarias
                    "CREATE INDEX IF NOT EXISTS idx_daily_metrics_date ON daily_metrics(date)"
                ]
                
                for index_sql in indexes:
                    try:
                        conn.execute(text(index_sql))
                        conn.commit()
                    except Exception as e:
                        logger.warning(f"Error creando índice: {e}")
                
                logger.info("Índices creados exitosamente")
                
        except Exception as e:
            logger.error(f"Error creando índices: {e}")
    
    def create_views(self):
        """Crear vistas para facilitar consultas en Power BI"""
        try:
            with self.engine.connect() as conn:
                views = [
                    # Vista de resumen de transacciones por día
                    """
                    CREATE OR REPLACE VIEW v_daily_transaction_summary AS
                    SELECT 
                        date,
                        COUNT(*) as total_transactions,
                        SUM(value) as total_volume,
                        AVG(gas_price) as avg_gas_price,
                        SUM(gas_used) as total_gas_used,
                        COUNT(DISTINCT from_address) as unique_senders,
                        COUNT(DISTINCT to_address) as unique_receivers,
                        COUNT(DISTINCT contract_address) as unique_contracts
                    FROM transactions
                    GROUP BY date
                    ORDER BY date DESC
                    """,
                    
                    # Vista de top direcciones por volumen
                    """
                    CREATE OR REPLACE VIEW v_top_addresses_by_volume AS
                    SELECT 
                        address,
                        address_type,
                        transaction_count,
                        total_sent,
                        total_received,
                        balance,
                        balance_category
                    FROM addresses
                    ORDER BY ABS(balance) DESC
                    LIMIT 100
                    """,
                    
                    # Vista de análisis de contratos
                    """
                    CREATE OR REPLACE VIEW v_contract_analysis AS
                    SELECT 
                        c.address,
                        c.name,
                        c.symbol,
                        c.contract_type,
                        c.total_supply,
                        c.supply_category,
                        COUNT(t.id) as transaction_count,
                        SUM(t.value) as total_volume
                    FROM smart_contracts c
                    LEFT JOIN transactions t ON c.address = t.contract_address
                    GROUP BY c.address, c.name, c.symbol, c.contract_type, c.total_supply, c.supply_category
                    ORDER BY transaction_count DESC
                    """,
                    
                    # Vista de métricas de red
                    """
                    CREATE OR REPLACE VIEW v_network_metrics AS
                    SELECT 
                        dm.date,
                        dm.total_transactions,
                        dm.total_volume,
                        dm.avg_gas_price,
                        dm.total_gas_used,
                        dm.unique_addresses,
                        dm.new_contracts,
                        AVG(b.gas_utilization) as avg_gas_utilization,
                        AVG(b.transaction_count) as avg_tx_per_block
                    FROM daily_metrics dm
                    LEFT JOIN blocks b ON dm.date = b.date
                    GROUP BY dm.date, dm.total_transactions, dm.total_volume, 
                             dm.avg_gas_price, dm.total_gas_used, dm.unique_addresses, dm.new_contracts
                    ORDER BY dm.date DESC
                    """
                ]
                
                for view_sql in views:
                    try:
                        conn.execute(text(view_sql))
                        conn.commit()
                    except Exception as e:
                        logger.warning(f"Error creando vista: {e}")
                
                logger.info("Vistas creadas exitosamente")
                
        except Exception as e:
            logger.error(f"Error creando vistas: {e}")
    
    def get_table_info(self):
        """Obtener información de las tablas cargadas"""
        try:
            with self.engine.connect() as conn:
                tables = ['transactions', 'blocks', 'smart_contracts', 'addresses', 'daily_metrics']
                
                logger.info("=== INFORMACIÓN DE TABLAS ===")
                for table in tables:
                    result = conn.execute(text(f"SELECT COUNT(*) as count FROM {table}"))
                    count = result.fetchone()[0]
                    logger.info(f"{table}: {count} registros")
                
        except Exception as e:
            logger.error(f"Error obteniendo información de tablas: {e}")
    
    def close_connection(self):
        """Cerrar conexión con MySQL"""
        if self.engine:
            self.engine.dispose()
            logger.info("Conexión a MySQL cerrada")

def main():
    """Función principal para cargar datos a MySQL"""
    loader = MySQLDataLoader()
    try:
        loader.load_all_data()
        loader.get_table_info()
    finally:
        loader.close_connection()

if __name__ == "__main__":
    main()
