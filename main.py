"""
Script principal para ejecutar el pipeline completo de análisis de blockchain
"""
import logging
import sys
from datetime import datetime

# Importar módulos del proyecto
from database_setup import create_database, create_tables, test_connection
from corda_data_extractor import CordaDataExtractor
from data_processor import BlockchainDataProcessor
from mysql_loader import MySQLDataLoader

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('blockchain_analytics.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class BlockchainAnalyticsPipeline:
    """Pipeline principal para análisis de blockchain"""
    
    def __init__(self):
        self.extractor = CordaDataExtractor()
        self.processor = BlockchainDataProcessor()
        self.loader = MySQLDataLoader()
    
    def run_full_pipeline(self, 
                         transaction_limit: int = 100000,
                         block_limit: int = 1000,
                         contract_limit: int = 500):
        """
        Ejecutar el pipeline completo de análisis de blockchain
        
        Args:
            transaction_limit: Límite de transacciones a extraer
            block_limit: Límite de bloques a extraer
            contract_limit: Límite de contratos a extraer
        """
        start_time = datetime.now()
        logger.info("=== INICIANDO PIPELINE DE ANÁLISIS DE BLOCKCHAIN ===")
        
        try:
            # Paso 1: Configurar base de datos
            logger.info("PASO 1: Configurando base de datos...")
            self.setup_database()
            
            # Paso 2: Extraer datos de Corda
            logger.info("PASO 2: Extrayendo datos de Corda...")
            self.extract_data(transaction_limit, block_limit, contract_limit)
            
            # Paso 3: Procesar datos
            logger.info("PASO 3: Procesando datos...")
            self.process_data()
            
            # Paso 4: Cargar a MySQL
            logger.info("PASO 4: Cargando datos a MySQL...")
            self.load_to_mysql()
            
            # Paso 5: Mostrar resumen
            logger.info("PASO 5: Generando resumen...")
            self.show_final_summary()
            
            end_time = datetime.now()
            duration = end_time - start_time
            logger.info(f"=== PIPELINE COMPLETADO EN {duration} ===")
            
        except Exception as e:
            logger.error(f"Error en el pipeline: {e}")
            raise
    
    def setup_database(self):
        """Configurar base de datos MySQL"""
        try:
            create_database()
            create_tables()
            
            if test_connection():
                logger.info("Base de datos configurada exitosamente")
            else:
                raise Exception("No se pudo conectar a la base de datos")
                
        except Exception as e:
            logger.error(f"Error configurando base de datos: {e}")
            raise
    
    def extract_data(self, transaction_limit: int, block_limit: int, contract_limit: int):
        """Extraer datos de la red Corda"""
        try:
            self.extractor.extract_all_data(
                transaction_limit=transaction_limit,
                block_limit=block_limit,
                contract_limit=contract_limit
            )
            logger.info("Extracción de datos completada")
            
        except Exception as e:
            logger.error(f"Error extrayendo datos: {e}")
            raise
    
    def process_data(self):
        """Procesar datos extraídos"""
        try:
            self.processor.process_all_data()
            logger.info("Procesamiento de datos completado")
            
        except Exception as e:
            logger.error(f"Error procesando datos: {e}")
            raise
    
    def load_to_mysql(self):
        """Cargar datos procesados a MySQL"""
        try:
            self.loader.load_all_data()
            logger.info("Carga a MySQL completada")
            
        except Exception as e:
            logger.error(f"Error cargando datos a MySQL: {e}")
            raise
    
    def show_final_summary(self):
        """Mostrar resumen final del pipeline"""
        try:
            logger.info("=== RESUMEN FINAL ===")
            
            # Información de tablas
            self.loader.get_table_info()
            
            # Información de archivos generados
            import os
            from config import CSV_DIR
            
            csv_files = [f for f in os.listdir(CSV_DIR) if f.endswith('.csv')]
            logger.info(f"Archivos CSV generados: {len(csv_files)}")
            for file in csv_files:
                logger.info(f"  - {file}")
            
            logger.info("=== PRÓXIMOS PASOS ===")
            logger.info("1. Abrir Power BI Desktop")
            logger.info("2. Conectar a MySQL usando las credenciales:")
            logger.info(f"   - Servidor: {self.loader.engine.url.host}")
            logger.info(f"   - Base de datos: {self.loader.engine.url.database}")
            logger.info("   - Usuario: blockchainuser")
            logger.info("   - Contraseña: 1234")
            logger.info("3. Importar las tablas y vistas creadas")
            logger.info("4. Crear dashboards interactivos")
            
        except Exception as e:
            logger.error(f"Error mostrando resumen: {e}")

def main():
    """Función principal"""
    try:
        # Crear pipeline
        pipeline = BlockchainAnalyticsPipeline()
        
        # Ejecutar pipeline completo
        pipeline.run_full_pipeline(
            transaction_limit=100000,
            block_limit=1000,
            contract_limit=500
        )
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrumpido por el usuario")
    except Exception as e:
        logger.error(f"Error fatal en el pipeline: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
