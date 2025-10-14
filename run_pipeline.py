"""
Script de ejemplo para ejecutar el pipeline completo de Blockchain Analytics
"""
import sys
import logging
from datetime import datetime

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline_execution.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Ejecutar el pipeline completo paso a paso"""
    
    logger.info("INICIANDO PIPELINE DE BLOCKCHAIN ANALYTICS")
    logger.info("=" * 60)
    
    try:
        # Paso 1: Configurar base de datos
        logger.info("PASO 1: Configurando base de datos MySQL...")
        from database_setup import create_database, create_tables, test_connection
        
        create_database()
        create_tables()
        
        if test_connection():
            logger.info("Base de datos configurada exitosamente")
        else:
            raise Exception("Error configurando base de datos")
        
        # Paso 2: Extraer datos de Corda
        logger.info("PASO 2: Extrayendo datos de Corda testnet...")
        from corda_data_extractor import CordaDataExtractor
        
        extractor = CordaDataExtractor()
        extractor.extract_all_data(
            transaction_limit=100000,
            block_limit=1000,
            contract_limit=500
        )
        logger.info("Extracción de datos completada")
        
        # Paso 3: Procesar datos con Pandas
        logger.info("PASO 3: Procesando datos con Python/Pandas...")
        from data_processor import BlockchainDataProcessor
        
        processor = BlockchainDataProcessor()
        processor.process_all_data()
        logger.info("Procesamiento de datos completado")
        
        # Paso 4: Cargar datos a MySQL
        logger.info("PASO 4: Cargando datos a MySQL...")
        from mysql_loader import MySQLDataLoader
        
        loader = MySQLDataLoader()
        loader.load_all_data()
        loader.get_table_info()
        logger.info("Carga a MySQL completada")
        
        # Paso 5: Análisis de IA
        logger.info("PASO 5: Ejecutando análisis de IA...")
        from ai_analytics import BlockchainAIAnalytics
        
        ai_analytics = BlockchainAIAnalytics()
        ai_analytics.generate_ai_report()
        logger.info("Análisis de IA completado")
        
        # Resumen final
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETADO EXITOSAMENTE")
        logger.info("=" * 60)
        
        logger.info("PROXIMOS PASOS:")
        logger.info("1. Abrir Power BI Desktop")
        logger.info("2. Conectar a MySQL:")
        logger.info("   - Servidor: localhost")
        logger.info("   - Base de datos: blockchain_analytics")
        logger.info("   - Usuario: blockchainuser")
        logger.info("   - Contraseña: 1234")
        logger.info("3. Importar tablas y vistas")
        logger.info("4. Crear dashboards interactivos")
        logger.info("5. Implementar análisis de IA")
        
        logger.info("DATOS DISPONIBLES:")
        logger.info("- Transacciones: Tabla 'transactions'")
        logger.info("- Bloques: Tabla 'blocks'")
        logger.info("- Contratos: Tabla 'smart_contracts'")
        logger.info("- Direcciones: Tabla 'addresses'")
        logger.info("- Métricas diarias: Tabla 'daily_metrics'")
        logger.info("- Insights de IA: Tabla 'ai_insights'")
        
        logger.info("VISTAS PARA POWER BI:")
        logger.info("- v_daily_transaction_summary")
        logger.info("- v_top_addresses_by_volume")
        logger.info("- v_contract_analysis")
        logger.info("- v_network_metrics")
        
        logger.info("ARCHIVOS GENERADOS:")
        logger.info("- Logs: pipeline_execution.log")
        logger.info("- Datos CSV: data/csv/")
        logger.info("- Datos JSON: data/raw/")
        
        logger.info("Tu sistema de análisis de blockchain está listo!")
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrumpido por el usuario")
    except Exception as e:
        logger.error(f"Error en el pipeline: {e}")
        logger.error("Revisa los logs para más detalles")
        sys.exit(1)

if __name__ == "__main__":
    main()
