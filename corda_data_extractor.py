"""
Extractor de datos de la red testnet de Corda
"""
import requests
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict
import logging
from config import CORDA_CONFIG, RAW_DATA_DIR
import os

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CordaDataExtractor:
    """Clase para extraer datos de la red testnet de Corda"""
    
    def __init__(self):
        self.base_url = CORDA_CONFIG['testnet_url']
        self.api_endpoint = CORDA_CONFIG['api_endpoint']
        self.timeout = CORDA_CONFIG['timeout']
        self.session = requests.Session()
        
    def get_transactions(self, limit: int = 1000, offset: int = 0) -> List[Dict]:
        """
        Obtener transacciones de la red Corda
        
        Args:
            limit: Número máximo de transacciones a obtener
            offset: Número de transacciones a omitir
            
        Returns:
            Lista de diccionarios con datos de transacciones
        """
        try:
            # Simular datos de transacciones (en un caso real, esto sería una llamada a la API de Corda)
            transactions = []
            
            # Generar datos de ejemplo para simular transacciones de Corda
            for i in range(offset, offset + limit):
                transaction = {
                    'id': f"corda_tx_{i:06d}",
                    'block_hash': f"corda_block_{i//100:06d}",
                    'block_number': i // 100,
                    'transaction_index': i % 100,
                    'from_address': f"corda_address_{i%50:03d}",
                    'to_address': f"corda_address_{(i+1)%50:03d}",
                    'value': round(1000 + (i * 0.1), 8),
                    'gas_used': 21000 + (i % 1000),
                    'gas_price': round(0.00000002 + (i % 100) * 0.000000001, 8),
                    'timestamp': datetime.now() - timedelta(hours=i),
                    'status': 'confirmed' if i % 10 != 0 else 'pending',
                    'contract_address': f"corda_contract_{i%20:03d}" if i % 5 == 0 else None
                }
                transactions.append(transaction)
            
            logger.info(f"Obtenidas {len(transactions)} transacciones")
            return transactions
            
        except Exception as e:
            logger.error(f"Error obteniendo transacciones: {e}")
            return []
    
    def get_blocks(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """
        Obtener bloques de la red Corda
        
        Args:
            limit: Número máximo de bloques a obtener
            offset: Número de bloques a omitir
            
        Returns:
            Lista de diccionarios con datos de bloques
        """
        try:
            blocks = []
            
            # Generar datos de ejemplo para simular bloques de Corda
            for i in range(offset, offset + limit):
                block = {
                    'block_number': i,
                    'block_hash': f"corda_block_{i:06d}",
                    'parent_hash': f"corda_block_{i-1:06d}" if i > 0 else None,
                    'timestamp': datetime.now() - timedelta(hours=i*2),
                    'gas_limit': 8000000,
                    'gas_used': 6000000 + (i % 1000000),
                    'transaction_count': 50 + (i % 100),
                    'size_bytes': 1024 + (i % 1000)
                }
                blocks.append(block)
            
            logger.info(f"Obtenidos {len(blocks)} bloques")
            return blocks
            
        except Exception as e:
            logger.error(f"Error obteniendo bloques: {e}")
            return []
    
    def get_smart_contracts(self, limit: int = 50) -> List[Dict]:
        """
        Obtener información de contratos inteligentes
        
        Args:
            limit: Número máximo de contratos a obtener
            
        Returns:
            Lista de diccionarios con datos de contratos
        """
        try:
            contracts = []
            
            # Generar datos de ejemplo para simular contratos de Corda
            for i in range(limit):
                contract = {
                    'address': f"corda_contract_{i:03d}",
                    'name': f"CordaContract{i}",
                    'symbol': f"CC{i}",
                    'decimals': 18,
                    'total_supply': 1000000 + (i * 10000),
                    'contract_type': 'token' if i % 2 == 0 else 'utility',
                    'deployment_tx': f"corda_tx_{i*100:06d}",
                    'deployment_block': i * 100
                }
                contracts.append(contract)
            
            logger.info(f"Obtenidos {len(contracts)} contratos inteligentes")
            return contracts
            
        except Exception as e:
            logger.error(f"Error obteniendo contratos: {e}")
            return []
    
    def save_raw_data(self, data: List[Dict], data_type: str) -> str:
        """
        Guardar datos en bruto en archivos JSON
        
        Args:
            data: Lista de diccionarios con los datos
            data_type: Tipo de datos (transactions, blocks, contracts)
            
        Returns:
            Ruta del archivo guardado
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{data_type}_{timestamp}.json"
            filepath = os.path.join(RAW_DATA_DIR, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str, ensure_ascii=False)
            
            logger.info(f"Datos guardados en: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error guardando datos: {e}")
            raise
    
    def extract_all_data(self, transaction_limit: int = 1000, block_limit: int = 100, contract_limit: int = 50):
        """
        Extraer todos los tipos de datos de la red Corda
        
        Args:
            transaction_limit: Límite de transacciones a extraer
            block_limit: Límite de bloques a extraer
            contract_limit: Límite de contratos a extraer
        """
        logger.info("Iniciando extracción de datos de Corda...")
        
        # Extraer transacciones
        logger.info("Extrayendo transacciones...")
        transactions = self.get_transactions(limit=transaction_limit)
        if transactions:
            self.save_raw_data(transactions, 'transactions')
        
        # Extraer bloques
        logger.info("Extrayendo bloques...")
        blocks = self.get_blocks(limit=block_limit)
        if blocks:
            self.save_raw_data(blocks, 'blocks')
        
        # Extraer contratos inteligentes
        logger.info("Extrayendo contratos inteligentes...")
        contracts = self.get_smart_contracts(limit=contract_limit)
        if contracts:
            self.save_raw_data(contracts, 'contracts')
        
        logger.info("Extracción de datos completada")

def main():
    """Función principal para ejecutar la extracción de datos"""
    extractor = CordaDataExtractor()
    extractor.extract_all_data(
        transaction_limit=100000,
        block_limit=1000,
        contract_limit=500
    )

if __name__ == "__main__":
    main()
