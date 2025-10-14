"""
Procesador de datos de blockchain usando Pandas
Transforma los datos extraídos en formato estructurado
"""
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, CSV_DIR

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BlockchainDataProcessor:
    """Clase para procesar y transformar datos de blockchain"""
    
    def __init__(self):
        self.processed_data = {}
        
    def load_raw_data(self, data_type: str) -> List[Dict]:
        """
        Cargar datos en bruto desde archivos JSON
        
        Args:
            data_type: Tipo de datos (transactions, blocks, contracts)
            
        Returns:
            Lista de diccionarios con los datos
        """
        try:
            # Buscar el archivo más reciente del tipo especificado
            files = [f for f in os.listdir(RAW_DATA_DIR) if f.startswith(data_type) and f.endswith('.json')]
            if not files:
                logger.warning(f"No se encontraron archivos para {data_type}")
                return []
            
            latest_file = sorted(files)[-1]
            filepath = os.path.join(RAW_DATA_DIR, latest_file)
            
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"Cargados {len(data)} registros de {data_type}")
            return data
            
        except Exception as e:
            logger.error(f"Error cargando datos de {data_type}: {e}")
            return []
    
    def process_transactions(self) -> pd.DataFrame:
        """
        Procesar y transformar datos de transacciones
        
        Returns:
            DataFrame con transacciones procesadas
        """
        logger.info("Procesando transacciones...")
        
        raw_data = self.load_raw_data('transactions')
        if not raw_data:
            return pd.DataFrame()
        
        # Crear DataFrame
        df = pd.DataFrame(raw_data)
        
        # Convertir tipos de datos
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df['gas_used'] = pd.to_numeric(df['gas_used'], errors='coerce')
        df['gas_price'] = pd.to_numeric(df['gas_price'], errors='coerce')
        df['block_number'] = pd.to_numeric(df['block_number'], errors='coerce')
        df['transaction_index'] = pd.to_numeric(df['transaction_index'], errors='coerce')
        
        # Calcular campos derivados
        df['gas_cost'] = df['gas_used'] * df['gas_price']
        df['date'] = df['timestamp'].dt.date
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.day_name()
        
        # Limpiar datos
        df = df.dropna(subset=['value', 'gas_used', 'gas_price'])
        
        # Agregar categorías de transacción
        df['transaction_type'] = df.apply(self._categorize_transaction, axis=1)
        df['value_category'] = pd.cut(df['value'], 
                                    bins=[0, 100, 1000, 10000, float('inf')], 
                                    labels=['Low', 'Medium', 'High', 'Very High'])
        
        # Ordenar por timestamp
        df = df.sort_values('timestamp')
        
        logger.info(f"Transacciones procesadas: {len(df)} registros")
        self.processed_data['transactions'] = df
        return df
    
    def process_blocks(self) -> pd.DataFrame:
        """
        Procesar y transformar datos de bloques
        
        Returns:
            DataFrame con bloques procesados
        """
        logger.info("Procesando bloques...")
        
        raw_data = self.load_raw_data('blocks')
        if not raw_data:
            return pd.DataFrame()
        
        # Crear DataFrame
        df = pd.DataFrame(raw_data)
        
        # Convertir tipos de datos
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['block_number'] = pd.to_numeric(df['block_number'], errors='coerce')
        df['gas_limit'] = pd.to_numeric(df['gas_limit'], errors='coerce')
        df['gas_used'] = pd.to_numeric(df['gas_used'], errors='coerce')
        df['transaction_count'] = pd.to_numeric(df['transaction_count'], errors='coerce')
        df['size_bytes'] = pd.to_numeric(df['size_bytes'], errors='coerce')
        
        # Calcular campos derivados
        df['gas_utilization'] = (df['gas_used'] / df['gas_limit']) * 100
        df['avg_tx_size'] = df['size_bytes'] / df['transaction_count']
        df['date'] = df['timestamp'].dt.date
        df['hour'] = df['timestamp'].dt.hour
        
        # Calcular tiempo entre bloques
        df = df.sort_values('block_number')
        df['block_time'] = df['timestamp'].diff().dt.total_seconds()
        
        # Limpiar datos
        df = df.dropna(subset=['gas_used', 'gas_limit', 'transaction_count'])
        
        logger.info(f"Bloques procesados: {len(df)} registros")
        self.processed_data['blocks'] = df
        return df
    
    def process_contracts(self) -> pd.DataFrame:
        """
        Procesar y transformar datos de contratos inteligentes
        
        Returns:
            DataFrame con contratos procesados
        """
        logger.info("Procesando contratos inteligentes...")
        
        raw_data = self.load_raw_data('contracts')
        if not raw_data:
            return pd.DataFrame()
        
        # Crear DataFrame
        df = pd.DataFrame(raw_data)
        
        # Convertir tipos de datos
        df['decimals'] = pd.to_numeric(df['decimals'], errors='coerce')
        df['total_supply'] = pd.to_numeric(df['total_supply'], errors='coerce')
        df['deployment_block'] = pd.to_numeric(df['deployment_block'], errors='coerce')
        
        # Limpiar datos
        df = df.dropna(subset=['total_supply', 'decimals'])
        
        # Agregar categorías
        df['supply_category'] = pd.cut(df['total_supply'], 
                                     bins=[0, 100000, 1000000, 10000000, float('inf')], 
                                     labels=['Small', 'Medium', 'Large', 'Very Large'])
        
        logger.info(f"Contratos procesados: {len(df)} registros")
        self.processed_data['contracts'] = df
        return df
    
    def _categorize_transaction(self, row) -> str:
        """
        Categorizar transacción basada en sus características
        
        Args:
            row: Fila del DataFrame
            
        Returns:
            Categoría de la transacción
        """
        if pd.isna(row['contract_address']):
            return 'Transfer'
        elif row['value'] == 0:
            return 'Contract Call'
        else:
            return 'Token Transfer'
    
    def create_addresses_dataframe(self) -> pd.DataFrame:
        """
        Crear DataFrame de direcciones únicas basado en transacciones
        
        Returns:
            DataFrame con información de direcciones
        """
        if 'transactions' not in self.processed_data:
            logger.warning("No hay datos de transacciones procesados")
            return pd.DataFrame()
        
        df = self.processed_data['transactions']
        
        # Recopilar todas las direcciones únicas
        from_addresses = df[['from_address', 'value']].rename(columns={'from_address': 'address', 'value': 'sent_value'})
        to_addresses = df[['to_address', 'value']].rename(columns={'to_address': 'address', 'value': 'received_value'})
        
        # Agregar información de direcciones
        addresses_info = []
        
        for address in set(df['from_address'].tolist() + df['to_address'].tolist()):
            sent_txs = df[df['from_address'] == address]
            received_txs = df[df['to_address'] == address]
            
            address_info = {
                'address': address,
                'address_type': 'contract' if address in df['contract_address'].dropna().values else 'wallet',
                'first_seen_block': min(sent_txs['block_number'].min(), received_txs['block_number'].min()) if not sent_txs.empty or not received_txs.empty else None,
                'transaction_count': len(sent_txs) + len(received_txs),
                'total_sent': sent_txs['value'].sum() if not sent_txs.empty else 0,
                'total_received': received_txs['value'].sum() if not received_txs.empty else 0,
                'balance': (received_txs['value'].sum() if not received_txs.empty else 0) - (sent_txs['value'].sum() if not sent_txs.empty else 0)
            }
            addresses_info.append(address_info)
        
        addresses_df = pd.DataFrame(addresses_info)
        addresses_df['balance_category'] = pd.cut(addresses_df['balance'], 
                                                bins=[-float('inf'), 0, 1000, 10000, float('inf')], 
                                                labels=['Negative', 'Low', 'Medium', 'High'])
        
        logger.info(f"Direcciones procesadas: {len(addresses_df)} registros")
        self.processed_data['addresses'] = addresses_df
        return addresses_df
    
    def create_daily_metrics(self) -> pd.DataFrame:
        """
        Crear métricas diarias agregadas
        
        Returns:
            DataFrame con métricas diarias
        """
        if 'transactions' not in self.processed_data:
            logger.warning("No hay datos de transacciones procesados")
            return pd.DataFrame()
        
        df = self.processed_data['transactions']
        
        # Agregar por fecha
        daily_metrics = df.groupby('date').agg({
            'id': 'count',  # Total de transacciones
            'value': 'sum',  # Volumen total
            'gas_price': 'mean',  # Precio promedio de gas
            'gas_used': 'sum',  # Gas total usado
            'from_address': 'nunique',  # Direcciones únicas
            'contract_address': lambda x: x.notna().sum()  # Nuevos contratos
        }).rename(columns={
            'id': 'total_transactions',
            'value': 'total_volume',
            'gas_price': 'avg_gas_price',
            'gas_used': 'total_gas_used',
            'from_address': 'unique_addresses',
            'contract_address': 'new_contracts'
        })
        
        daily_metrics = daily_metrics.reset_index()
        daily_metrics['date'] = pd.to_datetime(daily_metrics['date'])
        
        logger.info(f"Métricas diarias creadas: {len(daily_metrics)} registros")
        self.processed_data['daily_metrics'] = daily_metrics
        return daily_metrics
    
    def save_processed_data(self):
        """Guardar todos los datos procesados en archivos CSV"""
        logger.info("Guardando datos procesados...")
        
        for data_type, df in self.processed_data.items():
            if not df.empty:
                filename = f"{data_type}_processed.csv"
                filepath = os.path.join(CSV_DIR, filename)
                df.to_csv(filepath, index=False, encoding='utf-8')
                logger.info(f"Datos guardados: {filepath}")
    
    def process_all_data(self):
        """Procesar todos los tipos de datos"""
        logger.info("Iniciando procesamiento de todos los datos...")
        
        # Procesar cada tipo de datos
        self.process_transactions()
        self.process_blocks()
        self.process_contracts()
        
        # Crear datos derivados
        self.create_addresses_dataframe()
        self.create_daily_metrics()
        
        # Guardar datos procesados
        self.save_processed_data()
        
        logger.info("Procesamiento de datos completado")
        
        # Mostrar resumen
        self.show_summary()
    
    def show_summary(self):
        """Mostrar resumen de los datos procesados"""
        logger.info("=== RESUMEN DE DATOS PROCESADOS ===")
        
        for data_type, df in self.processed_data.items():
            if not df.empty:
                logger.info(f"{data_type.upper()}: {len(df)} registros")
                if data_type == 'transactions':
                    logger.info(f"  - Rango de fechas: {df['timestamp'].min()} a {df['timestamp'].max()}")
                    logger.info(f"  - Volumen total: {df['value'].sum():.2f}")
                    logger.info(f"  - Gas total usado: {df['gas_used'].sum():,}")
                elif data_type == 'blocks':
                    logger.info(f"  - Bloques procesados: {df['block_number'].min()} a {df['block_number'].max()}")
                    logger.info(f"  - Utilización promedio de gas: {df['gas_utilization'].mean():.2f}%")
                elif data_type == 'contracts':
                    logger.info(f"  - Tipos de contratos: {df['contract_type'].value_counts().to_dict()}")

def main():
    """Función principal para ejecutar el procesamiento de datos"""
    processor = BlockchainDataProcessor()
    processor.process_all_data()

if __name__ == "__main__":
    main()
