"""
Crear dashboards con Matplotlib para visualización directa
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from config import CSV_DIR

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurar estilo
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class MatplotlibDashboard:
    """Clase para crear dashboards con Matplotlib"""
    
    def __init__(self):
        self.data = {}
        self.load_data()
    
    def load_data(self):
        """Cargar todos los datos CSV"""
        try:
            # Cargar datos principales
            self.data['transactions'] = pd.read_csv(f"{CSV_DIR}/transactions_processed.csv")
            self.data['daily_metrics'] = pd.read_csv(f"{CSV_DIR}/daily_metrics_processed.csv")
            self.data['blocks'] = pd.read_csv(f"{CSV_DIR}/blocks_processed.csv")
            self.data['contracts'] = pd.read_csv(f"{CSV_DIR}/contracts_processed.csv")
            self.data['addresses'] = pd.read_csv(f"{CSV_DIR}/addresses_processed.csv")
            
            # Convertir fechas
            self.data['transactions']['timestamp'] = pd.to_datetime(self.data['transactions']['timestamp'])
            self.data['daily_metrics']['date'] = pd.to_datetime(self.data['daily_metrics']['date'])
            self.data['blocks']['timestamp'] = pd.to_datetime(self.data['blocks']['timestamp'])
            
            logger.info("Datos cargados exitosamente")
            
        except Exception as e:
            logger.error(f"Error cargando datos: {e}")
    
    def create_overview_dashboard(self):
        """Crear dashboard de resumen general"""
        logger.info("Creando dashboard de resumen...")
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Dashboard de Resumen - Blockchain Analytics', fontsize=20, fontweight='bold')
        
        # 1. Transacciones por día
        daily_data = self.data['daily_metrics'].copy()
        daily_data = daily_data.sort_values('date')
        
        axes[0, 0].plot(daily_data['date'], daily_data['total_transactions'], linewidth=2, color='blue')
        axes[0, 0].set_title('Transacciones por Día', fontweight='bold')
        axes[0, 0].set_xlabel('Fecha')
        axes[0, 0].set_ylabel('Número de Transacciones')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Volumen por día
        axes[0, 1].plot(daily_data['date'], daily_data['total_volume'], linewidth=2, color='green')
        axes[0, 1].set_title('Volumen por Día', fontweight='bold')
        axes[0, 1].set_xlabel('Fecha')
        axes[0, 1].set_ylabel('Volumen Total')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Gas promedio por día
        axes[0, 2].plot(daily_data['date'], daily_data['avg_gas_price'], linewidth=2, color='red')
        axes[0, 2].set_title('Gas Promedio por Día', fontweight='bold')
        axes[0, 2].set_xlabel('Fecha')
        axes[0, 2].set_ylabel('Gas Price Promedio')
        axes[0, 2].tick_params(axis='x', rotation=45)
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Distribución de tipos de contratos
        contract_types = self.data['contracts']['contract_type'].value_counts()
        axes[1, 0].pie(contract_types.values, labels=contract_types.index, autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('Distribución de Tipos de Contratos', fontweight='bold')
        
        # 5. Top 10 direcciones por balance
        top_addresses = self.data['addresses'].nlargest(10, 'balance')
        axes[1, 1].barh(range(len(top_addresses)), top_addresses['balance'])
        axes[1, 1].set_yticks(range(len(top_addresses)))
        axes[1, 1].set_yticklabels([addr[:10] + '...' for addr in top_addresses['address']])
        axes[1, 1].set_title('Top 10 Direcciones por Balance', fontweight='bold')
        axes[1, 1].set_xlabel('Balance')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Utilización de gas por bloque
        blocks_data = self.data['blocks'].copy()
        axes[1, 2].scatter(blocks_data['block_number'], blocks_data['gas_utilization'], alpha=0.6, s=20)
        axes[1, 2].set_title('Utilización de Gas por Bloque', fontweight='bold')
        axes[1, 2].set_xlabel('Número de Bloque')
        axes[1, 2].set_ylabel('Utilización de Gas (%)')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('dashboard_overview.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("Dashboard de resumen guardado como dashboard_overview.png")
    
    def create_transaction_analysis(self):
        """Crear análisis detallado de transacciones"""
        logger.info("Creando análisis de transacciones...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Análisis Detallado de Transacciones', fontsize=18, fontweight='bold')
        
        transactions = self.data['transactions'].copy()
        
        # 1. Distribución de valores de transacciones
        axes[0, 0].hist(transactions['value'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Distribución de Valores de Transacciones', fontweight='bold')
        axes[0, 0].set_xlabel('Valor de Transacción')
        axes[0, 0].set_ylabel('Frecuencia')
        axes[0, 0].set_yscale('log')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Gas price vs Gas used
        sample_data = transactions.sample(n=min(5000, len(transactions)))
        scatter = axes[0, 1].scatter(sample_data['gas_used'], sample_data['gas_price'], 
                                   c=sample_data['value'], cmap='viridis', alpha=0.6, s=10)
        axes[0, 1].set_title('Gas Used vs Gas Price', fontweight='bold')
        axes[0, 1].set_xlabel('Gas Used')
        axes[0, 1].set_ylabel('Gas Price')
        axes[0, 1].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[0, 1], label='Valor de Transacción')
        
        # 3. Transacciones por hora del día
        transactions['hour'] = transactions['timestamp'].dt.hour
        hourly_counts = transactions['hour'].value_counts().sort_index()
        axes[1, 0].bar(hourly_counts.index, hourly_counts.values, color='orange', alpha=0.7)
        axes[1, 0].set_title('Transacciones por Hora del Día', fontweight='bold')
        axes[1, 0].set_xlabel('Hora del Día')
        axes[1, 0].set_ylabel('Número de Transacciones')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Costo de gas por transacción
        axes[1, 1].hist(transactions['gas_cost'], bins=50, alpha=0.7, color='red', edgecolor='black')
        axes[1, 1].set_title('Distribución del Costo de Gas', fontweight='bold')
        axes[1, 1].set_xlabel('Costo de Gas')
        axes[1, 1].set_ylabel('Frecuencia')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('dashboard_transaction_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("Análisis de transacciones guardado como dashboard_transaction_analysis.png")
    
    def create_network_metrics(self):
        """Crear métricas de red"""
        logger.info("Creando métricas de red...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Métricas de Red Blockchain', fontsize=18, fontweight='bold')
        
        daily_data = self.data['daily_metrics'].copy()
        daily_data = daily_data.sort_values('date')
        
        # 1. Direcciones activas por día
        axes[0, 0].plot(daily_data['date'], daily_data['unique_addresses'], linewidth=2, color='purple')
        axes[0, 0].set_title('Direcciones Activas por Día', fontweight='bold')
        axes[0, 0].set_xlabel('Fecha')
        axes[0, 0].set_ylabel('Direcciones Únicas')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Tamaño promedio de bloques
        blocks_data = self.data['blocks'].copy()
        axes[0, 1].plot(blocks_data['block_number'], blocks_data['size_bytes'], linewidth=2, color='brown')
        axes[0, 1].set_title('Tamaño de Bloques', fontweight='bold')
        axes[0, 1].set_xlabel('Número de Bloque')
        axes[0, 1].set_ylabel('Tamaño del Bloque (bytes)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Transacciones por bloque
        axes[1, 0].plot(blocks_data['block_number'], blocks_data['transaction_count'], linewidth=2, color='teal')
        axes[1, 0].set_title('Transacciones por Bloque', fontweight='bold')
        axes[1, 0].set_xlabel('Número de Bloque')
        axes[1, 0].set_ylabel('Número de Transacciones')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Gas limit vs Gas used
        axes[1, 1].scatter(blocks_data['gas_limit'], blocks_data['gas_used'], alpha=0.6, s=20, color='green')
        axes[1, 1].set_title('Gas Limit vs Gas Used', fontweight='bold')
        axes[1, 1].set_xlabel('Gas Limit')
        axes[1, 1].set_ylabel('Gas Used')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('dashboard_network_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("Métricas de red guardadas como dashboard_network_metrics.png")
    
    def create_summary_report(self):
        """Crear reporte de resumen con estadísticas"""
        logger.info("Creando reporte de resumen...")
        
        # Calcular estadísticas
        total_transactions = len(self.data['transactions'])
        total_volume = self.data['transactions']['value'].sum()
        avg_gas_price = self.data['transactions']['gas_price'].mean()
        unique_addresses = self.data['transactions']['from_address'].nunique()
        total_blocks = len(self.data['blocks'])
        total_contracts = len(self.data['contracts'])
        total_days = len(self.data['daily_metrics'])
        
        # Crear figura con texto
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')
        
        # Título
        ax.text(0.5, 0.95, 'REPORTE DE RESUMEN - BLOCKCHAIN ANALYTICS', 
                ha='center', va='top', fontsize=24, fontweight='bold', transform=ax.transAxes)
        
        # Estadísticas principales
        stats_text = f"""
        📊 ESTADÍSTICAS PRINCIPALES:
        
        • Total de Transacciones: {total_transactions:,}
        • Volumen Total: ${total_volume:,.2f}
        • Gas Price Promedio: {avg_gas_price:.8f}
        • Direcciones Únicas: {unique_addresses:,}
        • Total de Bloques: {total_blocks:,}
        • Contratos Inteligentes: {total_contracts:,}
        • Días de Datos: {total_days:,}
        
        📈 MÉTRICAS ADICIONALES:
        
        • Transacciones por Día (Promedio): {total_transactions/total_days:.0f}
        • Volumen por Día (Promedio): ${total_volume/total_days:,.2f}
        • Transacciones por Bloque (Promedio): {total_transactions/total_blocks:.0f}
        • Direcciones Activas por Día (Promedio): {unique_addresses/total_days:.0f}
        
        🎯 RANGOS TEMPORALES:
        
        • Fecha Inicio: {self.data['daily_metrics']['date'].min().strftime('%Y-%m-%d')}
        • Fecha Fin: {self.data['daily_metrics']['date'].max().strftime('%Y-%m-%d')}
        • Período Total: {(self.data['daily_metrics']['date'].max() - self.data['daily_metrics']['date'].min()).days} días
        """
        
        ax.text(0.05, 0.85, stats_text, ha='left', va='top', fontsize=14, 
                transform=ax.transAxes, fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('dashboard_summary_report.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("Reporte de resumen guardado como dashboard_summary_report.png")
    
    def create_all_dashboards(self):
        """Crear todos los dashboards"""
        logger.info("Creando todos los dashboards con Matplotlib...")
        
        self.create_overview_dashboard()
        self.create_transaction_analysis()
        self.create_network_metrics()
        self.create_summary_report()
        
        logger.info("Todos los dashboards han sido creados exitosamente")

def main():
    """Función principal"""
    try:
        dashboard = MatplotlibDashboard()
        dashboard.create_all_dashboards()
        
        print("\n" + "="*60)
        print("🎉 DASHBOARDS CON MATPLOTLIB CREADOS EXITOSAMENTE")
        print("="*60)
        print("📊 Archivos generados:")
        print("  - dashboard_overview.png")
        print("  - dashboard_transaction_analysis.png")
        print("  - dashboard_network_metrics.png")
        print("  - dashboard_summary_report.png")
        print("\n🖼️ Estos archivos PNG se pueden abrir directamente en tu sistema")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error creando dashboards: {e}")

if __name__ == "__main__":
    main()
