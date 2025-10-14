"""
Crear dashboards interactivos con Python usando los datos de blockchain
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from config import CSV_DIR

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BlockchainDashboard:
    """Clase para crear dashboards de blockchain con Python"""
    
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
    
    def create_kpi_dashboard(self):
        """Crear dashboard con KPIs principales"""
        logger.info("Creando dashboard de KPIs...")
        
        # Calcular KPIs
        total_transactions = len(self.data['transactions'])
        total_volume = self.data['transactions']['value'].sum()
        avg_gas_price = self.data['transactions']['gas_price'].mean()
        unique_addresses = self.data['transactions']['from_address'].nunique()
        total_days = len(self.data['daily_metrics'])
        
        # Crear subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Transacciones Totales', 'Volumen Total', 'Gas Promedio', 
                          'Direcciones Únicas', 'Días de Datos', 'Utilización de Gas'),
            specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]]
        )
        
        # Agregar indicadores
        fig.add_trace(go.Indicator(
            mode="number",
            value=total_transactions,
            title={"text": "Transacciones"},
            number={'font': {'size': 40}}
        ), row=1, col=1)
        
        fig.add_trace(go.Indicator(
            mode="number",
            value=total_volume,
            title={"text": "Volumen Total"},
            number={'font': {'size': 40}, 'valueformat': '$,.0f'}
        ), row=1, col=2)
        
        fig.add_trace(go.Indicator(
            mode="number",
            value=avg_gas_price,
            title={"text": "Gas Promedio"},
            number={'font': {'size': 40}, 'valueformat': '.8f'}
        ), row=1, col=3)
        
        fig.add_trace(go.Indicator(
            mode="number",
            value=unique_addresses,
            title={"text": "Direcciones Únicas"},
            number={'font': {'size': 40}}
        ), row=2, col=1)
        
        fig.add_trace(go.Indicator(
            mode="number",
            value=total_days,
            title={"text": "Días de Datos"},
            number={'font': {'size': 40}}
        ), row=2, col=2)
        
        avg_gas_utilization = self.data['blocks']['gas_utilization'].mean()
        fig.add_trace(go.Indicator(
            mode="number+delta",
            value=avg_gas_utilization,
            title={"text": "Utilización Gas %"},
            number={'font': {'size': 40}, 'valueformat': '.1f'},
            delta={'reference': 75}
        ), row=2, col=3)
        
        fig.update_layout(
            title="Dashboard de KPIs - Blockchain Analytics",
            height=600,
            showlegend=False
        )
        
        return fig
    
    def create_temporal_analysis(self):
        """Crear análisis temporal de transacciones"""
        logger.info("Creando análisis temporal...")
        
        # Preparar datos
        daily_data = self.data['daily_metrics'].copy()
        daily_data = daily_data.sort_values('date')
        
        # Crear subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Transacciones por Día', 'Volumen por Día', 
                          'Gas Promedio por Día', 'Direcciones Activas por Día'),
            vertical_spacing=0.1
        )
        
        # Transacciones por día
        fig.add_trace(go.Scatter(
            x=daily_data['date'],
            y=daily_data['total_transactions'],
            mode='lines',
            name='Transacciones',
            line=dict(color='blue', width=2)
        ), row=1, col=1)
        
        # Volumen por día
        fig.add_trace(go.Scatter(
            x=daily_data['date'],
            y=daily_data['total_volume'],
            mode='lines',
            name='Volumen',
            line=dict(color='green', width=2)
        ), row=1, col=2)
        
        # Gas promedio por día
        fig.add_trace(go.Scatter(
            x=daily_data['date'],
            y=daily_data['avg_gas_price'],
            mode='lines',
            name='Gas Promedio',
            line=dict(color='red', width=2)
        ), row=2, col=1)
        
        # Direcciones activas por día
        fig.add_trace(go.Scatter(
            x=daily_data['date'],
            y=daily_data['unique_addresses'],
            mode='lines',
            name='Direcciones Activas',
            line=dict(color='purple', width=2)
        ), row=2, col=2)
        
        fig.update_layout(
            title="Análisis Temporal - Blockchain Analytics",
            height=800,
            showlegend=False
        )
        
        return fig
    
    def create_contract_analysis(self):
        """Crear análisis de contratos inteligentes"""
        logger.info("Creando análisis de contratos...")
        
        # Preparar datos
        contracts = self.data['contracts'].copy()
        
        # Crear subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Distribución por Tipo', 'Suministro Total por Tipo'),
            specs=[[{"type": "pie"}, {"type": "bar"}]]
        )
        
        # Gráfico de pastel por tipo
        type_counts = contracts['contract_type'].value_counts()
        fig.add_trace(go.Pie(
            labels=type_counts.index,
            values=type_counts.values,
            name="Tipos de Contratos"
        ), row=1, col=1)
        
        # Gráfico de barras por suministro
        supply_by_type = contracts.groupby('contract_type')['total_supply'].sum()
        fig.add_trace(go.Bar(
            x=supply_by_type.index,
            y=supply_by_type.values,
            name="Suministro Total"
        ), row=1, col=2)
        
        fig.update_layout(
            title="Análisis de Contratos Inteligentes",
            height=500
        )
        
        return fig
    
    def create_address_analysis(self):
        """Crear análisis de direcciones"""
        logger.info("Creando análisis de direcciones...")
        
        # Preparar datos
        addresses = self.data['addresses'].copy()
        
        # Crear gráfico de dispersión
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=addresses['transaction_count'],
            y=addresses['balance'],
            mode='markers',
            marker=dict(
                size=addresses['total_sent'] / 1000,
                color=addresses['total_received'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Total Recibido")
            ),
            text=addresses['address'],
            hovertemplate='<b>%{text}</b><br>' +
                         'Transacciones: %{x}<br>' +
                         'Balance: %{y}<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title="Análisis de Direcciones - Balance vs Transacciones",
            xaxis_title="Número de Transacciones",
            yaxis_title="Balance",
            height=600
        )
        
        return fig
    
    def create_gas_analysis(self):
        """Crear análisis de gas"""
        logger.info("Creando análisis de gas...")
        
        # Preparar datos
        transactions = self.data['transactions'].copy()
        
        # Crear subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Distribución de Gas Price', 'Gas Used vs Value',
                          'Gas Cost por Transacción', 'Utilización de Gas por Bloque'),
            vertical_spacing=0.1
        )
        
        # Histograma de gas price
        fig.add_trace(go.Histogram(
            x=transactions['gas_price'],
            nbinsx=50,
            name="Gas Price"
        ), row=1, col=1)
        
        # Scatter plot gas used vs value
        fig.add_trace(go.Scatter(
            x=transactions['gas_used'],
            y=transactions['value'],
            mode='markers',
            marker=dict(size=4, opacity=0.6),
            name="Gas vs Value"
        ), row=1, col=2)
        
        # Histograma de gas cost
        fig.add_trace(go.Histogram(
            x=transactions['gas_cost'],
            nbinsx=50,
            name="Gas Cost"
        ), row=2, col=1)
        
        # Utilización de gas por bloque
        blocks = self.data['blocks'].copy()
        fig.add_trace(go.Scatter(
            x=blocks['block_number'],
            y=blocks['gas_utilization'],
            mode='lines',
            name="Utilización Gas"
        ), row=2, col=2)
        
        fig.update_layout(
            title="Análisis de Gas - Blockchain Analytics",
            height=800,
            showlegend=False
        )
        
        return fig
    
    def save_dashboards(self):
        """Guardar todos los dashboards como archivos HTML"""
        logger.info("Guardando dashboards...")
        
        dashboards = {
            'kpi_dashboard': self.create_kpi_dashboard(),
            'temporal_analysis': self.create_temporal_analysis(),
            'contract_analysis': self.create_contract_analysis(),
            'address_analysis': self.create_address_analysis(),
            'gas_analysis': self.create_gas_analysis()
        }
        
        for name, fig in dashboards.items():
            filename = f"dashboard_{name}.html"
            fig.write_html(filename)
            logger.info(f"Dashboard guardado: {filename}")
        
        logger.info("Todos los dashboards han sido guardados como archivos HTML")
        logger.info("Puedes abrirlos en tu navegador web para interactuar con ellos")

def main():
    """Función principal"""
    try:
        dashboard = BlockchainDashboard()
        dashboard.save_dashboards()
        
        print("\n" + "="*60)
        print("🎉 DASHBOARDS CREADOS EXITOSAMENTE")
        print("="*60)
        print("📊 Archivos generados:")
        print("  - dashboard_kpi_dashboard.html")
        print("  - dashboard_temporal_analysis.html")
        print("  - dashboard_contract_analysis.html")
        print("  - dashboard_address_analysis.html")
        print("  - dashboard_gas_analysis.html")
        print("\n🌐 Abre estos archivos en tu navegador para ver los dashboards interactivos")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error creando dashboards: {e}")

if __name__ == "__main__":
    main()
