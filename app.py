"""
Aplicación Web Interactiva de Blockchain Analytics con IA
Usando Plotly Dash para visualización y scikit-learn para predicciones
"""

import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
import mysql.connector
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings('ignore')

# Importar modelos de IA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Configuración de la base de datos
DB_CONFIG = {
    'host': 'localhost',
    'user': 'blockchainuser',
    'password': '1234',
    'database': 'blockchain_analytics',
    'port': 3306,
    'charset': 'utf8mb4'
}

class BlockchainDataProcessor:
    """Clase para procesar y cargar datos de blockchain"""
    
    def __init__(self):
        self.engine = None
        self.connect_to_db()
    
    def connect_to_db(self):
        """Conectar a la base de datos MySQL"""
        try:
            connection_string = f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
            self.engine = create_engine(connection_string)
            print("Conexion a MySQL establecida")
        except Exception as e:
            print(f"Error conectando a MySQL: {e}")
            self.engine = None
    
    def load_daily_metrics(self):
        """Cargar métricas diarias desde MySQL"""
        try:
            if self.engine:
                query = """
                SELECT date, total_transactions, total_volume, avg_gas_price, 
                       total_gas_used, unique_addresses, new_contracts
                FROM daily_metrics 
                ORDER BY date
                """
                df = pd.read_sql(query, self.engine)
                df['date'] = pd.to_datetime(df['date'])
                return df
            else:
                # Fallback a CSV si no hay conexión a MySQL
                return pd.read_csv('data/csv/daily_metrics_processed.csv')
        except Exception as e:
            print(f"Error cargando desde MySQL, usando CSV: {e}")
            return pd.read_csv('data/csv/daily_metrics_processed.csv')
    
    def load_transactions(self, limit=10000):
        """Cargar transacciones desde MySQL"""
        try:
            if self.engine:
                query = f"""
                SELECT date, value, gas_price, gas_used, transaction_type, 
                       from_address, to_address
                FROM transactions 
                ORDER BY date DESC 
                LIMIT {limit}
                """
                df = pd.read_sql(query, self.engine)
                df['date'] = pd.to_datetime(df['date'])
                return df
            else:
                return pd.read_csv('data/csv/transactions_processed.csv', nrows=limit)
        except Exception as e:
            print(f"Error cargando transacciones: {e}")
            return pd.read_csv('data/csv/transactions_processed.csv', nrows=limit)

class AIModel:
    """Clase para modelos de IA y predicciones"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
    
    def prepare_features(self, df, target_column, lookback_days=7):
        """Preparar características para el modelo"""
        df = df.copy()
        df = df.sort_values('date').reset_index(drop=True)
        
        # Crear características temporales
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        
        # Crear características de lag (valores pasados)
        for i in range(1, lookback_days + 1):
            df[f'{target_column}_lag_{i}'] = df[target_column].shift(i)
        
        # Crear características de media móvil
        for window in [3, 7, 14]:
            df[f'{target_column}_ma_{window}'] = df[target_column].rolling(window=window).mean()
        
        # Crear características de tendencia
        df[f'{target_column}_trend'] = df[target_column].diff()
        
        # Eliminar filas con NaN
        df = df.dropna()
        
        # Seleccionar características
        feature_columns = [col for col in df.columns if col not in ['date', target_column]]
        self.feature_columns = feature_columns
        
        return df, feature_columns
    
    def train_model(self, df, target_column, model_type='linear', lookback_days=7):
        """Entrenar modelo de IA"""
        df_processed, feature_columns = self.prepare_features(df, target_column, lookback_days)
        
        X = df_processed[feature_columns]
        y = df_processed[target_column]
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Escalar características
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Seleccionar modelo
        if model_type == 'linear':
            model = LinearRegression()
        elif model_type == 'ridge':
            model = Ridge(alpha=1.0)
        elif model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            model = LinearRegression()
        
        # Entrenar modelo
        model.fit(X_train_scaled, y_train)
        
        # Evaluar modelo
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Guardar modelo y scaler
        self.models[target_column] = model
        self.scalers[target_column] = scaler
        
        return {
            'model': model,
            'scaler': scaler,
            'mse': mse,
            'r2': r2,
            'feature_columns': feature_columns
        }
    
    def predict_future(self, df, target_column, days_ahead=30):
        """Predecir valores futuros"""
        if target_column not in self.models:
            return None
        
        model = self.models[target_column]
        scaler = self.scalers[target_column]
        
        # Obtener los últimos datos
        last_data = df.tail(30).copy()  # Usar últimos 30 días
        
        predictions = []
        current_data = last_data.copy()
        
        for i in range(days_ahead):
            # Preparar características para predicción
            current_data, _ = self.prepare_features(current_data, target_column)
            
            if len(current_data) == 0:
                break
                
            # Obtener la última fila
            last_row = current_data.iloc[-1:][self.feature_columns]
            last_row_scaled = scaler.transform(last_row)
            
            # Hacer predicción
            pred = model.predict(last_row_scaled)[0]
            predictions.append(pred)
            
            # Agregar predicción a los datos para la siguiente iteración
            next_date = current_data['date'].iloc[-1] + timedelta(days=1)
            new_row = current_data.iloc[-1:].copy()
            new_row['date'] = next_date
            new_row[target_column] = pred
            current_data = pd.concat([current_data, new_row], ignore_index=True)
        
        return predictions

# Inicializar procesador de datos y modelo de IA
data_processor = BlockchainDataProcessor()
ai_model = AIModel()

# Cargar datos
print("Cargando datos...")
daily_metrics = data_processor.load_daily_metrics()
transactions = data_processor.load_transactions(limit=5000)

print(f"Datos cargados: {len(daily_metrics)} registros diarios, {len(transactions)} transacciones")

# Inicializar aplicación Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Blockchain Analytics Dashboard con IA"

# Definir layout de la aplicación
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("🚀 Blockchain Analytics Dashboard", className="text-center mb-4"),
            html.P("Análisis de datos blockchain con predicciones de IA", className="text-center text-muted")
        ])
    ]),
    
    # Controles
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🎛️ Controles de Análisis"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Seleccionar Métrica:"),
                            dcc.Dropdown(
                                id='metric-dropdown',
                                options=[
                                    {'label': 'Transacciones Totales', 'value': 'total_transactions'},
                                    {'label': 'Volumen Total', 'value': 'total_volume'},
                                    {'label': 'Precio Promedio de Gas', 'value': 'avg_gas_price'},
                                    {'label': 'Gas Total Usado', 'value': 'total_gas_used'},
                                    {'label': 'Direcciones Únicas', 'value': 'unique_addresses'},
                                    {'label': 'Nuevos Contratos', 'value': 'new_contracts'}
                                ],
                                value='total_transactions',
                                clearable=False
                            )
                        ], width=6),
                        dbc.Col([
                            html.Label("Modelo de IA:"),
                            dcc.Dropdown(
                                id='model-dropdown',
                                options=[
                                    {'label': 'Regresión Lineal', 'value': 'linear'},
                                    {'label': 'Ridge Regression', 'value': 'ridge'},
                                    {'label': 'Random Forest', 'value': 'random_forest'}
                                ],
                                value='linear',
                                clearable=False
                            )
                        ], width=6)
                    ], className="mb-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Label("Días de Predicción:"),
                            dcc.Slider(
                                id='prediction-days',
                                min=7,
                                max=90,
                                step=7,
                                value=30,
                                marks={i: f'{i}d' for i in range(7, 91, 14)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], width=6),
                        dbc.Col([
                            html.Label("Días de Lookback:"),
                            dcc.Slider(
                                id='lookback-days',
                                min=3,
                                max=30,
                                step=1,
                                value=7,
                                marks={i: f'{i}d' for i in range(3, 31, 5)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], width=6)
                    ]),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Button("🔄 Entrenar Modelo", id="train-button", color="primary", className="me-2"),
                            dbc.Button("📊 Actualizar Gráficos", id="update-button", color="success")
                        ], className="mt-3")
                    ])
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    # Gráficos principales
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📈 Análisis Temporal con Predicciones de IA"),
                dbc.CardBody([
                    dcc.Graph(id='main-chart')
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    # Gráficos secundarios
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🔍 Análisis de Dispersión"),
                dbc.CardBody([
                    dcc.Graph(id='scatter-chart')
                ])
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📊 Métricas del Modelo"),
                dbc.CardBody([
                    html.Div(id='model-metrics')
                ])
            ])
        ], width=6)
    ], className="mb-4"),
    
    # Información del dataset
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📋 Información del Dataset"),
                dbc.CardBody([
                    html.Div(id='dataset-info')
                ])
            ])
        ], width=12)
    ])
], fluid=True)

# Callbacks para interactividad
@app.callback(
    [Output('main-chart', 'figure'),
     Output('scatter-chart', 'figure'),
     Output('model-metrics', 'children'),
     Output('dataset-info', 'children')],
    [Input('train-button', 'n_clicks'),
     Input('update-button', 'n_clicks')],
    [State('metric-dropdown', 'value'),
     State('model-dropdown', 'value'),
     State('prediction-days', 'value'),
     State('lookback-days', 'value')]
)
def update_charts(train_clicks, update_clicks, selected_metric, model_type, prediction_days, lookback_days):
    """Actualizar gráficos basado en los controles del usuario"""
    
    # Entrenar modelo si se hizo clic en entrenar
    ctx = dash.callback_context
    if ctx.triggered and 'train-button' in ctx.triggered[0]['prop_id']:
        print(f"🤖 Entrenando modelo {model_type} para {selected_metric}...")
        model_info = ai_model.train_model(daily_metrics, selected_metric, model_type, lookback_days)
        print(f"✅ Modelo entrenado - R²: {model_info['r2']:.3f}, MSE: {model_info['mse']:.3f}")
    
    # Crear gráfico principal
    fig_main = create_main_chart(daily_metrics, selected_metric, prediction_days, lookback_days)
    
    # Crear gráfico de dispersión
    fig_scatter = create_scatter_chart(daily_metrics, selected_metric)
    
    # Métricas del modelo
    model_metrics = create_model_metrics(selected_metric)
    
    # Información del dataset
    dataset_info = create_dataset_info()
    
    return fig_main, fig_scatter, model_metrics, dataset_info

def create_main_chart(df, metric, prediction_days, lookback_days):
    """Crear gráfico principal con datos históricos y predicciones"""
    
    # Filtrar datos válidos
    df_clean = df.dropna(subset=[metric])
    
    # Crear gráfico
    fig = go.Figure()
    
    # Agregar datos históricos
    fig.add_trace(go.Scatter(
        x=df_clean['date'],
        y=df_clean[metric],
        mode='lines+markers',
        name='Datos Históricos',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=4)
    ))
    
    # Generar predicciones si el modelo está entrenado
    if metric in ai_model.models:
        try:
            predictions = ai_model.predict_future(df_clean, metric, prediction_days)
            if predictions:
                # Crear fechas futuras
                last_date = df_clean['date'].iloc[-1]
                future_dates = [last_date + timedelta(days=i+1) for i in range(len(predictions))]
                
                # Agregar predicciones
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=predictions,
                    mode='lines+markers',
                    name='Predicciones IA',
                    line=dict(color='#ff7f0e', width=3, dash='dash'),
                    marker=dict(size=6, symbol='diamond')
                ))
                
                # Agregar área de confianza (simulada)
                fig.add_trace(go.Scatter(
                    x=future_dates + future_dates[::-1],
                    y=[p * 1.1 for p in predictions] + [p * 0.9 for p in predictions[::-1]],
                    fill='toself',
                    fillcolor='rgba(255, 127, 14, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Zona de Confianza',
                    showlegend=True
                ))
        except Exception as e:
            print(f"⚠️ Error generando predicciones: {e}")
    
    # Configurar layout
    fig.update_layout(
        title=f"📈 {metric.replace('_', ' ').title()} - Análisis Temporal con IA",
        xaxis_title="Fecha",
        yaxis_title=metric.replace('_', ' ').title(),
        hovermode='x unified',
        template='plotly_white',
        height=500,
        showlegend=True
    )
    
    return fig

def create_scatter_chart(df, metric):
    """Crear gráfico de dispersión para análisis de correlaciones"""
    
    # Calcular correlaciones con otras métricas
    numeric_cols = ['total_transactions', 'total_volume', 'avg_gas_price', 
                   'total_gas_used', 'unique_addresses', 'new_contracts']
    
    # Encontrar la métrica con mayor correlación
    correlations = df[numeric_cols].corr()[metric].abs().sort_values(ascending=False)
    best_corr_metric = correlations.index[1] if len(correlations) > 1 else 'total_volume'
    
    # Crear gráfico de dispersión
    fig = px.scatter(
        df.dropna(),
        x=best_corr_metric,
        y=metric,
        color='avg_gas_price',
        size='total_gas_used',
        hover_data=['date'],
        title=f"🔍 Correlación: {metric.replace('_', ' ').title()} vs {best_corr_metric.replace('_', ' ').title()}",
        labels={
            metric: metric.replace('_', ' ').title(),
            best_corr_metric: best_corr_metric.replace('_', ' ').title()
        }
    )
    
    fig.update_layout(
        template='plotly_white',
        height=400
    )
    
    return fig

def create_model_metrics(metric):
    """Crear métricas del modelo"""
    
    if metric in ai_model.models:
        # Obtener información del modelo
        model = ai_model.models[metric]
        scaler = ai_model.scalers[metric]
        
        # Calcular métricas básicas
        df_clean = daily_metrics.dropna(subset=[metric])
        
        return dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("🎯 Precisión del Modelo", className="card-title"),
                        html.P(f"Tipo: {type(model).__name__}", className="mb-1"),
                        html.P(f"Características: {len(ai_model.feature_columns)}", className="mb-1"),
                        html.P(f"Datos de entrenamiento: {len(df_clean)} registros", className="mb-1"),
                        html.P(f"Última actualización: {datetime.now().strftime('%H:%M:%S')}", className="mb-0")
                    ])
                ], color="light")
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("📊 Estadísticas", className="card-title"),
                        html.P(f"Media: {df_clean[metric].mean():.2f}", className="mb-1"),
                        html.P(f"Desv. Estándar: {df_clean[metric].std():.2f}", className="mb-1"),
                        html.P(f"Mínimo: {df_clean[metric].min():.2f}", className="mb-1"),
                        html.P(f"Máximo: {df_clean[metric].max():.2f}", className="mb-0")
                    ])
                ], color="info")
            ], width=6)
        ])
    else:
        return dbc.Alert("🤖 Haz clic en 'Entrenar Modelo' para ver las métricas", color="warning")

def create_dataset_info():
    """Crear información del dataset"""
    
    return dbc.Row([
        dbc.Col([
            html.H6("📊 Resumen del Dataset", className="mb-3"),
            html.P(f"📅 Período: {daily_metrics['date'].min().strftime('%Y-%m-%d')} a {daily_metrics['date'].max().strftime('%Y-%m-%d')}"),
            html.P(f"📈 Registros diarios: {len(daily_metrics):,}"),
            html.P(f"💳 Transacciones: {len(transactions):,}"),
            html.P(f"⛽ Gas promedio: {daily_metrics['avg_gas_price'].mean():.2e}"),
            html.P(f"💰 Volumen total: ${daily_metrics['total_volume'].sum():,.2f}")
        ], width=6),
        dbc.Col([
            html.H6("🔧 Configuración", className="mb-3"),
            html.P(f"🗄️ Base de datos: {DB_CONFIG['database']}"),
            html.P(f"🌐 Host: {DB_CONFIG['host']}:{DB_CONFIG['port']}"),
            html.P(f"👤 Usuario: {DB_CONFIG['user']}"),
            html.P(f"🤖 Modelos disponibles: Linear, Ridge, Random Forest"),
            html.P(f"📊 Métricas disponibles: {len(['total_transactions', 'total_volume', 'avg_gas_price', 'total_gas_used', 'unique_addresses', 'new_contracts'])}")
        ], width=6)
    ])

# Función para ejecutar la aplicación
def run_app():
    """Ejecutar la aplicación Dash"""
    print("Iniciando Blockchain Analytics Dashboard...")
    print("Dashboard disponible en: http://localhost:8050")
    print("Presiona Ctrl+C para detener la aplicacion")
    
    app.run(debug=True, host='0.0.0.0', port=8050)

if __name__ == '__main__':
    run_app()
