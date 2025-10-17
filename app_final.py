"""
Aplicación Web Interactiva de Blockchain Analytics con IA - Versión Final
Usando Plotly Dash para visualización y scikit-learn para predicciones
"""

import os
import sys

# Desactivar carga automática de .env
os.environ['FLASK_SKIP_DOTENV'] = '1'

import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Importar modelos de IA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Importar Gemini API
import google.generativeai as genai
import json

# Configurar Gemini API
GEMINI_API_KEY = "AIzaSyAWHUUqZdCx0HZTvHjpktaTbgDytqT3Bx8"
genai.configure(api_key=GEMINI_API_KEY)

class GeminiForecaster:
    """Clase para pronósticos usando Gemini AI"""
    
    def __init__(self):
        try:
            print("Inicializando Gemini AI...")
            # Usar el modelo más reciente y potente
            self.model = genai.GenerativeModel('gemini-2.5-flash')
            print("OK - Gemini AI inicializado correctamente con modelo gemini-2.5-flash")
        except Exception as e:
            print(f"ERROR - Error inicializando Gemini: {e}")
            try:
                print("Intentando con modelo gemini-1.5-flash...")
                self.model = genai.GenerativeModel('gemini-1.5-flash')
                print("OK - Gemini AI inicializado con modelo gemini-1.5-flash")
            except Exception as e2:
                print(f"ERROR - Error con modelo gemini-1.5-flash: {e2}")
                try:
                    print("Intentando con modelo gemini-pro...")
                    self.model = genai.GenerativeModel('gemini-pro')
                    print("OK - Gemini AI inicializado con modelo gemini-pro")
                except Exception as e3:
                    print(f"ERROR - Error con modelo gemini-pro: {e3}")
                    self.model = None
    
    def generate_forecast(self, data, metric, time_grouping, historical_trends):
        """Generar pronóstico usando Gemini AI"""
        try:
            if self.model is None:
                return "Error: Modelo de Gemini no disponible. Verifica tu API key y conexión a internet."
            
            print(f"Generando pronóstico con IA Blockchain para {metric} ({time_grouping})...")
            
            # Preparar contexto más detallado
            context = self._prepare_detailed_context(data, metric, time_grouping, historical_trends)
            
            # Prompt optimizado para evitar timeout
            prompt = f"""
            Eres un experto analista de blockchain. Analiza estos datos y genera un pronóstico detallado.
            
            DATOS:
            {context}
            
            MÉTRICA: {metric}
            AGRUPACIÓN: {time_grouping}
            
            Genera un pronóstico que incluya:
            1. Análisis de tendencias históricas
            2. Factores que influyen en la métrica
            3. Pronóstico para próximos períodos
            4. Nivel de confianza
            5. Recomendaciones estratégicas
            
            Responde en español con números específicos. Máximo 500 palabras.
            """
            
            print("Enviando prompt a IA Blockchain...")
            # Configurar timeout y retry
            import time
            max_retries = 3
            timeout_seconds = 30
            
            for attempt in range(max_retries):
                try:
                    print(f"Intento {attempt + 1}/{max_retries}...")
                    response = self.model.generate_content(prompt)
                    break
                except Exception as e:
                    if "timeout" in str(e).lower() or "504" in str(e):
                        print(f"Timeout en intento {attempt + 1}, reintentando...")
                        if attempt < max_retries - 1:
                            time.sleep(2)
                            continue
                        else:
                            raise e
                    else:
                        raise e
            
            if not response.candidates:
                print("Gemini no devolvió candidatos para el pronóstico.")
                return "❌ Error: Gemini AI no pudo generar el pronóstico. Verifica tu conexión a internet y la API key."
            
            forecast_text = response.text
            if not forecast_text:
                print("Gemini devolvió texto vacío.")
                return "❌ Error: Gemini AI devolvió una respuesta vacía. Intenta nuevamente."

            print("Pronóstico extenso de IA Blockchain recibido exitosamente.")
            return forecast_text
            
        except Exception as e:
            print(f"Error generando pronóstico con IA Blockchain: {e}")
            return f"❌ Error de IA Blockchain: {str(e)}\n\nPor favor, verifica tu API key y conexión a internet."
    
    def _prepare_detailed_context(self, data, metric, time_grouping, historical_trends):
        """Preparar contexto detallado para Gemini"""
        try:
            if len(data) == 0:
                return "No hay datos disponibles para análisis."
            
            # Estadísticas básicas
            mean_val = data[metric].mean()
            std_val = data[metric].std()
            min_val = data[metric].min()
            max_val = data[metric].max()
            median_val = data[metric].median()
            
            # Análisis de tendencias más detallado
            if len(data) > 1:
                trend = "creciente" if data[metric].iloc[-1] > data[metric].iloc[0] else "decreciente"
                growth_rate = ((data[metric].iloc[-1] - data[metric].iloc[0]) / data[metric].iloc[0]) * 100
                
                # Análisis de volatilidad
                volatility = (std_val / mean_val) * 100 if mean_val != 0 else 0
                
                # Análisis de cuartiles
                q1 = data[metric].quantile(0.25)
                q3 = data[metric].quantile(0.75)
                iqr = q3 - q1
                
                # Análisis de valores extremos
                outliers = len(data[(data[metric] < q1 - 1.5*iqr) | (data[metric] > q3 + 1.5*iqr)])
                
                # Análisis de tendencia reciente (últimos 10% de datos)
                recent_data = data.tail(max(1, len(data)//10))
                recent_trend = "creciente" if recent_data[metric].iloc[-1] > recent_data[metric].iloc[0] else "decreciente"
                recent_growth = ((recent_data[metric].iloc[-1] - recent_data[metric].iloc[0]) / recent_data[metric].iloc[0]) * 100 if recent_data[metric].iloc[0] != 0 else 0
                
            else:
                trend = "estable"
                growth_rate = 0
                volatility = 0
                q1 = q3 = iqr = 0
                outliers = 0
                recent_trend = "estable"
                recent_growth = 0
            
            # Período de datos
            if 'date' in data.columns:
                start_date = data['date'].min()
                end_date = data['date'].max()
                duration_days = (end_date - start_date).days
            else:
                start_date = "N/A"
                end_date = "N/A"
                duration_days = 0
            
            # Análisis de distribución
            skewness = data[metric].skew() if len(data) > 2 else 0
            kurtosis = data[metric].kurtosis() if len(data) > 3 else 0
            
            # Análisis de autocorrelación (simplificado)
            if len(data) > 5:
                lag1_corr = data[metric].autocorr(lag=1) if len(data) > 1 else 0
            else:
                lag1_corr = 0
            
            context = f"""
            📊 DATOS HISTÓRICOS DETALLADOS:
            
            📅 PERÍODO DE ANÁLISIS:
            • Fecha inicio: {start_date}
            • Fecha fin: {end_date}
            • Duración: {duration_days} días
            • Registros analizados: {len(data)}
            • Agrupación temporal: {time_grouping}
            
            📈 ESTADÍSTICAS DESCRIPTIVAS:
            • Valor promedio: {mean_val:.2f}
            • Mediana: {median_val:.2f}
            • Desviación estándar: {std_val:.2f}
            • Valor mínimo: {min_val:.2f}
            • Valor máximo: {max_val:.2f}
            • Rango: {max_val - min_val:.2f}
            
            📊 ANÁLISIS DE DISTRIBUCIÓN:
            • Q1 (25%): {q1:.2f}
            • Q3 (75%): {q3:.2f}
            • Rango intercuartílico: {iqr:.2f}
            • Valores atípicos: {outliers}
            • Asimetría: {skewness:.3f}
            • Curtosis: {kurtosis:.3f}
            
            📈 ANÁLISIS DE TENDENCIAS:
            • Tendencia general: {trend}
            • Tasa de crecimiento total: {growth_rate:.2f}%
            • Volatilidad (CV): {volatility:.2f}%
            • Tendencia reciente: {recent_trend}
            • Crecimiento reciente: {recent_growth:.2f}%
            • Autocorrelación (lag-1): {lag1_corr:.3f}
            
            🎯 MÉTRICA ANALIZADA: {metric}
            🔍 AGRUPACIÓN: {time_grouping}
            """
            
            return context
            
        except Exception as e:
            print(f"Error preparando contexto detallado: {e}")
            return "Error preparando datos para análisis"
    
    def _prepare_context(self, data, metric, time_grouping, historical_trends):
        """Preparar contexto básico para Gemini (fallback)"""
        try:
            # Estadísticas básicas
            mean_val = data[metric].mean()
            std_val = data[metric].std()
            min_val = data[metric].min()
            max_val = data[metric].max()
            
            # Tendencias
            if len(data) > 1:
                trend = "creciente" if data[metric].iloc[-1] > data[metric].iloc[0] else "decreciente"
                growth_rate = ((data[metric].iloc[-1] - data[metric].iloc[0]) / data[metric].iloc[0]) * 100
            else:
                trend = "estable"
                growth_rate = 0
            
            # Período de datos
            if 'date' in data.columns:
                start_date = data['date'].min()
                end_date = data['date'].max()
            else:
                start_date = "N/A"
                end_date = "N/A"
            
            context = f"""
            PERÍODO: {start_date} a {end_date}
            REGISTROS: {len(data)}
            VALOR PROMEDIO: {mean_val:.2f}
            DESVIACIÓN ESTÁNDAR: {std_val:.2f}
            RANGO: {min_val:.2f} - {max_val:.2f}
            TENDENCIA: {trend}
            TASA DE CRECIMIENTO: {growth_rate:.2f}%
            AGRUPACIÓN: {time_grouping}
            """
            
            return context
            
        except Exception as e:
            print(f"Error preparando contexto: {e}")
            return "Error preparando datos para análisis"
    

class BlockchainDataProcessor:
    """Clase para procesar y cargar datos de blockchain"""
    
    def __init__(self):
        self.engine = None
    
    def load_daily_metrics(self):
        """Cargar métricas diarias desde CSV"""
        try:
            df = pd.read_csv('data/csv/daily_metrics_processed.csv')
            # Convertir columna date a datetime
            df['date'] = pd.to_datetime(df['date'])
            return df
        except Exception as e:
            print(f"Error cargando CSV: {e}")
            # Crear datos de muestra si no existe el archivo
            return self.create_sample_data()
    
    def load_transactions(self, limit=5000):
        """Cargar transacciones desde CSV"""
        try:
            df = pd.read_csv('data/csv/transactions_processed.csv', nrows=limit)
            # Convertir columna date a datetime
            df['date'] = pd.to_datetime(df['date'])
            return df
        except Exception as e:
            print(f"Error cargando transacciones: {e}")
            return self.create_sample_transactions(limit)
    
    def create_sample_data(self):
        """Crear datos de muestra"""
        dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')
        return pd.DataFrame({
            'date': dates,
            'total_transactions': np.random.randint(1000, 10000, len(dates)),
            'total_volume': np.random.uniform(100000, 1000000, len(dates)),
            'avg_gas_price': np.random.uniform(1e-8, 1e-6, len(dates)),
            'total_gas_used': np.random.randint(100000, 1000000, len(dates)),
            'unique_addresses': np.random.randint(50, 500, len(dates)),
            'new_contracts': np.random.randint(0, 10, len(dates))
        })
    
    def create_sample_transactions(self, limit):
        """Crear transacciones de muestra"""
        return pd.DataFrame({
            'date': pd.date_range(start='2020-01-01', periods=limit, freq='H'),
            'value': np.random.uniform(100, 10000, limit),
            'gas_price': np.random.uniform(1e-8, 1e-6, limit),
            'gas_used': np.random.randint(21000, 100000, limit),
            'transaction_type': np.random.choice(['Transfer', 'Token Transfer', 'Contract Call'], limit),
            'from_address': [f'address_{i}' for i in range(limit)],
            'to_address': [f'address_{i+1}' for i in range(limit)]
        })

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
        
        # Verificar que la columna objetivo existe
        if target_column not in df.columns:
            print(f"Columna {target_column} no encontrada en los datos")
            return df, []
        
        # Crear características temporales
        if 'date' in df.columns:
            df['day_of_week'] = df['date'].dt.dayofweek
            df['month'] = df['date'].dt.month
            df['year'] = df['date'].dt.year
        else:
            # Si no hay fecha, crear características básicas
            df['day_of_week'] = 0
            df['month'] = 1
            df['year'] = 2024
        
        # Crear características de lag (valores pasados) solo si hay suficientes datos
        if len(df) > lookback_days:
            for i in range(1, min(lookback_days + 1, len(df))):
                df[f'{target_column}_lag_{i}'] = df[target_column].shift(i)
        else:
            # Si no hay suficientes datos, crear características básicas
            for i in range(1, lookback_days + 1):
                df[f'{target_column}_lag_{i}'] = 0
        
        # Crear características de media móvil
        for window in [3, 7, 14]:
            if len(df) >= window:
                df[f'{target_column}_ma_{window}'] = df[target_column].rolling(window=window).mean()
            else:
                df[f'{target_column}_ma_{window}'] = df[target_column]
        
        # Crear características de tendencia
        df[f'{target_column}_trend'] = df[target_column].diff()
        
        # Eliminar filas con NaN
        df = df.dropna()
        
        # Seleccionar características disponibles
        feature_columns = []
        for col in df.columns:
            if col not in ['date', target_column] and col in df.columns:
                feature_columns.append(col)
        
        self.feature_columns = feature_columns
        
        return df, feature_columns
    
    def train_model(self, df, target_column, model_type='linear', lookback_days=7):
        """Entrenar modelo de IA"""
        try:
            df_processed, feature_columns = self.prepare_features(df, target_column, lookback_days)
            
            if len(df_processed) < 5:
                print("Datos insuficientes para entrenar el modelo")
                return None
            
            X = df_processed[feature_columns]
            y = df_processed[target_column]
            
            # Verificar que hay suficientes datos
            if len(X) < 3:
                print("Muy pocos datos para entrenar")
                return None
            
            # Dividir datos solo si hay suficientes
            if len(X) >= 10:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            else:
                # Usar todos los datos para entrenar si hay pocos
                X_train = X_test = X
                y_train = y_test = y
            
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
                model = RandomForestRegressor(n_estimators=50, random_state=42)
            else:
                model = LinearRegression()
            
            # Entrenar modelo
            model.fit(X_train_scaled, y_train)
            
            # Evaluar modelo solo si hay datos de prueba
            if len(X_test) > 0:
                y_pred = model.predict(X_test_scaled)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
            else:
                mse = 0
                r2 = 1.0
            
            # Guardar modelo y scaler
            self.models[target_column] = model
            self.scalers[target_column] = scaler
            self.feature_columns = feature_columns
            
            return {
                'model': model,
                'scaler': scaler,
                'mse': mse,
                'r2': r2,
                'feature_columns': feature_columns
            }
            
        except Exception as e:
            print(f"Error entrenando modelo: {e}")
            return None
    
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

# Inicializar procesador de datos, modelo de IA y Gemini
data_processor = BlockchainDataProcessor()
ai_model = AIModel()
gemini_forecaster = GeminiForecaster()

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
            html.H1("Blockchain Analytics Dashboard", className="text-center mb-4"),
            html.P("Analisis de datos blockchain con predicciones de IA", className="text-center text-muted")
        ])
    ]),
    
    # Controles
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Controles de Analisis"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Seleccionar Metrica:"),
                            dcc.Dropdown(
                                id='metric-dropdown',
                                options=[
                                    {'label': 'Transacciones Totales', 'value': 'total_transactions'},
                                    {'label': 'Volumen Total', 'value': 'total_volume'},
                                    {'label': 'Precio Promedio de Gas', 'value': 'avg_gas_price'},
                                    {'label': 'Gas Total Usado', 'value': 'total_gas_used'},
                                    {'label': 'Direcciones Unicas', 'value': 'unique_addresses'},
                                    {'label': 'Nuevos Contratos', 'value': 'new_contracts'}
                                ],
                                value='total_transactions',
                                clearable=False
                            )
                        ], width=4),
                        dbc.Col([
                            html.Label("Agrupacion Temporal:"),
                            dcc.Dropdown(
                                id='time-grouping',
                                options=[
                                    {'label': 'Por Dia', 'value': 'daily'},
                                    {'label': 'Por Año', 'value': 'yearly'},
                                    {'label': 'Por Mes', 'value': 'monthly'}
                                ],
                                value='daily',
                                clearable=False
                            )
                        ], width=4),
                        dbc.Col([
                            html.Label("Modelo de IA:"),
                            dcc.Dropdown(
                                id='model-dropdown',
                                options=[
                                    {'label': 'Regresion Lineal', 'value': 'linear'},
                                    {'label': 'Ridge Regression', 'value': 'ridge'},
                                    {'label': 'Random Forest', 'value': 'random_forest'}
                                ],
                                value='linear',
                                clearable=False
                            )
                        ], width=4)
                    ], className="mb-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Label("Dias de Prediccion:"),
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
                            html.Label("Dias de Lookback:"),
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
                            dbc.Button("Entrenar Modelo", id="train-button", color="primary", className="me-2"),
                            dbc.Button("Actualizar Graficos", id="update-button", color="success", className="me-2"),
                            dbc.Button("🤖 Pronostico IA Blockchain", id="gemini-forecast-button", color="warning")
                        ], className="mt-3")
                    ]),
                    
                    # Sección informativa de modelos de IA
                    html.Hr(),
                    html.H5("🤖 Información de Modelos de IA", className="text-center mb-4"),
                    
                    dbc.Row([
                        # Regresión Lineal
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader([
                                    html.H5("📈 Regresión Lineal", className="mb-0 text-primary")
                                ]),
                                dbc.CardBody([
                                    html.P("Modelo que encuentra la mejor línea recta que se ajusta a los datos históricos.", className="text-muted"),
                                    html.H6("✅ Ventajas:", className="text-success"),
                                    html.Ul([
                                        html.Li("Rápido y eficiente"),
                                        html.Li("Fácil de interpretar"),
                                        html.Li("Bueno para tendencias lineales")
                                    ]),
                                    html.H6("🎯 Mejor para:", className="text-info"),
                                    html.P("Métricas con tendencias consistentes y predicciones simples", className="small")
                                ])
                            ], color="light", outline=True)
                        ], width=4),
                        
                        # Ridge Regression
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader([
                                    html.H5("🛡️ Ridge Regression", className="mb-0 text-warning")
                                ]),
                                dbc.CardBody([
                                    html.P("Regresión lineal con regularización L2 para evitar sobreajuste.", className="text-muted"),
                                    html.H6("✅ Ventajas:", className="text-success"),
                                    html.Ul([
                                        html.Li("Maneja multicolinealidad"),
                                        html.Li("Más robusto que regresión lineal"),
                                        html.Li("Evita sobreajuste")
                                    ]),
                                    html.H6("🎯 Mejor para:", className="text-info"),
                                    html.P("Datos con muchas características correlacionadas", className="small")
                                ])
                            ], color="light", outline=True)
                        ], width=4),
                        
                        # Random Forest
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader([
                                    html.H5("🌲 Random Forest", className="mb-0 text-success")
                                ]),
                                dbc.CardBody([
                                    html.P("Ensemble de árboles de decisión que captura relaciones no lineales.", className="text-muted"),
                                    html.H6("✅ Ventajas:", className="text-success"),
                                    html.Ul([
                                        html.Li("Maneja relaciones complejas"),
                                        html.Li("Resistente a outliers"),
                                        html.Li("No requiere escalado de datos")
                                    ]),
                                    html.H6("🎯 Mejor para:", className="text-info"),
                                    html.P("Patrones complejos y análisis avanzados", className="small")
                                ])
                            ], color="light", outline=True)
                        ], width=4)
                    ], className="mt-3")
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    # Gráficos principales
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Analisis Temporal con Predicciones de IA"),
                dbc.CardBody([
                    dcc.Graph(id='main-chart'),
                    html.Hr(),
                    html.Div(id='main-chart-analysis')
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    # Gráficos secundarios
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Analisis de Dispersion y Correlaciones"),
                dbc.CardBody([
                    dcc.Graph(id='scatter-chart'),
                    html.Hr(),
                    html.Div(id='scatter-chart-analysis')
                ])
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Metricas del Modelo y Rendimiento"),
                dbc.CardBody([
                    html.Div(id='model-metrics'),
                    html.Hr(),
                    html.Div(id='model-performance-analysis')
                ])
            ])
        ], width=6)
    ], className="mb-4"),
    
    # Pronóstico de IA Blockchain
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🤖 Pronostico Inteligente con IA Blockchain"),
                dbc.CardBody([
                    html.Div(id='gemini-forecast')
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    # Información del dataset
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Informacion del Dataset"),
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
     Output('dataset-info', 'children'),
     Output('main-chart-analysis', 'children'),
     Output('scatter-chart-analysis', 'children'),
     Output('model-performance-analysis', 'children'),
     Output('gemini-forecast', 'children')],
    [Input('train-button', 'n_clicks'),
     Input('update-button', 'n_clicks'),
     Input('gemini-forecast-button', 'n_clicks')],
    [State('metric-dropdown', 'value'),
     State('model-dropdown', 'value'),
     State('prediction-days', 'value'),
     State('lookback-days', 'value'),
     State('time-grouping', 'value')]
)
def update_charts(train_clicks, update_clicks, gemini_clicks, selected_metric, model_type, prediction_days, lookback_days, time_grouping):
    """Actualizar gráficos basado en los controles del usuario"""
    
    # Entrenar modelo si se hizo clic en entrenar
    ctx = dash.callback_context
    if ctx.triggered and 'train-button' in ctx.triggered[0]['prop_id']:
        print(f"Entrenando modelo {model_type} para {selected_metric}...")
        model_info = ai_model.train_model(daily_metrics, selected_metric, model_type, lookback_days)
        print(f"Modelo entrenado - R²: {model_info['r2']:.3f}, MSE: {model_info['mse']:.3f}")
    
    # Procesar datos según agrupación temporal
    processed_data = process_temporal_data(daily_metrics, selected_metric, time_grouping)
    
    # Crear gráfico principal
    fig_main = create_main_chart(processed_data, selected_metric, prediction_days, lookback_days, time_grouping)
    
    # Crear gráfico de dispersión
    fig_scatter = create_scatter_chart(processed_data, selected_metric)
    
    # Métricas del modelo
    model_metrics = create_model_metrics(selected_metric)
    
    # Información del dataset
    dataset_info = create_dataset_info()
    
    # Análisis detallados
    main_analysis = create_main_chart_analysis(processed_data, selected_metric, model_type, prediction_days, lookback_days, time_grouping)
    scatter_analysis = create_scatter_chart_analysis(processed_data, selected_metric)
    performance_analysis = create_model_performance_analysis(selected_metric, model_type, prediction_days, lookback_days)
    
    # Pronóstico de Gemini
    gemini_forecast = create_gemini_forecast(processed_data, selected_metric, time_grouping, gemini_clicks)
    
    return fig_main, fig_scatter, model_metrics, dataset_info, main_analysis, scatter_analysis, performance_analysis, gemini_forecast

def process_temporal_data(df, metric, time_grouping):
    """Procesar datos según la agrupación temporal seleccionada"""
    
    if time_grouping == 'daily':
        return df
    elif time_grouping == 'yearly':
        # Agrupar por año
        df['year'] = df['date'].dt.year
        yearly_data = df.groupby('year').agg({
            metric: 'sum',
            'total_volume': 'sum',
            'avg_gas_price': 'mean',
            'total_gas_used': 'sum',
            'unique_addresses': 'mean',
            'new_contracts': 'sum'
        }).reset_index()
        yearly_data['date'] = pd.to_datetime(yearly_data['year'], format='%Y')
        return yearly_data
    elif time_grouping == 'monthly':
        # Agrupar por mes
        df['year_month'] = df['date'].dt.to_period('M')
        monthly_data = df.groupby('year_month').agg({
            metric: 'sum',
            'total_volume': 'sum',
            'avg_gas_price': 'mean',
            'total_gas_used': 'sum',
            'unique_addresses': 'mean',
            'new_contracts': 'sum'
        }).reset_index()
        monthly_data['date'] = monthly_data['year_month'].dt.to_timestamp()
        return monthly_data
    else:
        return df

def create_main_chart(df, metric, prediction_days, lookback_days, time_grouping='daily'):
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
        name='Datos Historicos',
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
            print(f"Error generando predicciones: {e}")
    
    # Configurar layout
    fig.update_layout(
        title=f"{metric.replace('_', ' ').title()} - Analisis Temporal con IA",
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
        title=f"Correlacion: {metric.replace('_', ' ').title()} vs {best_corr_metric.replace('_', ' ').title()}",
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
                        html.H5("Precision del Modelo", className="card-title"),
                        html.P(f"Tipo: {type(model).__name__}", className="mb-1"),
                        html.P(f"Caracteristicas: {len(ai_model.feature_columns)}", className="mb-1"),
                        html.P(f"Datos de entrenamiento: {len(df_clean)} registros", className="mb-1"),
                        html.P(f"Ultima actualizacion: {datetime.now().strftime('%H:%M:%S')}", className="mb-0")
                    ])
                ], color="light")
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Estadisticas", className="card-title"),
                        html.P(f"Media: {df_clean[metric].mean():.2f}", className="mb-1"),
                        html.P(f"Desv. Estandar: {df_clean[metric].std():.2f}", className="mb-1"),
                        html.P(f"Minimo: {df_clean[metric].min():.2f}", className="mb-1"),
                        html.P(f"Maximo: {df_clean[metric].max():.2f}", className="mb-0")
                    ])
                ], color="info")
            ], width=6)
        ])
    else:
        return dbc.Alert("Haz clic en 'Entrenar Modelo' para ver las metricas", color="warning")

def create_dataset_info():
    """Crear información del dataset"""
    
    # Asegurar que las fechas estén en formato datetime
    try:
        if not pd.api.types.is_datetime64_any_dtype(daily_metrics['date']):
            daily_metrics['date'] = pd.to_datetime(daily_metrics['date'])
        
        min_date = daily_metrics['date'].min().strftime('%Y-%m-%d')
        max_date = daily_metrics['date'].max().strftime('%Y-%m-%d')
    except Exception as e:
        print(f"Error procesando fechas: {e}")
        min_date = "N/A"
        max_date = "N/A"
    
    return dbc.Row([
        dbc.Col([
            html.H6("Resumen del Dataset", className="mb-3"),
            html.P(f"Periodo: {min_date} a {max_date}"),
            html.P(f"Registros diarios: {len(daily_metrics):,}"),
            html.P(f"Transacciones: {len(transactions):,}"),
            html.P(f"Gas promedio: {daily_metrics['avg_gas_price'].mean():.2e}"),
            html.P(f"Volumen total: ${daily_metrics['total_volume'].sum():,.2f}")
        ], width=6),
        dbc.Col([
            html.H6("Configuracion", className="mb-3"),
            html.P(f"Base de datos: CSV"),
            html.P(f"Modelos disponibles: Linear, Ridge, Random Forest"),
            html.P(f"Metricas disponibles: 6"),
            html.P(f"Predicciones: Hasta 90 dias"),
            html.P(f"Caracteristicas: Lag, MA, Tendencia")
        ], width=6)
    ])

def create_main_chart_analysis(df, metric, model_type, prediction_days, lookback_days, time_grouping='daily'):
    """Crear análisis detallado del gráfico principal"""
    
    # Calcular estadísticas básicas de forma segura
    try:
        df_clean = df.dropna(subset=[metric])
        if len(df_clean) == 0:
            return dbc.Alert("No hay datos disponibles para esta métrica", color="warning")
        
        mean_val = df_clean[metric].mean()
        std_val = df_clean[metric].std()
        min_val = df_clean[metric].min()
        max_val = df_clean[metric].max()
        trend = "creciente" if len(df_clean) > 1 and df_clean[metric].iloc[-1] > df_clean[metric].iloc[0] else "decreciente"
    except Exception as e:
        print(f"Error calculando estadísticas: {e}")
        return dbc.Alert("Error calculando estadísticas de la métrica", color="danger")
    
    # Información del modelo
    model_info = {
        'linear': {
            'name': 'Regresión Lineal',
            'description': 'Modelo que encuentra la mejor línea recta que se ajusta a los datos históricos',
            'advantages': 'Rápido, interpretable, bueno para tendencias lineales',
            'use_case': 'Ideal para métricas con tendencias consistentes'
        },
        'ridge': {
            'name': 'Ridge Regression',
            'description': 'Regresión lineal con regularización L2 para evitar sobreajuste',
            'advantages': 'Maneja multicolinealidad, más robusto que regresión lineal',
            'use_case': 'Perfecto cuando hay muchas características correlacionadas'
        },
        'random_forest': {
            'name': 'Random Forest',
            'description': 'Ensemble de árboles de decisión que captura relaciones no lineales',
            'advantages': 'Maneja relaciones complejas, resistente a outliers',
            'use_case': 'Excelente para patrones complejos y no lineales'
        }
    }
    
    model_data = model_info.get(model_type, model_info['linear'])
    
    # Información de agrupación temporal
    time_grouping_info = {
        'daily': {'name': 'Por Día', 'description': 'Análisis diario detallado'},
        'monthly': {'name': 'Por Mes', 'description': 'Análisis mensual agregado'},
        'yearly': {'name': 'Por Año', 'description': 'Análisis anual desde 2014'}
    }
    
    grouping_info = time_grouping_info.get(time_grouping, time_grouping_info['daily'])
    
    return dbc.Card([
        dbc.CardHeader([
            html.H5("📊 Análisis del Gráfico Temporal", className="mb-0"),
            html.Small(f"Modelo: {model_data['name']} | Métrica: {metric.replace('_', ' ').title()} | {grouping_info['name']}", className="text-muted")
        ]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H6("📈 Datos Evaluados", className="text-primary"),
                    html.P(f"• Registros analizados: {len(df_clean):,}"),
                    html.P(f"• Valor promedio: {mean_val:,.2f}"),
                    html.P(f"• Desviación estándar: {std_val:,.2f}"),
                    html.P(f"• Rango: {min_val:,.2f} - {max_val:,.2f}"),
                    html.P(f"• Tendencia general: {trend}")
                ], width=6),
                dbc.Col([
                    html.H6("🤖 Modelo de IA Utilizado", className="text-success"),
                    html.P(f"• Algoritmo: {model_data['name']}"),
                    html.P(f"• Descripción: {model_data['description']}"),
                    html.P(f"• Ventajas: {model_data['advantages']}"),
                    html.P(f"• Caso de uso: {model_data['use_case']}")
                ], width=6)
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    html.H6("⚙️ Parámetros de Configuración", className="text-info"),
                    html.P(f"• Agrupación temporal: {grouping_info['name']}"),
                    html.P(f"• Descripción: {grouping_info['description']}"),
                    html.P(f"• Días de Predicción: {prediction_days} días"),
                    html.P(f"• Días de Lookback: {lookback_days} días"),
                    html.P(f"• Características temporales: Día de semana, mes, año")
                ], width=6),
                dbc.Col([
                    html.H6("💡 Justificación del Análisis", className="text-warning"),
                    html.P(f"• El modelo {model_data['name']} es ideal para {metric.replace('_', ' ')} porque:"),
                    html.P(f"• {model_data['use_case']}"),
                    html.P(f"• {grouping_info['description']} desde 2014"),
                    html.P(f"• Los {lookback_days} días de lookback capturan patrones estacionales"),
                    html.P(f"• Las predicciones de {prediction_days} días permiten planificación estratégica")
                ], width=6)
            ])
        ])
    ], color="light")

def create_scatter_chart_analysis(df, metric):
    """Crear análisis del gráfico de dispersión"""
    
    # Calcular correlaciones
    numeric_cols = ['total_transactions', 'total_volume', 'avg_gas_price', 
                   'total_gas_used', 'unique_addresses', 'new_contracts']
    
    correlations = df[numeric_cols].corr()[metric].abs().sort_values(ascending=False)
    best_corr_metric = correlations.index[1] if len(correlations) > 1 else 'total_volume'
    correlation_strength = correlations[best_corr_metric]
    
    # Interpretar correlación
    if correlation_strength > 0.7:
        corr_interpretation = "fuerte correlación positiva"
    elif correlation_strength > 0.3:
        corr_interpretation = "correlación moderada"
    else:
        corr_interpretation = "correlación débil"
    
    return dbc.Card([
        dbc.CardHeader([
            html.H5("🔍 Análisis de Correlaciones", className="mb-0"),
            html.Small(f"Métrica principal: {metric.replace('_', ' ').title()}", className="text-muted")
        ]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H6("📊 Correlación Principal", className="text-primary"),
                    html.P(f"• Variable correlacionada: {best_corr_metric.replace('_', ' ').title()}"),
                    html.P(f"• Fuerza de correlación: {correlation_strength:.3f}"),
                    html.P(f"• Interpretación: {corr_interpretation}"),
                    html.P(f"• Significado: Las dos métricas tienden a moverse juntas")
                ], width=6),
                dbc.Col([
                    html.H6("🎯 Insights del Gráfico", className="text-success"),
                    html.P(f"• El tamaño de los puntos representa gas total usado"),
                    html.P(f"• El color indica precio promedio de gas"),
                    html.P(f"• Los puntos más grandes = mayor consumo de gas"),
                    html.P(f"• Los colores más intensos = gas más caro"),
                    html.P(f"• Patrones visibles = relaciones entre variables")
                ], width=6)
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    html.H6("💡 Interpretación de Datos", className="text-info"),
                    html.P(f"• Si los puntos forman una línea diagonal: correlación fuerte"),
                    html.P(f"• Si los puntos están dispersos: poca correlación"),
                    html.P(f"• Agrupaciones de puntos: patrones estacionales"),
                    html.P(f"• Outliers: eventos inusuales o anomalías")
                ], width=12)
            ])
        ])
    ], color="light")

def create_model_performance_analysis(metric, model_type, prediction_days, lookback_days):
    """Crear análisis de rendimiento del modelo"""
    
    if metric in ai_model.models:
        model = ai_model.models[metric]
        scaler = ai_model.scalers[metric]
        
        # Información del modelo entrenado
        model_name = type(model).__name__
        n_features = len(ai_model.feature_columns) if hasattr(ai_model, 'feature_columns') else 0
        
        # Calcular métricas de rendimiento de forma segura
        try:
            df_clean = daily_metrics.dropna(subset=[metric])
            if len(df_clean) > 0 and n_features > 0:
                # Usar solo las columnas que existen
                available_features = [col for col in ai_model.feature_columns if col in df_clean.columns]
                if available_features:
                    X_test = df_clean[available_features].iloc[-min(100, len(df_clean)):]
                    y_test = df_clean[metric].iloc[-min(100, len(df_clean)):]
                    X_test_scaled = scaler.transform(X_test)
                    r2_score_val = model.score(X_test_scaled, y_test)
                else:
                    r2_score_val = 0.0
            else:
                r2_score_val = 0.0
        except Exception as e:
            print(f"Error calculando R²: {e}")
            r2_score_val = 0.0
        
        return dbc.Card([
            dbc.CardHeader([
                html.H5("🎯 Rendimiento del Modelo", className="mb-0"),
                html.Small(f"Modelo entrenado: {model_name}", className="text-muted")
            ]),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H6("📈 Métricas de Precisión", className="text-primary"),
                        html.P(f"• R² Score: {r2_score_val:.3f}"),
                        html.P(f"• Características utilizadas: {n_features}"),
                        html.P(f"• Datos de entrenamiento: {len(daily_metrics):,} registros"),
                        html.P(f"• Última actualización: {datetime.now().strftime('%H:%M:%S')}")
                    ], width=6),
                    dbc.Col([
                        html.H6("⚙️ Configuración del Modelo", className="text-success"),
                        html.P(f"• Algoritmo: {model_name}"),
                        html.P(f"• Días de predicción: {prediction_days}"),
                        html.P(f"• Días de lookback: {lookback_days}"),
                        html.P(f"• Características: Lag, MA, tendencias temporales"),
                        html.P(f"• Escalado: StandardScaler aplicado")
                    ], width=6)
                ], className="mb-3"),
                
                dbc.Row([
                    dbc.Col([
                        html.H6("💡 Interpretación del Rendimiento", className="text-info"),
                        html.P(f"• R² > 0.8: Excelente predicción"),
                        html.P(f"• R² 0.6-0.8: Buena predicción"),
                        html.P(f"• R² 0.4-0.6: Predicción moderada"),
                        html.P(f"• R² < 0.4: Predicción limitada"),
                        html.P(f"• Más características = mayor complejidad")
                    ], width=12)
                ])
            ])
        ], color="light")
    else:
        return dbc.Alert([
            html.H6("⚠️ Modelo No Entrenado", className="alert-heading"),
            html.P("Haz clic en 'Entrenar Modelo' para ver el análisis de rendimiento."),
            html.Hr(),
            html.P("Una vez entrenado, verás:"),
            html.Ul([
                html.Li("Métricas de precisión (R², MSE)"),
                html.Li("Características utilizadas"),
                html.Li("Interpretación del rendimiento"),
                html.Li("Recomendaciones de mejora")
            ])
        ], color="warning")

def create_gemini_forecast(data, metric, time_grouping, gemini_clicks):
    """Crear pronóstico usando IA Blockchain"""
    
    if gemini_clicks and gemini_clicks > 0:
        try:
            print(f"Generando pronóstico con IA Blockchain para {metric} ({time_grouping})...")
            
            # Mostrar pantalla de carga
            loading_card = dbc.Card([
                dbc.CardHeader([
                    html.H5("🤖 Generando Pronóstico con IA Blockchain", className="mb-0"),
                    html.Small(f"Métrica: {metric.replace('_', ' ').title()} | Agrupación: {time_grouping}", className="text-muted")
                ]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Spinner([
                                html.Div([
                                    html.H6("🔄 Procesando datos con IA Blockchain...", className="text-center"),
                                    html.P("Analizando tendencias históricas y generando pronóstico inteligente...", className="text-center text-muted"),
                                    html.P("Esto puede tomar unos segundos...", className="text-center text-muted")
                                ], className="text-center p-4")
                            ], size="lg", color="primary"),
                        ], width=12)
                    ])
                ])
            ], color="light")
            
            # Generar pronóstico con IA Blockchain
            forecast = gemini_forecaster.generate_forecast(data, metric, time_grouping, {})
            
            # Verificar si es un error
            if forecast.startswith("❌ Error"):
                return dbc.Alert([
                    html.H6("❌ Error de IA Blockchain", className="alert-heading"),
                    html.P(forecast, className="mb-0")
                ], color="danger")
            
            return dbc.Card([
                dbc.CardHeader([
                    html.H5("🤖 Pronóstico Inteligente con IA Blockchain", className="mb-0"),
                    html.Small(f"Métrica: {metric.replace('_', ' ').title()} | Agrupación: {time_grouping}", className="text-muted")
                ]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.P(forecast, className="mb-0", style={"white-space": "pre-wrap", "text-align": "justify"})
                            ], style={"maxHeight": "600px", "overflowY": "auto"})
                        ], width=12)
                    ]),
                    html.Hr(),
                    dbc.Row([
                        dbc.Col([
                            html.Small([
                                html.Strong("💡 Nota: "),
                                "Este pronóstico extenso es generado por IA Blockchain basado en los datos históricos y tendencias del mercado blockchain. ",
                                "Incluye análisis detallado, múltiples escenarios y recomendaciones estratégicas."
                            ], className="text-muted")
                        ], width=12)
                    ])
                ])
            ], color="light")
            
        except Exception as e:
            print(f"Error generando pronóstico: {e}")
            return dbc.Alert([
                html.H6("❌ Error Generando Pronóstico", className="alert-heading"),
                html.P(f"Error: {str(e)}"),
                html.P("Por favor, verifica tu conexión a internet y la API key de IA Blockchain.")
            ], color="danger")
    else:
        return dbc.Alert([
            html.H6("🤖 Pronóstico Extenso con IA Blockchain", className="alert-heading"),
            html.P("Haz clic en '🤖 Pronóstico IA Blockchain' para generar un análisis inteligente extenso."),
            html.Hr(),
            html.P("El pronóstico extenso incluirá:"),
            html.Ul([
                html.Li("📊 Análisis histórico detallado con patrones identificados"),
                html.Li("🔍 Factores técnicos, de mercado, regulatorios y tecnológicos"),
                html.Li("🔮 Pronósticos para 3, 6 y 12 meses con escenarios múltiples"),
                html.Li("📈 Análisis técnico avanzado con métricas de confianza"),
                html.Li("💡 Recomendaciones estratégicas de inversión"),
                html.Li("🎯 Insights específicos y alertas tempranas")
            ]),
            html.P("Análisis de 500 palabras con información extremadamente detallada.", className="text-muted")
        ], color="info")

# Función para ejecutar la aplicación
def run_app():
    """Ejecutar la aplicación Dash"""
    print("Iniciando Blockchain Analytics Dashboard...")
    print("Dashboard disponible en: http://localhost:8050")
    print("Presiona Ctrl+C para detener la aplicacion")
    
    app.run(debug=True, host='127.0.0.1', port=8050)

if __name__ == '__main__':
    run_app()
