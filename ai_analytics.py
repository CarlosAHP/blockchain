"""
Módulo de análisis de IA para detección de anomalías y predicciones
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
from config import DB_CONFIG
from sqlalchemy import create_engine, text
import warnings
warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BlockchainAIAnalytics:
    """Clase para análisis de IA en datos de blockchain"""
    
    def __init__(self):
        self.engine = None
        self.scaler = StandardScaler()
        self._connect()
    
    def _connect(self):
        """Conectar a la base de datos"""
        try:
            connection_string = f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
            self.engine = create_engine(connection_string, echo=False)
            logger.info("Conexión a base de datos establecida para análisis de IA")
        except Exception as e:
            logger.error(f"Error conectando a base de datos: {e}")
            raise
    
    def load_data_for_analysis(self) -> pd.DataFrame:
        """Cargar datos para análisis de IA"""
        try:
            with self.engine.connect() as conn:
                # Cargar datos de transacciones con métricas diarias
                query = """
                SELECT 
                    t.date,
                    t.total_transactions,
                    t.total_volume,
                    t.avg_gas_price,
                    t.total_gas_used,
                    t.unique_senders,
                    t.unique_receivers,
                    t.unique_contracts
                FROM v_daily_transaction_summary t
                ORDER BY t.date
                """
                
                df = pd.read_sql(query, conn)
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
                
                logger.info(f"Datos cargados para análisis: {len(df)} registros")
                return df
                
        except Exception as e:
            logger.error(f"Error cargando datos: {e}")
            return pd.DataFrame()
    
    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detectar anomalías en los datos de blockchain
        
        Args:
            df: DataFrame con datos de transacciones
            
        Returns:
            DataFrame con anomalías detectadas
        """
        try:
            logger.info("Iniciando detección de anomalías...")
            
            # Seleccionar características para detección de anomalías
            features = ['total_transactions', 'total_volume', 'avg_gas_price', 
                       'total_gas_used', 'unique_senders', 'unique_receivers']
            
            # Preparar datos
            X = df[features].fillna(0)
            X_scaled = self.scaler.fit_transform(X)
            
            # Aplicar Isolation Forest
            isolation_forest = IsolationForest(
                contamination=0.1,  # Esperamos 10% de anomalías
                random_state=42
            )
            
            anomaly_labels = isolation_forest.fit_predict(X_scaled)
            anomaly_scores = isolation_forest.decision_function(X_scaled)
            
            # Agregar resultados al DataFrame
            df['anomaly'] = anomaly_labels == -1
            df['anomaly_score'] = anomaly_scores
            
            # Identificar anomalías
            anomalies = df[df['anomaly']].copy()
            
            logger.info(f"Anomalías detectadas: {len(anomalies)} días")
            
            if len(anomalies) > 0:
                logger.info("Días con anomalías:")
                for _, row in anomalies.iterrows():
                    logger.info(f"  - {row['date'].strftime('%Y-%m-%d')}: "
                              f"Transacciones={row['total_transactions']}, "
                              f"Volumen={row['total_volume']:.2f}, "
                              f"Score={row['anomaly_score']:.3f}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error en detección de anomalías: {e}")
            return df
    
    def predict_future_metrics(self, df: pd.DataFrame, days_ahead: int = 7) -> pd.DataFrame:
        """
        Predecir métricas futuras usando regresión lineal
        
        Args:
            df: DataFrame con datos históricos
            days_ahead: Días a predecir hacia el futuro
            
        Returns:
            DataFrame con predicciones
        """
        try:
            logger.info(f"Iniciando predicción para {days_ahead} días...")
            
            # Preparar datos para predicción
            df_sorted = df.sort_values('date').copy()
            df_sorted['days_since_start'] = (df_sorted['date'] - df_sorted['date'].min()).dt.days
            
            # Métricas a predecir
            metrics = ['total_transactions', 'total_volume', 'avg_gas_price', 'total_gas_used']
            predictions = []
            
            for metric in metrics:
                # Entrenar modelo de regresión lineal
                X = df_sorted[['days_since_start']].values
                y = df_sorted[metric].values
                
                model = LinearRegression()
                model.fit(X, y)
                
                # Hacer predicciones
                last_day = df_sorted['days_since_start'].max()
                future_days = np.arange(last_day + 1, last_day + days_ahead + 1).reshape(-1, 1)
                future_predictions = model.predict(future_days)
                
                # Calcular métricas del modelo
                y_pred = model.predict(X)
                mse = mean_squared_error(y, y_pred)
                r2 = r2_score(y, y_pred)
                
                logger.info(f"Modelo para {metric}: R² = {r2:.3f}, MSE = {mse:.2f}")
                
                # Crear DataFrame de predicciones
                for i, (day, pred) in enumerate(zip(future_days.flatten(), future_predictions)):
                    pred_date = df_sorted['date'].min() + timedelta(days=int(day))
                    predictions.append({
                        'date': pred_date,
                        'metric': metric,
                        'predicted_value': pred,
                        'confidence': r2,
                        'days_ahead': i + 1
                    })
            
            predictions_df = pd.DataFrame(predictions)
            logger.info(f"Predicciones generadas: {len(predictions_df)} registros")
            
            return predictions_df
            
        except Exception as e:
            logger.error(f"Error en predicciones: {e}")
            return pd.DataFrame()
    
    def analyze_transaction_patterns(self, df: pd.DataFrame) -> dict:
        """
        Analizar patrones en las transacciones
        
        Args:
            df: DataFrame con datos de transacciones
            
        Returns:
            Diccionario con análisis de patrones
        """
        try:
            logger.info("Analizando patrones de transacciones...")
            
            # Análisis de tendencias
            df['day_of_week'] = df['date'].dt.day_name()
            df['month'] = df['date'].dt.month
            df['quarter'] = df['date'].dt.quarter
            
            patterns = {}
            
            # Patrones por día de la semana
            daily_patterns = df.groupby('day_of_week').agg({
                'total_transactions': 'mean',
                'total_volume': 'mean',
                'avg_gas_price': 'mean'
            }).round(2)
            
            patterns['daily_patterns'] = daily_patterns.to_dict()
            
            # Patrones por mes
            monthly_patterns = df.groupby('month').agg({
                'total_transactions': 'mean',
                'total_volume': 'mean',
                'avg_gas_price': 'mean'
            }).round(2)
            
            patterns['monthly_patterns'] = monthly_patterns.to_dict()
            
            # Correlaciones
            correlation_matrix = df[['total_transactions', 'total_volume', 
                                   'avg_gas_price', 'total_gas_used']].corr()
            patterns['correlations'] = correlation_matrix.to_dict()
            
            # Estadísticas descriptivas
            patterns['descriptive_stats'] = df[['total_transactions', 'total_volume', 
                                              'avg_gas_price', 'total_gas_used']].describe().to_dict()
            
            logger.info("Análisis de patrones completado")
            return patterns
            
        except Exception as e:
            logger.error(f"Error analizando patrones: {e}")
            return {}
    
    def create_ai_insights_table(self, df: pd.DataFrame, predictions_df: pd.DataFrame, patterns: dict):
        """
        Crear tabla con insights de IA para Power BI
        
        Args:
            df: DataFrame con datos y anomalías
            predictions_df: DataFrame con predicciones
            patterns: Diccionario con patrones analizados
        """
        try:
            with self.engine.connect() as conn:
                # Crear tabla de insights de IA
                create_table_sql = """
                CREATE TABLE IF NOT EXISTS ai_insights (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    date DATE,
                    insight_type VARCHAR(50),
                    insight_value DECIMAL(20,8),
                    insight_description TEXT,
                    confidence_score DECIMAL(5,3),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """
                
                conn.execute(text(create_table_sql))
                conn.commit()
                
                # Limpiar datos anteriores
                conn.execute(text("DELETE FROM ai_insights"))
                conn.commit()
                
                # Insertar anomalías detectadas
                anomalies = df[df['anomaly']]
                for _, row in anomalies.iterrows():
                    insert_sql = """
                    INSERT INTO ai_insights (date, insight_type, insight_value, insight_description, confidence_score)
                    VALUES (%s, %s, %s, %s, %s)
                    """
                    conn.execute(text(insert_sql), (
                        row['date'].date(),
                        'anomaly',
                        abs(row['anomaly_score']),
                        f"Anomalía detectada: {row['total_transactions']} transacciones, volumen {row['total_volume']:.2f}",
                        min(abs(row['anomaly_score']) * 10, 1.0)
                    ))
                
                # Insertar predicciones
                for _, row in predictions_df.iterrows():
                    insert_sql = """
                    INSERT INTO ai_insights (date, insight_type, insight_value, insight_description, confidence_score)
                    VALUES (%s, %s, %s, %s, %s)
                    """
                    conn.execute(text(insert_sql), (
                        row['date'].date(),
                        f"prediction_{row['metric']}",
                        row['predicted_value'],
                        f"Predicción de {row['metric']} para {row['days_ahead']} días adelante",
                        row['confidence']
                    ))
                
                conn.commit()
                logger.info("Tabla de insights de IA creada exitosamente")
                
        except Exception as e:
            logger.error(f"Error creando tabla de insights: {e}")
    
    def generate_ai_report(self):
        """Generar reporte completo de análisis de IA"""
        try:
            logger.info("=== INICIANDO ANÁLISIS DE IA ===")
            
            # Cargar datos
            df = self.load_data_for_analysis()
            if df.empty:
                logger.warning("No hay datos para analizar")
                return
            
            # Detectar anomalías
            df_with_anomalies = self.detect_anomalies(df)
            
            # Hacer predicciones
            predictions = self.predict_future_metrics(df_with_anomalies, days_ahead=7)
            
            # Analizar patrones
            patterns = self.analyze_transaction_patterns(df_with_anomalies)
            
            # Crear tabla de insights
            self.create_ai_insights_table(df_with_anomalies, predictions, patterns)
            
            # Mostrar resumen
            logger.info("=== RESUMEN DE ANÁLISIS DE IA ===")
            logger.info(f"Total de días analizados: {len(df)}")
            logger.info(f"Anomalías detectadas: {len(df_with_anomalies[df_with_anomalies['anomaly']])}")
            logger.info(f"Predicciones generadas: {len(predictions)}")
            
            logger.info("=== ANÁLISIS DE IA COMPLETADO ===")
            
        except Exception as e:
            logger.error(f"Error en análisis de IA: {e}")

def main():
    """Función principal para ejecutar análisis de IA"""
    try:
        ai_analytics = BlockchainAIAnalytics()
        ai_analytics.generate_ai_report()
    except Exception as e:
        logger.error(f"Error fatal en análisis de IA: {e}")

if __name__ == "__main__":
    main()
