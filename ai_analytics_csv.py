"""
Análisis de IA usando archivos CSV (sin MySQL)
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
import os
from config import CSV_DIR
import warnings
warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BlockchainAIAnalyticsCSV:
    """Clase para análisis de IA usando archivos CSV"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def load_data_from_csv(self) -> pd.DataFrame:
        """Cargar datos desde archivos CSV"""
        try:
            # Cargar métricas diarias
            csv_file = os.path.join(CSV_DIR, 'daily_metrics_processed.csv')
            
            if not os.path.exists(csv_file):
                logger.warning(f"Archivo CSV no encontrado: {csv_file}")
                return pd.DataFrame()
            
            df = pd.read_csv(csv_file)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            logger.info(f"Datos cargados desde CSV: {len(df)} registros")
            return df
            
        except Exception as e:
            logger.error(f"Error cargando datos CSV: {e}")
            return pd.DataFrame()
    
    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detectar anomalías en los datos"""
        try:
            logger.info("Iniciando detección de anomalías...")
            
            # Seleccionar características
            features = ['total_transactions', 'total_volume', 'avg_gas_price', 
                       'total_gas_used', 'unique_addresses', 'new_contracts']
            
            # Preparar datos
            X = df[features].fillna(0)
            X_scaled = self.scaler.fit_transform(X)
            
            # Aplicar Isolation Forest
            isolation_forest = IsolationForest(
                contamination=0.1,
                random_state=42
            )
            
            anomaly_labels = isolation_forest.fit_predict(X_scaled)
            anomaly_scores = isolation_forest.decision_function(X_scaled)
            
            # Agregar resultados
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
        """Predecir métricas futuras"""
        try:
            logger.info(f"Iniciando predicción para {days_ahead} días...")
            
            # Preparar datos
            df_sorted = df.sort_values('date').copy()
            df_sorted['days_since_start'] = (df_sorted['date'] - df_sorted['date'].min()).dt.days
            
            # Métricas a predecir
            metrics = ['total_transactions', 'total_volume', 'avg_gas_price', 'total_gas_used']
            predictions = []
            
            for metric in metrics:
                # Entrenar modelo
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
    
    def analyze_patterns(self, df: pd.DataFrame) -> dict:
        """Analizar patrones en los datos"""
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
    
    def save_insights_to_csv(self, df: pd.DataFrame, predictions_df: pd.DataFrame, patterns: dict):
        """Guardar insights en archivos CSV"""
        try:
            # Guardar anomalías
            anomalies = df[df['anomaly']]
            if not anomalies.empty:
                anomalies_file = os.path.join(CSV_DIR, 'ai_anomalies.csv')
                anomalies.to_csv(anomalies_file, index=False)
                logger.info(f"Anomalías guardadas en: {anomalies_file}")
            
            # Guardar predicciones
            if not predictions_df.empty:
                predictions_file = os.path.join(CSV_DIR, 'ai_predictions.csv')
                predictions_df.to_csv(predictions_file, index=False)
                logger.info(f"Predicciones guardadas en: {predictions_file}")
            
            # Guardar patrones
            if patterns:
                patterns_file = os.path.join(CSV_DIR, 'ai_patterns.json')
                import json
                with open(patterns_file, 'w') as f:
                    json.dump(patterns, f, indent=2, default=str)
                logger.info(f"Patrones guardados en: {patterns_file}")
            
        except Exception as e:
            logger.error(f"Error guardando insights: {e}")
    
    def generate_ai_report(self):
        """Generar reporte completo de análisis de IA"""
        try:
            logger.info("=== INICIANDO ANÁLISIS DE IA (CSV) ===")
            
            # Cargar datos
            df = self.load_data_from_csv()
            if df.empty:
                logger.warning("No hay datos para analizar")
                return
            
            # Detectar anomalías
            df_with_anomalies = self.detect_anomalies(df)
            
            # Hacer predicciones
            predictions = self.predict_future_metrics(df_with_anomalies, days_ahead=7)
            
            # Analizar patrones
            patterns = self.analyze_patterns(df_with_anomalies)
            
            # Guardar insights
            self.save_insights_to_csv(df_with_anomalies, predictions, patterns)
            
            # Mostrar resumen
            logger.info("=== RESUMEN DE ANÁLISIS DE IA ===")
            logger.info(f"Total de días analizados: {len(df)}")
            logger.info(f"Anomalías detectadas: {len(df_with_anomalies[df_with_anomalies['anomaly']])}")
            logger.info(f"Predicciones generadas: {len(predictions)}")
            
            logger.info("=== ANÁLISIS DE IA COMPLETADO ===")
            
        except Exception as e:
            logger.error(f"Error en análisis de IA: {e}")

def main():
    """Función principal"""
    try:
        ai_analytics = BlockchainAIAnalyticsCSV()
        ai_analytics.generate_ai_report()
    except Exception as e:
        logger.error(f"Error fatal en análisis de IA: {e}")

if __name__ == "__main__":
    main()
