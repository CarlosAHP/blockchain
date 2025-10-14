"""
Script de instalación de dependencias para Blockchain Analytics Dashboard
"""

import subprocess
import sys
import os

def install_requirements():
    """Instalar dependencias desde requirements.txt"""
    try:
        print("Instalando dependencias...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Dependencias instaladas correctamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error instalando dependencias: {e}")
        return False

def check_mysql_connection():
    """Verificar conexión a MySQL"""
    try:
        import mysql.connector
        from config import DB_CONFIG
        
        connection = mysql.connector.connect(**DB_CONFIG)
        connection.close()
        print("Conexion a MySQL verificada")
        return True
    except Exception as e:
        print(f"No se pudo conectar a MySQL: {e}")
        print("La aplicacion usara archivos CSV como respaldo")
        return False

def create_sample_data():
    """Crear datos de muestra si no hay conexión a MySQL"""
    try:
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        # Crear directorio de datos si no existe
        os.makedirs('data/csv', exist_ok=True)
        
        # Generar datos de muestra
        dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')
        
        # Crear métricas diarias sintéticas
        daily_metrics = pd.DataFrame({
            'date': dates,
            'total_transactions': np.random.randint(1000, 10000, len(dates)),
            'total_volume': np.random.uniform(100000, 1000000, len(dates)),
            'avg_gas_price': np.random.uniform(1e-8, 1e-6, len(dates)),
            'total_gas_used': np.random.randint(100000, 1000000, len(dates)),
            'unique_addresses': np.random.randint(50, 500, len(dates)),
            'new_contracts': np.random.randint(0, 10, len(dates))
        })
        
        # Guardar datos de muestra
        daily_metrics.to_csv('data/csv/daily_metrics_processed.csv', index=False)
        print("Datos de muestra creados")
        return True
        
    except Exception as e:
        print(f"Error creando datos de muestra: {e}")
        return False

def main():
    """Función principal de instalación"""
    print("Configurando Blockchain Analytics Dashboard...")
    
    # Instalar dependencias
    if not install_requirements():
        return False
    
    # Verificar MySQL
    mysql_available = check_mysql_connection()
    
    # Crear datos de muestra si no hay MySQL
    if not mysql_available:
        create_sample_data()
    
    print("\n¡Configuracion completada!")
    print("Para ejecutar la aplicacion:")
    print("   python app.py")
    print("Dashboard disponible en: http://localhost:8050")
    
    return True

if __name__ == "__main__":
    main()
