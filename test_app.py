"""
Script de prueba para verificar que la aplicación funciona correctamente
"""

import sys
import os

def test_imports():
    """Probar que todas las dependencias se pueden importar"""
    try:
        print("Probando importaciones...")
        
        import pandas as pd
        print("OK pandas")
        
        import numpy as np
        print("OK numpy")
        
        import plotly.graph_objs as go
        import plotly.express as px
        print("OK plotly")
        
        import dash
        from dash import dcc, html, Input, Output
        print("OK dash")
        
        import dash_bootstrap_components as dbc
        print("OK dash-bootstrap-components")
        
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor
        print("OK scikit-learn")
        
        import mysql.connector
        print("OK mysql-connector")
        
        from sqlalchemy import create_engine
        print("OK sqlalchemy")
        
        print("\nTodas las dependencias estan disponibles")
        return True
        
    except ImportError as e:
        print(f"Error importando dependencias: {e}")
        return False

def test_data_loading():
    """Probar carga de datos"""
    try:
        print("\nProbando carga de datos...")
        
        # Probar carga desde CSV
        import pandas as pd
        
        if os.path.exists('data/csv/daily_metrics_processed.csv'):
            df = pd.read_csv('data/csv/daily_metrics_processed.csv')
            print(f"OK CSV cargado: {len(df)} registros")
        else:
            print("Archivo CSV no encontrado, se creara automaticamente")
        
        return True
        
    except Exception as e:
        print(f"Error cargando datos: {e}")
        return False

def test_mysql_connection():
    """Probar conexión a MySQL"""
    try:
        print("\nProbando conexión a MySQL...")
        
        import mysql.connector
        from config import DB_CONFIG
        
        connection = mysql.connector.connect(**DB_CONFIG)
        connection.close()
        print("OK Conexion a MySQL exitosa")
        return True
        
    except Exception as e:
        print(f"No se pudo conectar a MySQL: {e}")
        print("La aplicacion usara archivos CSV como respaldo")
        return False

def test_ai_models():
    """Probar modelos de IA"""
    try:
        print("\nProbando modelos de IA...")
        
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor
        import numpy as np
        
        # Crear datos de prueba
        X = np.random.rand(100, 3)
        y = np.random.rand(100)
        
        # Probar regresión lineal
        model = LinearRegression()
        model.fit(X, y)
        predictions = model.predict(X[:10])
        print("OK Regresion Lineal")
        
        # Probar Random Forest
        rf = RandomForestRegressor(n_estimators=10, random_state=42)
        rf.fit(X, y)
        rf_predictions = rf.predict(X[:10])
        print("OK Random Forest")
        
        print("Modelos de IA funcionando correctamente")
        return True
        
    except Exception as e:
        print(f"Error probando modelos de IA: {e}")
        return False

def main():
    """Función principal de prueba"""
    print("Iniciando pruebas del Blockchain Analytics Dashboard...")
    
    tests = [
        ("Importaciones", test_imports),
        ("Carga de Datos", test_data_loading),
        ("Conexión MySQL", test_mysql_connection),
        ("Modelos IA", test_ai_models)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"Error en {test_name}: {e}")
            results.append((test_name, False))
    
    # Resumen de resultados
    print("\n" + "="*50)
    print("RESUMEN DE PRUEBAS")
    print("="*50)
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nPruebas pasadas: {passed}/{len(results)}")
    
    if passed == len(results):
        print("\n¡Todas las pruebas pasaron! La aplicacion esta lista.")
        print("Para ejecutar: python app.py")
        print("Dashboard: http://localhost:8050")
    else:
        print("\nAlgunas pruebas fallaron. Revisa los errores arriba.")
        print("Ejecuta: python install_dependencies.py")
    
    return passed == len(results)

if __name__ == "__main__":
    main()
