"""
Script de inicio rápido para el Blockchain Analytics Dashboard
"""

import os
import sys
import subprocess
import webbrowser
import time
from threading import Timer

def check_dependencies():
    """Verificar que las dependencias estén instaladas"""
    try:
        import dash
        import plotly
        import pandas
        import numpy
        import sklearn
        import dash_bootstrap_components
        return True
    except ImportError as e:
        print(f"Error: Dependencia faltante: {e}")
        print("Ejecuta: pip install -r requirements.txt")
        return False

def open_browser():
    """Abrir navegador automáticamente"""
    try:
        webbrowser.open('http://localhost:8050')
        print("Navegador abierto en http://localhost:8050")
    except Exception as e:
        print(f"No se pudo abrir el navegador automáticamente: {e}")
        print("Abre manualmente: http://localhost:8050")

def start_dashboard():
    """Iniciar el dashboard"""
    print("="*60)
    print("BLOCKCHAIN ANALYTICS DASHBOARD CON IA")
    print("="*60)
    print()
    
    # Verificar dependencias
    if not check_dependencies():
        return False
    
    print("✓ Dependencias verificadas")
    print("✓ Cargando datos...")
    print("✓ Iniciando servidor...")
    print()
    print("🌐 Dashboard disponible en: http://localhost:8050")
    print("🔄 Presiona Ctrl+C para detener")
    print()
    
    # Abrir navegador después de 3 segundos
    Timer(3.0, open_browser).start()
    
    try:
        # Importar y ejecutar la aplicación
        from app import run_app
        run_app()
    except KeyboardInterrupt:
        print("\n\n👋 Dashboard detenido por el usuario")
    except Exception as e:
        print(f"\n❌ Error iniciando dashboard: {e}")
        return False
    
    return True

def main():
    """Función principal"""
    print("🚀 Iniciando Blockchain Analytics Dashboard...")
    
    # Verificar si estamos en el directorio correcto
    if not os.path.exists('app.py'):
        print("❌ Error: No se encontró app.py")
        print("Asegúrate de estar en el directorio correcto")
        return False
    
    # Iniciar dashboard
    return start_dashboard()

if __name__ == "__main__":
    main()


