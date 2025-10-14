#!/usr/bin/env python3
"""
Script de prueba para verificar que Gemini AI funcione correctamente
"""

import google.generativeai as genai
import pandas as pd
from datetime import datetime

# Configurar Gemini API
GEMINI_API_KEY = "AIzaSyAWHUUqZdCx0HZTvHjpktaTbgDytqT3Bx8"
genai.configure(api_key=GEMINI_API_KEY)

def test_gemini_connection():
    """Probar conexión con Gemini AI"""
    print("=== PRUEBA DE CONEXION CON GEMINI AI ===")
    
    try:
        # Intentar con gemini-2.5-flash (más reciente)
        print("1. Probando modelo gemini-2.5-flash...")
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content("Hola, ¿puedes responder con 'OK' para confirmar que funcionas?")
        print(f"   Respuesta: {response.text}")
        print("   OK - gemini-2.5-flash funciona correctamente")
        return True
        
    except Exception as e:
        print(f"   ERROR con gemini-2.5-flash: {e}")
        
        try:
            # Intentar con gemini-1.5-flash
            print("2. Probando modelo gemini-1.5-flash...")
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content("Hola, ¿puedes responder con 'OK' para confirmar que funcionas?")
            print(f"   Respuesta: {response.text}")
            print("   OK - gemini-1.5-flash funciona correctamente")
            return True
            
        except Exception as e2:
            print(f"   ERROR con gemini-1.5-flash: {e2}")
            
            try:
                # Intentar con gemini-pro
                print("3. Probando modelo gemini-pro...")
                model = genai.GenerativeModel('gemini-pro')
                response = model.generate_content("Hola, ¿puedes responder con 'OK' para confirmar que funcionas?")
                print(f"   Respuesta: {response.text}")
                print("   OK - gemini-pro funciona correctamente")
                return True
                
            except Exception as e3:
                print(f"   ERROR con gemini-pro: {e3}")
                return False

def test_gemini_forecast():
    """Probar pronóstico con datos de ejemplo"""
    print("\n=== PRUEBA DE PRONOSTICO CON GEMINI AI ===")
    
    try:
        # Crear datos de ejemplo
        data = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=12, freq='M'),
            'total_transactions': [1000, 1200, 1100, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100]
        })
        
        # Configurar modelo
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Crear prompt de prueba
        prompt = f"""
        Eres un experto analista de blockchain. Analiza estos datos y genera un pronóstico breve:
        
        DATOS:
        {data.to_string()}
        
        MÉTRICA: total_transactions
        AGRUPACIÓN: monthly
        
        Genera un pronóstico de 2-3 párrafos en español.
        """
        
        print("Enviando prompt de prueba a Gemini...")
        response = model.generate_content(prompt)
        
        print("RESPUESTA DE GEMINI:")
        print("-" * 50)
        print(response.text)
        print("-" * 50)
        print("OK - Pronóstico generado exitosamente")
        return True
        
    except Exception as e:
        print(f"ERROR generando pronóstico: {e}")
        return False

def main():
    """Función principal de prueba"""
    print("INICIANDO PRUEBAS DE GEMINI AI")
    print("=" * 50)
    
    # Prueba 1: Conexión básica
    connection_ok = test_gemini_connection()
    
    if connection_ok:
        # Prueba 2: Pronóstico
        forecast_ok = test_gemini_forecast()
        
        if forecast_ok:
            print("\n" + "=" * 50)
            print("RESULTADO: TODAS LAS PRUEBAS EXITOSAS")
            print("Gemini AI está funcionando correctamente")
            print("=" * 50)
        else:
            print("\n" + "=" * 50)
            print("RESULTADO: ERROR EN PRONOSTICO")
            print("La conexión funciona pero hay problemas con el pronóstico")
            print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("RESULTADO: ERROR DE CONEXION")
        print("No se pudo conectar con Gemini AI")
        print("Verifica tu API key y conexión a internet")
        print("=" * 50)

if __name__ == "__main__":
    main()
