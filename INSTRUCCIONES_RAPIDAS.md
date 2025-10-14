# 🚀 Blockchain Analytics Dashboard - Instrucciones Rápidas

## ⚡ Inicio Rápido

### **Opción 1: Inicio Automático (Recomendado)**
```bash
python start_dashboard.py
```

### **Opción 2: Inicio Manual**
```bash
python app.py
```

### **Opción 3: Con Instalación Automática**
```bash
python install_dependencies.py
python app.py
```

## 🌐 Acceso al Dashboard

Una vez iniciado, abre tu navegador en:
**http://localhost:8050**

## 📊 Características del Dashboard

### **🎛️ Controles Disponibles:**
- **Seleccionar Métrica**: 6 métricas diferentes
- **Modelo de IA**: Linear, Ridge, Random Forest
- **Días de Predicción**: 7-90 días
- **Días de Lookback**: 3-30 días

### **📈 Métricas Disponibles:**
1. **Transacciones Totales**
2. **Volumen Total**
3. **Precio Promedio de Gas**
4. **Gas Total Usado**
5. **Direcciones Únicas**
6. **Nuevos Contratos**

### **🤖 Modelos de IA:**
- **Regresión Lineal**: Rápido y eficiente
- **Ridge Regression**: Manejo de multicolinealidad
- **Random Forest**: Predicciones robustas

## 🔧 Uso del Dashboard

### **1. Seleccionar Métrica**
- Usa el dropdown "Seleccionar Métrica"
- Elige la métrica que quieres analizar

### **2. Configurar Modelo**
- Selecciona el algoritmo de IA
- Ajusta los días de predicción
- Configura los días de lookback

### **3. Entrenar Modelo**
- Haz clic en "🔄 Entrenar Modelo"
- Espera a que se procesen los datos
- Ve las métricas de precisión

### **4. Ver Predicciones**
- Las predicciones aparecen en línea naranja
- Zona de confianza en área sombreada
- Datos históricos en línea azul

## 📊 Datos Disponibles

### **Fuente de Datos:**
- **MySQL**: Conexión directa a `blockchain_analytics`
- **CSV**: Respaldo automático con archivos CSV
- **Registros**: 4,167 días de datos históricos

### **Estructura de Datos:**
- **Período**: 2014-2025 (11 años)
- **Transacciones**: 100,000+ registros
- **Volumen**: ~600 millones
- **Contratos**: 500 contratos inteligentes

## 🚨 Solución de Problemas

### **Error de Dependencias**
```bash
pip install -r requirements.txt
```

### **Puerto Ocupado**
```bash
# Cambiar puerto en app.py línea final
app.run_server(debug=True, host='0.0.0.0', port=8051)
```

### **Error de MySQL**
- La aplicación usa CSV automáticamente
- No es necesario configurar MySQL

### **Navegador No Abre**
- Abre manualmente: http://localhost:8050
- Verifica que el puerto 8050 esté libre

## 📈 Casos de Uso

### **Análisis Ejecutivo**
- Dashboard de KPIs principales
- Tendencias a largo plazo
- Predicciones estratégicas

### **Análisis Operativo**
- Monitoreo en tiempo real
- Detección de anomalías
- Alertas automáticas

### **Análisis Técnico**
- Correlaciones entre métricas
- Análisis de gas y volumen
- Optimización de transacciones

## 🔮 Funcionalidades Avanzadas

### **Predicciones Temporales**
- Hasta 90 días en el futuro
- Múltiples algoritmos de IA
- Zonas de confianza

### **Análisis de Correlaciones**
- Gráficos de dispersión
- Relaciones entre métricas
- Identificación de patrones

### **Interactividad**
- Controles dinámicos
- Actualización en tiempo real
- Filtros cruzados

## 📞 Soporte

### **Archivos Importantes:**
- `app.py` - Aplicación principal
- `config.py` - Configuración de base de datos
- `requirements.txt` - Dependencias
- `start_dashboard.py` - Inicio automático

### **Logs y Debugging:**
- La aplicación muestra logs en consola
- Errores se muestran en tiempo real
- Modo debug activado por defecto

---

**🎉 ¡Tu Dashboard de Blockchain Analytics con IA está listo!**

**Dataset**: 100,000+ transacciones | **IA**: 3 modelos | **Predicciones**: Hasta 90 días


