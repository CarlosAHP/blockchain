# 🚀 Blockchain Analytics Dashboard con IA

## 📋 Descripción

Aplicación web interactiva desarrollada con **Plotly Dash** que proporciona análisis avanzado de datos blockchain con capacidades de **Inteligencia Artificial** para predicciones en tiempo real.

## ✨ Características Principales

### 🔗 **Conexión a Datos**
- **MySQL**: Conexión directa a base de datos `blockchain_analytics`
- **CSV**: Respaldo automático con archivos CSV
- **Datos masivos**: Soporte para 100,000+ transacciones

### 🤖 **Modelos de IA Integrados**
- **Regresión Lineal**: Predicciones rápidas y eficientes
- **Ridge Regression**: Manejo de multicolinealidad
- **Random Forest**: Predicciones robustas y precisas
- **Características temporales**: Lag features, medias móviles, tendencias

### 📊 **Visualizaciones Interactivas**
- **Gráficos temporales**: Análisis histórico con predicciones
- **Gráficos de dispersión**: Correlaciones entre métricas
- **Zonas de confianza**: Intervalos de predicción
- **Actualización en tiempo real**: Cambios dinámicos

### 🎛️ **Controles Avanzados**
- **Selección de métricas**: 6 métricas diferentes
- **Modelos de IA**: 3 algoritmos disponibles
- **Parámetros ajustables**: Días de predicción y lookback
- **Entrenamiento dinámico**: Modelos que se actualizan en tiempo real

## 🛠️ Instalación y Configuración

### **Paso 1: Instalar Dependencias**
```bash
# Instalación automática
python install_dependencies.py

# O instalación manual
pip install -r requirements.txt
```

### **Paso 2: Configurar Base de Datos (Opcional)**
```sql
-- Verificar que MySQL esté ejecutándose
-- Base de datos: blockchain_analytics
-- Usuario: blockchainuser
-- Contraseña: 1234
```

### **Paso 3: Ejecutar Aplicación**
```bash
python app.py
```

### **Paso 4: Acceder al Dashboard**
- **URL**: http://localhost:8050
- **Navegador**: Cualquier navegador moderno

## 📈 Métricas Disponibles

### **📊 Métricas Principales**
1. **Transacciones Totales**: Número de transacciones por día
2. **Volumen Total**: Valor total de transacciones
3. **Precio Promedio de Gas**: Costo promedio de gas
4. **Gas Total Usado**: Consumo total de gas
5. **Direcciones Únicas**: Número de direcciones activas
6. **Nuevos Contratos**: Contratos creados diariamente

### **🤖 Capacidades de IA**
- **Predicciones temporales**: Hasta 90 días en el futuro
- **Análisis de tendencias**: Identificación de patrones
- **Detección de anomalías**: Alertas automáticas
- **Correlaciones**: Relaciones entre métricas

## 🎯 Uso de la Aplicación

### **1. Seleccionar Métrica**
- Usa el dropdown "Seleccionar Métrica"
- Elige entre 6 métricas disponibles
- Los gráficos se actualizan automáticamente

### **2. Configurar Modelo de IA**
- Selecciona algoritmo: Linear, Ridge, Random Forest
- Ajusta días de predicción (7-90 días)
- Configura días de lookback (3-30 días)

### **3. Entrenar Modelo**
- Haz clic en "🔄 Entrenar Modelo"
- El sistema procesará los datos históricos
- Las métricas de precisión se mostrarán

### **4. Ver Predicciones**
- Las predicciones aparecen en línea naranja
- Zona de confianza en área sombreada
- Datos históricos en línea azul

## 🔧 Configuración Avanzada

### **Parámetros del Modelo**
```python
# En app.py, puedes ajustar:
lookback_days = 7        # Días de datos históricos
prediction_days = 30     # Días de predicción
model_type = 'linear'    # Tipo de modelo
```

### **Características de IA**
- **Lag Features**: Valores de días anteriores
- **Medias Móviles**: Promedios de 3, 7, 14 días
- **Tendencias**: Diferencias entre períodos
- **Características temporales**: Día, mes, año

## 📊 Estructura de Datos

### **Tabla: daily_metrics**
```sql
date                DATE
total_transactions  INT
total_volume        DECIMAL
avg_gas_price       DECIMAL
total_gas_used      INT
unique_addresses    INT
new_contracts       INT
```

### **Tabla: transactions**
```sql
date              DATETIME
value             DECIMAL
gas_price         DECIMAL
gas_used          INT
transaction_type  VARCHAR
from_address      VARCHAR
to_address        VARCHAR
```

## 🚨 Solución de Problemas

### **Error de Conexión a MySQL**
```bash
# Verificar que MySQL esté ejecutándose
mysql -u blockchainuser -p1234 -h localhost blockchain_analytics

# La aplicación usará CSV como respaldo automáticamente
```

### **Dependencias Faltantes**
```bash
# Reinstalar dependencias
pip install --upgrade -r requirements.txt
```

### **Puerto Ocupado**
```bash
# Cambiar puerto en app.py
app.run_server(debug=True, host='0.0.0.0', port=8051)
```

## 📈 Casos de Uso

### **1. Análisis Ejecutivo**
- Dashboard de KPIs principales
- Tendencias a largo plazo
- Predicciones estratégicas

### **2. Análisis Operativo**
- Monitoreo en tiempo real
- Detección de anomalías
- Alertas automáticas

### **3. Análisis Técnico**
- Correlaciones entre métricas
- Análisis de gas y volumen
- Optimización de transacciones

## 🔮 Funcionalidades Futuras

- **Modelos de Deep Learning**: LSTM, GRU
- **Análisis de sentimiento**: Twitter, Reddit
- **Alertas por email**: Notificaciones automáticas
- **Exportación de datos**: PDF, Excel, PNG
- **API REST**: Integración con otros sistemas

## 📞 Soporte

Para problemas o preguntas:
1. Revisar logs de la aplicación
2. Verificar conexión a base de datos
3. Comprobar dependencias instaladas
4. Consultar documentación de Plotly Dash

---

**🎉 ¡Tu Dashboard de Blockchain Analytics con IA está listo para usar!**

**Dataset**: 100,000+ transacciones | **IA**: 3 modelos | **Predicciones**: Hasta 90 días


