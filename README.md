# 🚀 Blockchain Analytics Platform

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![MySQL](https://img.shields.io/badge/MySQL-8.0+-orange?style=for-the-badge&logo=mysql)
![PowerBI](https://img.shields.io/badge/Power%20BI-Desktop-yellow?style=for-the-badge&logo=powerbi)
![AI](https://img.shields.io/badge/AI-Machine%20Learning-green?style=for-the-badge&logo=tensorflow)

**Plataforma completa de análisis de datos blockchain con capacidades de IA integradas**

[![GitHub stars](https://img.shields.io/github/stars/CarlosAHP/blockchain?style=social)](https://github.com/CarlosAHP/blockchain)
[![GitHub forks](https://img.shields.io/github/forks/CarlosAHP/blockchain?style=social)](https://github.com/CarlosAHP/blockchain)

</div>

---

## 🎯 ¿Qué es este proyecto?

Esta plataforma te permite **analizar datos de blockchain de manera inteligente** utilizando:

- 🔗 **Extracción de datos** de redes blockchain (Corda testnet)
- 🧠 **Inteligencia Artificial** para detectar anomalías y hacer predicciones
- 📊 **Visualizaciones interactivas** con Plotly Dash
- 🗄️ **Almacenamiento optimizado** en MySQL
- 📈 **Dashboards profesionales** con Power BI

## ✨ Características Principales

### 🤖 Inteligencia Artificial Integrada
- **Motor de análisis avanzado** con múltiples algoritmos de IA
- **Predicciones inteligentes** con Regresión Lineal, Ridge y Random Forest
- **Análisis de patrones** temporales automáticos
- **Pronósticos extensos** generados por IA avanzada
- **Detección de anomalías** con algoritmos especializados

### 📊 Visualizaciones Avanzadas
- **Dashboards interactivos** en tiempo real
- **Gráficos dinámicos** con Plotly
- **Filtros inteligentes** por fecha, tipo, volumen
- **Exportación** a múltiples formatos

### 🔄 Pipeline Automatizado
- **Extracción automática** de datos blockchain
- **Procesamiento inteligente** con Pandas
- **Carga optimizada** a MySQL
- **Análisis continuo** con IA

## 🚀 Inicio Rápido

### 1️⃣ Clonar el Repositorio
```bash
git clone https://github.com/CarlosAHP/blockchain.git
cd blockchain
```

### 2️⃣ Instalar Dependencias
```bash
pip install -r requirements.txt
```

### 3️⃣ Configurar Base de Datos
```bash
# Crear usuario y base de datos en MySQL
mysql -u root -p
```

```sql
CREATE USER 'blockchainuser'@'localhost' IDENTIFIED BY '1234';
CREATE DATABASE blockchain_analytics CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
GRANT ALL PRIVILEGES ON blockchain_analytics.* TO 'blockchainuser'@'localhost';
FLUSH PRIVILEGES;
```

### 4️⃣ Ejecutar la Aplicación
```bash
# Opción 1: Aplicación web completa con IA (recomendado)
python app_final.py

# Opción 2: Pipeline completo
python main.py

# Opción 3: Aplicación web simple
python app_simple.py
```

## 🎮 Cómo Usar la Aplicación

### 🌐 Aplicación Web Interactiva
1. Ejecuta `python app_final.py`
2. Abre tu navegador en `http://localhost:8050`
3. Explora los dashboards interactivos con IA
4. Usa los filtros para análisis específicos
5. Genera pronósticos inteligentes con IA

### 📊 Dashboards Disponibles

#### 📈 Dashboard Principal
- **Métricas en tiempo real**: Transacciones, volumen, gas
- **Análisis temporal**: Tendencias por día/semana/mes
- **Top performers**: Direcciones y contratos más activos
- **Alertas de IA**: Anomalías y predicciones

#### 🔍 Dashboard de Análisis
- **Distribución de transacciones**: Por tipo y valor
- **Análisis de gas**: Utilización y precios
- **Métricas de red**: Salud y rendimiento
- **Insights de IA**: Patrones y predicciones

### 🤖 Funciones de IA

#### Detección de Anomalías
```python
# El sistema automáticamente detecta:
- Días con actividad inusual
- Picos en transacciones o volumen
- Cambios en patrones de gas
- Comportamientos sospechosos
```

#### Predicciones Inteligentes
```python
# Predice automáticamente:
- Volumen de transacciones futuras
- Precios de gas
- Actividad de contratos
- Tendencias de red
```

## 🏗️ Arquitectura del Sistema

```mermaid
graph TB
    A[Corda Testnet] --> B[Python Extractor]
    B --> C[Data Processor]
    C --> D[MySQL Database]
    D --> E[AI Analytics]
    E --> F[Plotly Dash App]
    D --> G[Power BI]
    
    H[Web Interface] --> F
    I[Real-time Updates] --> F
    J[Export Features] --> F
```

## 📁 Estructura del Proyecto

```
blockchain/
├── 🚀 main.py                    # Pipeline principal
├── 🌐 app_final.py               # Aplicación web completa con IA
├── 🎯 app_simple.py             # Aplicación web simple
├── 🤖 ai_diagnostic.py          # Motor de análisis de IA avanzada
├── ⚙️ config.py                 # Configuración
├── 🗄️ database_setup.py         # Setup de MySQL
├── 📊 data_processor.py          # Procesamiento de datos
├── 🤖 ai_analytics.py           # Análisis de IA
├── 📈 create_dashboards.py      # Generador de dashboards
├── 📋 requirements.txt          # Dependencias
└── 📁 data/                      # Datos del proyecto
    ├── raw/                      # Datos originales
    ├── processed/                # Datos procesados
    └── csv/                      # Archivos CSV
```

## 🔧 Comandos Útiles

### Ejecutar Componentes Individuales
```bash
# Solo extracción de datos
python corda_data_extractor.py

# Solo procesamiento
python data_processor.py

# Solo análisis de IA
python ai_analytics.py

# Motor de análisis avanzado
python ai_diagnostic.py

# Solo dashboards
python create_dashboards.py
```

### Configuración de Base de Datos
```bash
# Configurar MySQL
python database_setup.py

# Cargar datos a MySQL
python mysql_loader.py
```

## 📊 Métricas y KPIs

### 📈 Métricas de Transacciones
- **Volumen total diario**
- **Número de transacciones**
- **Precio promedio de gas**
- **Tiempo de confirmación**

### 🌐 Métricas de Red
- **Utilización de gas**
- **Tamaño promedio de bloques**
- **Número de direcciones activas**
- **Nuevos contratos desplegados**

### 🤖 Métricas de IA
- **Motor de análisis avanzado** con 10 secciones de diagnóstico
- **Precisión de predicciones** con múltiples algoritmos
- **Confianza en insights** generados automáticamente
- **Pronósticos extensos** con análisis detallado
- **Detección de anomalías** con clasificación por severidad

## 🔧 Configuración Avanzada

### Variables de Entorno
```bash
# Crear archivo .env
DB_HOST=localhost
DB_USER=blockchainuser
DB_PASSWORD=1234
DB_NAME=blockchain_analytics
DB_PORT=3306
```

### Configuración de MySQL
```sql
-- Optimizar para análisis
SET GLOBAL innodb_buffer_pool_size = 1G;
SET GLOBAL query_cache_size = 256M;
SET GLOBAL max_connections = 200;
```

## 🚨 Solución de Problemas

### ❌ Error de Conexión a MySQL
```bash
# Verificar que MySQL esté ejecutándose
sudo systemctl status mysql

# Verificar credenciales
mysql -u blockchainuser -p1234 -h localhost blockchain_analytics
```

### ❌ Error de Dependencias
```bash
# Reinstalar dependencias
pip install --upgrade -r requirements.txt

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows
```

### ❌ Error de Permisos
```bash
# Crear directorios manualmente
mkdir -p data/raw data/processed data/csv
chmod 755 data/raw data/processed data/csv
```

## 🤖 Motor de Análisis de IA Avanzada

### 📊 Diagnósticos Completos
El motor de IA genera reportes extensos con **10 secciones principales**:

1. **📊 Metadatos del Análisis** - Información técnica del proceso
2. **📋 Resumen Ejecutivo** - Hallazgos clave y acciones inmediatas
3. **📈 Análisis de Mercado** - Tendencias y patrones de volumen
4. **📊 Análisis Técnico** - Indicadores RSI, MACD, medias móviles
5. **⚠️ Evaluación de Riesgos** - Factores de riesgo y mitigación
6. **🔮 Predicciones** - Escenarios optimista, base y pesimista
7. **🔍 Detección de Anomalías** - Patrones inusuales detectados
8. **🔗 Análisis de Correlaciones** - Relaciones entre variables
9. **💡 Recomendaciones** - Estrategias inmediatas y a largo plazo
10. **📝 Conclusiones** - Próximos pasos y métricas de éxito

### 🎯 Algoritmos de IA Disponibles
- **📈 Regresión Lineal**: Rápido y eficiente para tendencias lineales
- **🛡️ Ridge Regression**: Maneja multicolinealidad y evita sobreajuste
- **🌲 Random Forest**: Captura relaciones complejas y no lineales

### 🚀 Uso del Motor de IA
```python
# Crear motor de análisis
engine = AIDiagnosticEngine()

# Ejecutar análisis completo
diagnostic = engine.generate_comprehensive_diagnostic(blockchain_data)

# Generar reporte extenso
report = engine.format_diagnostic_report(diagnostic)
```

## 🎯 Casos de Uso

### 🔍 Para Analistas de Datos
- **Análisis de tendencias** blockchain con IA avanzada
- **Detección de patrones** anómalos automática
- **Predicciones de mercado** con múltiples escenarios
- **Reportes automatizados** extensos y detallados

### 🏢 Para Empresas
- **Monitoreo de transacciones**
- **Análisis de riesgo**
- **Optimización de gas**
- **Compliance y auditoría**

### 👨‍💻 Para Desarrolladores
- **API de análisis** blockchain
- **Integración con sistemas** existentes
- **Automatización de procesos**
- **Desarrollo de dashboards**

## 🚀 Próximas Mejoras

- [ ] 🔗 **Integración con más blockchains** (Ethereum, Bitcoin)
- [ ] 🐳 **Dockerización** completa del proyecto
- [ ] 🌐 **API REST** para consultas externas
- [ ] 📱 **Aplicación móvil** para monitoreo
- [ ] 🔔 **Alertas en tiempo real** por email/SMS
- [ ] 🧠 **Más algoritmos de ML** (LSTM, Random Forest)
- [ ] ☁️ **Despliegue en la nube** (AWS, Azure, GCP)

## 🤝 Contribuir

¡Las contribuciones son bienvenidas! Por favor:

1. 🍴 Fork el proyecto
2. 🌿 Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. 💾 Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. 📤 Push a la rama (`git push origin feature/AmazingFeature`)
5. 🔄 Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

## 👨‍💻 Autor

**Carlos AHP**
- GitHub: [@CarlosAHP](https://github.com/CarlosAHP)
- LinkedIn: [Carlos AHP](https://linkedin.com/in/carlosahp)

## 🙏 Agradecimientos

- **Corda Network** por la red testnet
- **Plotly** por las visualizaciones interactivas
- **scikit-learn** por las capacidades de ML
- **MySQL** por el almacenamiento robusto
- **Power BI** por los dashboards profesionales

---

<div align="center">

### ⭐ Si te gusta este proyecto, ¡dale una estrella! ⭐

[![GitHub stars](https://img.shields.io/github/stars/CarlosAHP/blockchain?style=social)](https://github.com/CarlosAHP/blockchain)

**Desarrollado con ❤️ para la comunidad blockchain**

</div>