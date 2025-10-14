-- Script para configurar MySQL para el proyecto Blockchain Analytics
-- Ejecutar como usuario root

-- Crear usuario blockchainuser
CREATE USER IF NOT EXISTS 'blockchainuser'@'localhost' IDENTIFIED BY '1234';

-- Crear base de datos
CREATE DATABASE IF NOT EXISTS blockchain_analytics 
CHARACTER SET utf8mb4 
COLLATE utf8mb4_unicode_ci;

-- Otorgar permisos
GRANT ALL PRIVILEGES ON blockchain_analytics.* TO 'blockchainuser'@'localhost';

-- Aplicar cambios
FLUSH PRIVILEGES;

-- Mostrar confirmación
SELECT 'Base de datos y usuario configurados exitosamente' as Status;
SHOW DATABASES LIKE 'blockchain_analytics';
SELECT User, Host FROM mysql.user WHERE User = 'blockchainuser';
