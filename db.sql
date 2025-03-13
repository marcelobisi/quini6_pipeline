-- --------------------------------------------------------
-- Host:                         127.0.0.1
-- Versión del servidor:         11.7.2-MariaDB - mariadb.org binary distribution
-- SO del servidor:              Win64
-- HeidiSQL Versión:             12.10.0.7000
-- --------------------------------------------------------

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET NAMES utf8 */;
/*!50503 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

-- Volcando estructura para procedimiento quini6_predict.DeleteModelo
DELIMITER //
CREATE PROCEDURE `DeleteModelo`(
    IN modelo_id INT
)
BEGIN
    -- Eliminar evaluaciones asociadas al modelo
    DELETE FROM evaluaciones WHERE id_modelo = modelo_id;

    -- Eliminar predicciones asociadas al modelo
    DELETE FROM predicciones WHERE id_modelo = modelo_id;

    -- Eliminar el modelo
    DELETE FROM modelos WHERE id = modelo_id;
END//
DELIMITER ;

-- Volcando estructura para tabla quini6_predict.evaluaciones
CREATE TABLE IF NOT EXISTS `evaluaciones` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `sorteo` int(11) NOT NULL,
  `fecha` date NOT NULL,
  `modalidad` enum('TRADICIONAL','SEGUNDA','REVANCHA','SIEMPRE SALE') NOT NULL,
  `aciertos` int(11) NOT NULL,
  `match_6` tinyint(1) NOT NULL,
  `match_3` tinyint(1) NOT NULL,
  `id_modelo` int(11) DEFAULT NULL,
  `fecha_evaluacion` timestamp NULL DEFAULT current_timestamp(),
  PRIMARY KEY (`id`),
  KEY `idx_id_modelo` (`id_modelo`),
  KEY `idx_fecha_evaluacion` (`fecha_evaluacion`),
  CONSTRAINT `evaluaciones_ibfk_1` FOREIGN KEY (`id_modelo`) REFERENCES `modelos` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3 COLLATE=utf8mb3_general_ci;

-- La exportación de datos fue deseleccionada.

-- Volcando estructura para procedimiento quini6_predict.GetEvaluacionesPorModelo
DELIMITER //
CREATE PROCEDURE `GetEvaluacionesPorModelo`(
    IN modelo_nombre VARCHAR(100)
)
BEGIN
    SELECT e.*
    FROM evaluaciones e
    JOIN modelos m ON e.id_modelo = m.id
    WHERE m.nombre = modelo_nombre
    ORDER BY e.fecha_evaluacion, e.fecha, e.sorteo, e.modalidad;
END//
DELIMITER ;

-- Volcando estructura para procedimiento quini6_predict.GetPrediccionesPorModelo
DELIMITER //
CREATE PROCEDURE `GetPrediccionesPorModelo`(
    IN modelo_nombre VARCHAR(100)
)
BEGIN
    SELECT p.*
    FROM predicciones p
    JOIN modelos m ON p.id_modelo = m.id
    WHERE m.nombre = modelo_nombre
    ORDER BY p.fecha, p.sorteo, p.modalidad;
END//
DELIMITER ;

-- Volcando estructura para procedimiento quini6_predict.GetSorteosPorFecha
DELIMITER //
CREATE PROCEDURE `GetSorteosPorFecha`(
    IN fecha_inicio DATE,
    IN fecha_fin DATE
)
BEGIN
    SELECT * FROM sorteos WHERE fecha BETWEEN fecha_inicio AND fecha_fin ORDER BY fecha;
END//
DELIMITER ;

-- Volcando estructura para procedimiento quini6_predict.InsertModelo
DELIMITER //
CREATE PROCEDURE `InsertModelo`(
    IN p_nombre VARCHAR(100),
    IN p_descripcion TEXT,
    IN p_datos_modelo LONGBLOB,
    IN p_metricas JSON
)
BEGIN
    -- Validar que el nombre no esté vacío
    IF p_nombre IS NULL OR p_nombre = '' THEN
        SIGNAL SQLSTATE '45000'
        SET MESSAGE_TEXT = 'Error: El nombre del modelo no puede estar vacío.';
    END IF;

    -- Validar que datos_modelo no sea NULL
    IF p_datos_modelo IS NULL THEN
        SIGNAL SQLSTATE '45000'
        SET MESSAGE_TEXT = 'Error: Los datos del modelo no pueden ser NULL.';
    END IF;

    -- Verificar si ya existe un modelo con el mismo nombre
    IF EXISTS (SELECT 1 FROM modelos WHERE nombre = p_nombre) THEN
        SIGNAL SQLSTATE '45000'
        SET MESSAGE_TEXT = 'Error: Ya existe un modelo con el mismo nombre.';
    ELSE
        INSERT INTO modelos (nombre, descripcion, datos_modelo, metricas)
        VALUES (p_nombre, p_descripcion, p_datos_modelo, p_metricas);
    END IF;
END//
DELIMITER ;

-- Volcando estructura para procedimiento quini6_predict.InsertPrediccion
DELIMITER //
CREATE PROCEDURE `InsertPrediccion`(
    IN p_sorteo INT,
    IN p_fecha DATE,
    IN p_modalidad ENUM('TRADICIONAL', 'SEGUNDA', 'REVANCHA', 'SIEMPRE SALE'),
    IN p_n1 TINYINT,
    IN p_n2 TINYINT,
    IN p_n3 TINYINT,
    IN p_n4 TINYINT,
    IN p_n5 TINYINT,
    IN p_n6 TINYINT,
    IN p_prob1 FLOAT,
    IN p_prob2 FLOAT,
    IN p_prob3 FLOAT,
    IN p_prob4 FLOAT,
    IN p_prob5 FLOAT,
    IN p_prob6 FLOAT,
    IN p_prob_acumulada FLOAT,
    IN p_modelo_usado VARCHAR(100)
)
BEGIN
    DECLARE v_id_modelo INT;

    SELECT id INTO v_id_modelo FROM modelos WHERE nombre = p_modelo_usado LIMIT 1;

    IF v_id_modelo IS NULL THEN
        SIGNAL SQLSTATE '45000'
        SET MESSAGE_TEXT = 'Error: Modelo no encontrado con el nombre proporcionado.';
    ELSE
        INSERT INTO predicciones (sorteo, fecha, modalidad, n1, n2, n3, n4, n5, n6,
                                    prob1, prob2, prob3, prob4, prob5, prob6, prob_acumulada, id_modelo)
        VALUES (p_sorteo, p_fecha, p_modalidad, p_n1, p_n2, p_n3, p_n4, p_n5, p_n6,
                p_prob1, p_prob2, p_prob3, p_prob4, p_prob5, p_prob6, p_prob_acumulada, v_id_modelo);
    END IF;
END//
DELIMITER ;

-- Volcando estructura para tabla quini6_predict.logs
CREATE TABLE IF NOT EXISTS `logs` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `timestamp` datetime NOT NULL,
  `level` varchar(20) NOT NULL,
  `name` varchar(255) NOT NULL,
  `message` text NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=3 DEFAULT CHARSET=utf8mb3 COLLATE=utf8mb3_general_ci;

-- La exportación de datos fue deseleccionada.

-- Volcando estructura para tabla quini6_predict.modelos
CREATE TABLE IF NOT EXISTS `modelos` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `nombre` varchar(100) NOT NULL,
  `fecha_entrenamiento` timestamp NULL DEFAULT current_timestamp(),
  `descripcion` text DEFAULT NULL,
  `datos_modelo` longblob NOT NULL,
  `metricas` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT NULL CHECK (json_valid(`metricas`)),
  PRIMARY KEY (`id`),
  KEY `idx_nombre` (`nombre`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3 COLLATE=utf8mb3_general_ci;

-- La exportación de datos fue deseleccionada.

-- Volcando estructura para tabla quini6_predict.predicciones
CREATE TABLE IF NOT EXISTS `predicciones` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `sorteo` int(11) NOT NULL,
  `fecha` date NOT NULL,
  `modalidad` enum('TRADICIONAL','SEGUNDA','REVANCHA','SIEMPRE SALE') NOT NULL,
  `n1` tinyint(4) NOT NULL,
  `n2` tinyint(4) NOT NULL,
  `n3` tinyint(4) NOT NULL,
  `n4` tinyint(4) NOT NULL,
  `n5` tinyint(4) NOT NULL,
  `n6` tinyint(4) NOT NULL,
  `prob1` float NOT NULL,
  `prob2` float NOT NULL,
  `prob3` float NOT NULL,
  `prob4` float NOT NULL,
  `prob5` float NOT NULL,
  `prob6` float NOT NULL,
  `prob_acumulada` float NOT NULL,
  `id_modelo` int(11) DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `sorteo` (`sorteo`,`modalidad`),
  KEY `idx_id_modelo` (`id_modelo`),
  CONSTRAINT `predicciones_ibfk_1` FOREIGN KEY (`id_modelo`) REFERENCES `modelos` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3 COLLATE=utf8mb3_general_ci;

-- La exportación de datos fue deseleccionada.

-- Volcando estructura para tabla quini6_predict.scalers
CREATE TABLE IF NOT EXISTS `scalers` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `nombre` varchar(100) NOT NULL,
  `datos` text NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `nombre` (`nombre`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3 COLLATE=utf8mb3_general_ci;

-- La exportación de datos fue deseleccionada.

-- Volcando estructura para tabla quini6_predict.scalers_json
CREATE TABLE IF NOT EXISTS `scalers_json` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `nombre` varchar(100) NOT NULL,
  `datos` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL CHECK (json_valid(`datos`)),
  PRIMARY KEY (`id`),
  UNIQUE KEY `nombre` (`nombre`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3 COLLATE=utf8mb3_general_ci;

-- La exportación de datos fue deseleccionada.

-- Volcando estructura para tabla quini6_predict.sorteos
CREATE TABLE IF NOT EXISTS `sorteos` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `sorteo` int(11) NOT NULL,
  `fecha` date NOT NULL,
  `modalidad` enum('TRADICIONAL','SEGUNDA','REVANCHA','SIEMPRE SALE') NOT NULL,
  `n1` tinyint(4) NOT NULL,
  `n2` tinyint(4) NOT NULL,
  `n3` tinyint(4) NOT NULL,
  `n4` tinyint(4) NOT NULL,
  `n5` tinyint(4) NOT NULL,
  `n6` tinyint(4) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `sorteo` (`sorteo`,`modalidad`),
  KEY `idx_fecha` (`fecha`)
) ENGINE=InnoDB AUTO_INCREMENT=7825 DEFAULT CHARSET=utf8mb3 COLLATE=utf8mb3_general_ci;

-- La exportación de datos fue deseleccionada.

-- Volcando estructura para procedimiento quini6_predict.UpdateModeloMetricas
DELIMITER //
CREATE PROCEDURE `UpdateModeloMetricas`(
    IN modelo_id INT,
    IN p_metricas JSON
)
BEGIN
    UPDATE modelos
    SET metricas = p_metricas
    WHERE id = modelo_id;
END//
DELIMITER ;

-- Volcando estructura para disparador quini6_predict.before_insert_prediccion
SET @OLDTMP_SQL_MODE=@@SQL_MODE, SQL_MODE='STRICT_TRANS_TABLES,ERROR_FOR_DIVISION_BY_ZERO,NO_AUTO_CREATE_USER,NO_ENGINE_SUBSTITUTION';
DELIMITER //
CREATE TRIGGER before_insert_prediccion
BEFORE INSERT ON predicciones
FOR EACH ROW
BEGIN
    IF NEW.id_modelo IS NULL THEN
        SIGNAL SQLSTATE '45000'
        SET MESSAGE_TEXT = 'Error: id_modelo no puede ser NULL en predicciones';
    END IF;
END//
DELIMITER ;
SET SQL_MODE=@OLDTMP_SQL_MODE;

/*!40103 SET TIME_ZONE=IFNULL(@OLD_TIME_ZONE, 'system') */;
/*!40101 SET SQL_MODE=IFNULL(@OLD_SQL_MODE, '') */;
/*!40014 SET FOREIGN_KEY_CHECKS=IFNULL(@OLD_FOREIGN_KEY_CHECKS, 1) */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40111 SET SQL_NOTES=IFNULL(@OLD_SQL_NOTES, 1) */;
