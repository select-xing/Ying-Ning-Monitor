-- MySQL dump 10.13  Distrib 8.0.37, for Win64 (x86_64)
--
-- Host: 127.0.0.1    Database: wlwdatabase
-- ------------------------------------------------------
-- Server version	8.0.37

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `comdatatable`
--

DROP TABLE IF EXISTS `comdatatable`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `comdatatable` (
  `id` int NOT NULL AUTO_INCREMENT,
  `user` varchar(25) DEFAULT NULL,
  `text` varchar(1000) DEFAULT NULL,
  `time` datetime DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=4 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `comdatatable`
--

LOCK TABLES `comdatatable` WRITE;
/*!40000 ALTER TABLE `comdatatable` DISABLE KEYS */;
INSERT INTO `comdatatable` VALUES (1,'newlife','今天天气好','2025-05-17 18:20:59'),(2,'newlife','适合去散步','2025-05-17 18:21:40'),(3,'hello','今天天气好\n','2025-05-19 02:32:00');
/*!40000 ALTER TABLE `comdatatable` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `datatable`
--

DROP TABLE IF EXISTS `datatable`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `datatable` (
  `switch` int DEFAULT NULL,
  `humidity` int DEFAULT NULL,
  `body_temperature` int DEFAULT NULL,
  `respiratory_rate` int DEFAULT NULL,
  `heart_rate` int DEFAULT NULL,
  `cry_detection` int DEFAULT NULL,
  `smoke_detection` int DEFAULT NULL,
  `external_temperature` int DEFAULT NULL,
  `lng2` int DEFAULT NULL,
  `lat2` int DEFAULT NULL,
  `id` int DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `datatable`
--

LOCK TABLES `datatable` WRITE;
/*!40000 ALTER TABLE `datatable` DISABLE KEYS */;
/*!40000 ALTER TABLE `datatable` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `sleeptable`
--

DROP TABLE IF EXISTS `sleeptable`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `sleeptable` (
  `id` int NOT NULL AUTO_INCREMENT,
  `averageRespiration` int DEFAULT NULL,
  `averageHeartbeat` int DEFAULT NULL,
  `turnoverNumber` int DEFAULT NULL,
  `largeBodyMove` int DEFAULT NULL,
  `minorBodyMove` int DEFAULT NULL,
  `apneaEvents` int DEFAULT NULL,
  `eWakeDuration` int DEFAULT NULL,
  `eDeepSleepDuration` int DEFAULT NULL,
  `eShallowSleepDuration` int DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `sleeptable`
--

LOCK TABLES `sleeptable` WRITE;
/*!40000 ALTER TABLE `sleeptable` DISABLE KEYS */;
/*!40000 ALTER TABLE `sleeptable` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `usertable`
--

DROP TABLE IF EXISTS `usertable`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `usertable` (
  `id` int NOT NULL AUTO_INCREMENT,
  `username` varchar(20) DEFAULT NULL,
  `password` varchar(20) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=3 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `usertable`
--

LOCK TABLES `usertable` WRITE;
/*!40000 ALTER TABLE `usertable` DISABLE KEYS */;
INSERT INTO `usertable` VALUES (1,'newlife','123456'),(2,'hello','123456');
/*!40000 ALTER TABLE `usertable` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2025-05-20 13:28:21
