from pyspark.sql import SparkSession

# Créer une session Spark
spark = SparkSession.builder \
    .appName("ETL_SmartBuilding") \
    .getOrCreate()

# Lire le CSV
df = spark.read.csv("smart_building_ETL_ready.csv", header=True, inferSchema=True)

# Afficher un aperçu
df.show(5)
df.printSchema()
