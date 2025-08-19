import org.apache.spark.ml.classification.GBTClassifier
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator


val dataPath = "hdfs://hadoop-master:9000/AppsBigData2/survey.csv"

val rawData = spark.read.option("header", "true").option("inferSchema", "true").csv(dataPath)

val selectedColumns = Seq("Age", "Gender", "self_employed", "family_history", "work_interfere", "no_employees", "remote_work", "tech_company", "benefits", "seek_help", "anonymity", "leave", "mental_health_consequence", "phys_health_consequence", "treatment")

val data = rawData.select(selectedColumns.map(c => rawData(c)): _*)

val validData = data.filter("Age >= 0 AND Age <= 120")

val avgAge = validData.select("Age").na.drop().agg("Age" -> "avg").first().getDouble(0)

val categoricalCols = selectedColumns.filterNot(col => col == "Age" || col == "treatment")

val dataFilled = data.na.fill("Unknown", categoricalCols).na.fill(avgAge, Seq("Age"))

val indexers = categoricalCols.map(colName => new StringIndexer().setInputCol(colName).setOutputCol(s"${colName}_index").fit(dataFilled))

val dataIndexed = indexers.foldLeft(dataFilled)((df, indexer) => indexer.transform(df))

val featureCols = dataIndexed.columns.filter(col => col.endsWith("_index") || col == "Age")

val assembler = new VectorAssembler().setInputCols(featureCols.toArray).setOutputCol("features")

val dataAssembled = assembler.transform(dataIndexed)

val labelIndexer = new StringIndexer().setInputCol("treatment").setOutputCol("label")

val finalData = labelIndexer.fit(dataAssembled).transform(dataAssembled)

val Array(trainingData, testData) = finalData.randomSplit(Array(0.7, 0.3), seed = 123)

val gbt = new GBTClassifier().setLabelCol("label").setFeaturesCol("features").setMaxIter(15).setMaxBins(128)

val smallData = trainingData.sample(false, 0.2, 42L)

val gbtModel = gbt.fit(smallData)

val modelPath = "hdfs://hadoop-master:9000/AppsBigData2/gbt_model3"

gbtModel.write.overwrite().save(modelPath)

val predictions = gbtModel.transform(testData)

val evaluator = new BinaryClassificationEvaluator().setLabelCol("label").setRawPredictionCol("rawPrediction")

val auc = evaluator.evaluate(predictions)

println(s"Area Under ROC: $auc")

