import org.apache.spark.ml.classification.GBTClassificationModel
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.SparkSession
val modelPath = "hdfs://hadoop-master:9000/AppsBigData2/gbt_model3"
val gbtModel = GBTClassificationModel.load(modelPath)
val newEmployeeData = spark.createDataFrame(Seq(
  (1, 28, "Male", "No", "No", "Often", "6-25", "Yes", "Yes", "Yes", "No", "Somewhat easy", "No", "No"),
  (2, 34, "Female", "Yes", "Yes", "Rarely", "26-100", "No", "No", "Don't know", "Don't know", "Somewhat difficult", "Maybe", "Yes"))).toDF("ID", "Age", "Gender", "self_employed", "family_history", "work_interfere", "no_employees", "remote_work", "tech_company", "benefits", "seek_help", "anonymity", "leave", "mental_health_consequence")
val categoricalCols = Seq("Gender", "self_employed", "family_history", "work_interfere", "no_employees", "remote_work", "tech_company", "benefits", "seek_help", "anonymity", "leave", "mental_health_consequence")
val avgAge = 31 
val newDataFilled = newEmployeeData.na.fill("Unknown", categoricalCols).na.fill(avgAge, Seq("Age"))
val indexers = categoricalCols.map(colName => new StringIndexer().setInputCol(colName).setOutputCol(s"${colName}_index").fit(newDataFilled))
val newDataIndexed = indexers.foldLeft(newDataFilled)((df, indexer) => indexer.transform(df))
val featureCols = categoricalCols.map(_ + "_index") :+ "Age"
val assembler = new VectorAssembler().setInputCols(featureCols.toArray).setOutputCol("features")
val newDataAssembled = assembler.transform(newDataIndexed)
val predictions = gbtModel.transform(newDataAssembled)
val results = predictions.select("ID", "Age", "Gender", "features", "prediction")
results.show()
