package movie.recommend.engine

import java.util.Properties
import java.util.concurrent.TimeUnit

import org.apache.log4j.{Level, LogManager}
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.io.Source

object Model {
  private val propsReader = Source.fromURL(getClass.getResource("/engine.properties")).bufferedReader()
  private val config = new Properties()
  config.load(propsReader)
  private val log = LogManager.getRootLogger
  log.setLevel(Level.WARN)
  private val ratingSourcePath = {
    val dataSource = config.getProperty("config.spark.source.url")
    if (dataSource == null || dataSource.isEmpty) {
      log.error("Could not load data for training. Data source parameter was missed.")
      throw new IllegalArgumentException("Data source parameter is missed!")
    } else {
      dataSource
    }
  }

  private val sparkConf = {
    val master = config.getProperty("config.spark.master", "local[*]")
    new SparkConf()
      .setMaster(master)
      .setAppName("Movie recommendation")
  }
  val sc = new SparkContext(sparkConf)

  private var matrixFactorizationModel: MatrixFactorizationModel = {
    val m = trainModel()
    runTrainingScheduler()
    m
  }

  private[engine] def model: MatrixFactorizationModel = {
    matrixFactorizationModel
  }

  private def trainModel(): MatrixFactorizationModel = {
    val rawData = sc.textFile(ratingSourcePath)
    val del = config.getProperty("config.data.delimiter")
    val data = rawData
      .map(_.split(del))
      .map(arr => (Rating(arr(0).toInt, arr(1).toInt, arr(2).toDouble), arr(3).toLong))
    val trainingSet = data
      .filter(_._2 % 10 < 6)
      .keys
      .cache()
    val validationSet = data
      .filter(x => x._2 % 10 > 6 && x._2 % 10 < 8)
      .keys
      .cache()
    val testSet = data
      .filter(_._2 % 10 > 8)
      .keys
      .cache()

    val countTraining = trainingSet.count()
    val countValidation = validationSet.count()

    val ranks = 1 to 10
    val lambdas = 0.1 to 1 by 0.1
    val numIters = 10 to 20
    var bestModel: Option[MatrixFactorizationModel] = None
    var bestValidationRmse = Double.MaxValue
    var bestRank = 0
    var bestLambda = -1.0
    var bestNumIter = -1
    for (rank <- ranks; lambda <- lambdas; numIter <- numIters) {
      val model = ALS.train(trainingSet, rank, numIter, lambda)
      val validationRmse = computeRMSE(model, validationSet, countValidation)
      if (validationRmse < bestValidationRmse) {
        bestModel = Some(model)
        bestValidationRmse = validationRmse
        bestRank = rank
        bestLambda = lambda
        bestNumIter = numIter
      }
    }

    val testRmse = computeRMSE(bestModel.get, testSet, countTraining)
    log.warn(
      s"""
        The best model was trained with rank = $bestRank and lambda $bestLambda,
        and numIter = $bestNumIter, and its RMSE on the test set is $testRmse ".
      """
    )
    bestModel.get
  }

  private def runTrainingScheduler(): Unit = {
    val period = config.getProperty("config.model.training_period").toLong
    val trainScheduler = new Thread(new Runnable {
      override def run(): Unit = {
        while (true) {
          TimeUnit.HOURS.sleep(period)
          matrixFactorizationModel = trainModel()
        }
      }
    })
    trainScheduler.setDaemon(true)
    trainScheduler.start()
    log.info("Training scheduler was launched with period " + period)
  }

  private def computeRMSE(model: MatrixFactorizationModel, data: RDD[Rating], n: Long): Double = {
    val predictions: RDD[Rating] = model.predict(data.map(x => (x.user, x.product)))
    val predictionsAndRatings = predictions.map(x => ((x.user, x.product), x.rating))
      .join(data.map(x => ((x.user, x.product), x.rating)))
      .values
    math.sqrt(predictionsAndRatings.map(x => (x._1 - x._2) * (x._1 - x._2)).reduce(_ + _) / n)
  }
}
