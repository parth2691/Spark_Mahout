package com.rp.mahout

import org.apache.mahout.math._
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.math.drm._
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.math.drm.RLikeDrmOps._
import org.apache.mahout.sparkbindings._

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf

object TestClusterAssign {

  def main(args1: Array[String]): Unit = {
    val conf = new SparkConf()
      .setAppName("Simple Application")
      .setMaster("local[*]")
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .set("spark.kryo.registrator", "org.apache.mahout.sparkbindings.io.MahoutKryoRegistrator")
    implicit val sc = new SparkDistributedContext(new SparkContext(conf))
    val inCoreA = dense((1,1, 2, 3), (1,2, 3, 4), (1,3, 4, 5), (1,4, 5, 6))
    val A = drmParallelize(m = inCoreA)
    val drm2 = A.mapBlock() {
      case (keys, block) =>
        for(row <- 0 until keys.size) {
          keys(row) = 1
        }
        (keys, block)
    }
    val aggTranspose = drm2.t
    println("Result of aggregating tranpose")
    println(""+aggTranspose.collect)
  }

}
