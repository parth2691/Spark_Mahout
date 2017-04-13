package com.rp.mahout

import scala.collection.JavaConversions.iterableAsScalaIterable
import scala.collection.Seq
import scala.util.Random

import org.apache.mahout.math.Matrix
import org.apache.mahout.math.RandomAccessSparseVector
import org.apache.mahout.math.SparseRowMatrix
import org.apache.mahout.math.Vector
import org.apache.mahout.math.decompositions.DQR.dqrThin
import org.apache.mahout.math.drm.DistributedContext
import org.apache.mahout.math.drm.DrmLike
import org.apache.mahout.math.drm.RLikeDrmOps.drm2cpops
import org.apache.mahout.math.drm.RLikeDrmOps.drmInt2RLikeOps
import org.apache.mahout.math.drm.bcast2val
import org.apache.mahout.math.drm.drm2Checkpointed
import org.apache.mahout.math.drm.drm2InCore
import org.apache.mahout.math.drm.drmBroadcast
import org.apache.mahout.math.drm.drmSampleKRows
import org.apache.mahout.sparkbindings.drmWrap
import org.apache.mahout.math.scalabindings.::
import org.apache.mahout.math.scalabindings.RLikeOps.m2mOps
import org.apache.mahout.math.scalabindings.RLikeOps.v2vOps
import org.apache.mahout.math.scalabindings.diag
import org.apache.mahout.math.scalabindings.dvec
import org.apache.mahout.math.scalabindings.solve
import org.apache.mahout.math.scalabindings.svd
import org.apache.mahout.sparkbindings.DrmRdd
import org.apache.mahout.sparkbindings.SparkDistributedContext
import org.apache.mahout.sparkbindings.drmWrap
import org.apache.mahout.sparkbindings.sdc2sc
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.mahout.math.DenseVector

object DRMExample {

  
  // main method
  def main(args: Array[String]) {
     //1. initialize the spark and mahout context
    val conf = new SparkConf()
      .setAppName("DRMExample")
      .setMaster(args(0))
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .set("spark.kryo.registrator", "org.apache.mahout.sparkbindings.io.MahoutKryoRegistrator")
    implicit val sc = new SparkDistributedContext(new SparkContext(conf))
    
    //2. read the data file and save it in the rdd
    val lines = sc.textFile(args(1))
    
    //3. convert data read in as string in to array of double
    val test = lines.map(line => line.split('\t').map(_.toDouble))
    
    //4. add a column having value 1 in array of double this will create something like (1 | D)',  which will be used while calculating (1 | D)'
    val augumentedArray = test.map(addCentriodColumn _)
    
    //5. convert rdd of array of double in rdd of DenseVector
    val rdd = augumentedArray.map(dvec(_))
    
    //6. convert rdd to DrmRdd
    val rddMatrixLike: DrmRdd[Int] = rdd.zipWithIndex.map { case (v, idx) => (idx.toInt, v) }
    
    //7. convert DrmRdd to CheckpointedDrm[Int]
    val matrix = drmWrap(rddMatrixLike)

    //8. seperating the column having all ones created in step 4 and will use it later
    val oneVector = matrix(::, 0 until 1)
    
    //9. final input data in DrmLike[Int] format
    val dataDrmX = matrix(::, 1 until 4)
    
    
    //9. Sampling to select initial centriods
    val centriods = drmSampleKRows(dataDrmX, 2, false)
    centriods.size
    //10. Broad Casting the initial centriods
    val broadCastMatrix = drmBroadcast(centriods)
    
    
    //11. Iterating over the Data Matrix(in DrmLike[Int] format) to calculate the initial centriods
    dataDrmX.mapBlock() {
      case (keys, block) =>
        for (row <- 0 until block.nrow) {
          var dataPoint = block(row, ::)
          
          //12. findTheClosestCentriod find the closest centriod to the Data point specified by "dataPoint"
          val closesetIndex = findTheClosestCentriod(dataPoint, centriods)
          
          //13. assigning closest index to key
          keys(row) = closesetIndex
        }
        keys -> block
    }
    
    //14. Calculating the (1|D)  
    val b = (oneVector cbind dataDrmX)
    
    //15. Aggregating Transpose
    val bTranspose = (oneVector cbind dataDrmX).t
    // after step 15 bTranspose will have data in the following format
    
    /*(n+1)*K where n=dimension of the data point, K=number of clusters
    * zeroth row will contain the count of points assigned to each cluster
    * assuming 3d data points 
    * 
    */
    val nrows = b.nrow.toInt
    //16. slicing the count vectors out 
    val pointCountVectors = drmBroadcast(b(0 until 1, ::).collect(0, ::))
    val vectorSums = b(1 until nrows, ::)
    //17. dividing the data point by count vector
    vectorSums.mapBlock() {
      case (keys, block) =>
        for (row <- 0 until block.nrow) {
          block(row, ::) /= pointCountVectors
        }
        keys -> block
    }
    //18. seperating the count vectors
    val newCentriods = vectorSums.t(::,1 until centriods.size)
    
    
    //19. iterate over the above code till convergence criteria is meet 
  }
  
  // method to find the closest centriod to data point( vec: Vector  in the arguments)
  def findTheClosestCentriod(vec: Vector, matrix: Matrix): Int = {
    var index = 0
    var closest = Double.PositiveInfinity
    for (row <- 0 until matrix.nrow) {
      val squaredSum = ssr(vec, matrix(row, ::))
      val tempDist = Math.sqrt(ssr(vec, matrix(row, ::)))
      if (tempDist < closest) {
        closest = tempDist
        index = row
      }
    }
    index
  }
    
  //calculating the sum of squared distance between the points(Vectors)
  def ssr(a: Vector, b: Vector): Double = {
    (a - b) ^= 2 sum
  }

  //method used to create (1|D)
  def addCentriodColumn(arg: Array[Double]): Array[Double] = {
    val newArr = new Array[Double](arg.length + 1)
    newArr(0) = 1.0;
    for (i <- 0 until (arg.size)) {
      newArr(i + 1) = arg(i);
    }
    newArr
  }

}