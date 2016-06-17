package movie.recommend.engine

import org.jblas.DoubleMatrix

class MovieAdvisor extends Serializable {

  def recommendMovies(userId: Int, productsNumber: Int = 1): Seq[Int] = {
    Model.model
      .recommendProducts(userId, productsNumber)
      .map(r => r.product)
  }

  def recommendUsers(movieId: Int, usersNumber: Int = 1): Seq[Int] = {
    Model.model
      .recommendUsers(movieId, usersNumber)
      .map(r => r.user)
  }

  def findSimilarMovies(productId: Int, productsNumber: Int = 1): Seq[Int] = {
    val productFeatures = Model.model.productFeatures
    val productFactor = productFeatures.lookup(productId).head
    val productVector = new DoubleMatrix(productFactor)
    val similarities = productFeatures.map {
      case (id, factor) =>
        val factorVector = new DoubleMatrix(factor)
        val sim = cosineSimilarity(factorVector, productVector)
        (id, sim)
    }
    similarities.filter(_._1 != productId)
      .top(productsNumber)(Ordering.by[(Int, Double), Double] {
        case (id, sim) => sim
      }).map { case (id, sim) => id }
  }

  private def cosineSimilarity(vec1: DoubleMatrix, vec2: DoubleMatrix): Double = {
    vec1.dot(vec2) / (vec1.norm2() * vec2.norm2())
  }
}
