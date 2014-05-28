package edu.uw.tacoma.tcss;

/**
 * @author Hussein Huwaidi
 * @version 1.0
 * 
 * Description:
 * This class performs consensus kmeans clustering using Spark technology.
 *
 * History:
 * 
 * Version	Date			Description
 * -------	----			-----------
 * v1.0		05/28/2014		Initial implementation
 */

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import scala.util.Random;

import java.util.Set;
import java.util.TreeSet;
import java.util.regex.Pattern;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.util.Vector;

import scala.Tuple2;

public final class SparkConsensusKMeans {
	private static final Pattern SPACE = Pattern.compile(" ");
	static JavaSparkContext sc;

	/**
	 * CLT interface for executing the algorithm.
	 * @param args Usage: SparkConsensusKMeans <master> <file> <k-values i.e. 1,4,...,10,20,...,11,21> <Sample_Size> <Converge Distance> <max_iterations>
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {

		// Checking parameters length
		if (args.length < 5) {
			System.err
					.println("Usage: SparkConsensusKMeans <master> <file> <k-values i.e. 1,4,...,10,20,...,11,21> <Sample_Size> <Converge Distance> <max_iterations>");
			System.exit(1);
		}

		// Reading parameters
		String master = args[0];
		String path = args[1];
		ArrayList<Integer> K = readK(args[2]);
		int sampleSize = Integer.parseInt(args[3]);
		double convergeDist = Double.parseDouble(args[4]);
		int maxIterations = Integer.parseInt(args[5]);

		// Validating input values
		if (sampleSize < Collections.max(K)) {
			System.err
					.println("Sample size must be equal or more than maximum K value entered.");
			System.exit(1);
		}

		// Creating a Spark Context object
		sc = new JavaSparkContext(master, "SparkConsensusKMeans",
				"/Development/spark", "/Development/spark/JARS/kmeans.jar");

		// Reading data points using separator
		JavaRDD<Vector> data = sc.textFile(path)
				.map(new Function<String, Vector>() {
					@Override
					public Vector call(String line) {
						return parseVector(line);
					}
				}).cache(); // Keeping data in RDD for faster access by nodes

		long[] times = new long[3]; // Running for 3 times to get average execution time
		for (int t = 0; t < 3; t++) {
			long time = System.currentTimeMillis();

			// Performing consensus kmeans clustering
			double bestConcensus = Double.MAX_VALUE;
			int bestK = -1;
			for (int i = 0; i < K.size(); i++) {
				double tempConsensus = calculateConsensus(data, sampleSize,
						K.get(i), convergeDist, maxIterations);

				// Best consensus is the one closer to either 0 or 1
				if ((bestConcensus > Math.abs(Math.sqrt(1 - Math.pow(
						tempConsensus, 2))))
						|| (bestConcensus > Math.abs(Math.sqrt(0 - Math.pow(
								tempConsensus, 2))))) {
					bestConcensus = tempConsensus;
					bestK = K.get(i);
				}
			}
			kmeans(data, bestK, convergeDist, maxIterations);
			
			// Done!
			
			time = System.currentTimeMillis() - time;
			times[t] = time;
			System.out.println("Best K= " + bestK + " with consensus of "
					+ bestConcensus);
		}

		// Printing timing results
		System.out.println("Average Times taken: "
				+ ((times[0] + times[1] + times[2]) / 3.0) / 1000 + " seconds");

		System.exit(0);
	}
	
	/**
	 * THis method subtracts one list from the other and return a new list with different items i.e. MINUS
	 * @param list1
	 * @param list2
	 * @return
	 */
	public static List<Vector> subtract(List<Vector> list1, List<Vector> list2) {  
        List<Vector> result = new ArrayList<Vector>(); 
        boolean found = false;
        for (Vector t1 : list1) { 
        	for(Vector t2 : list2){
	            if( t1.dist(t2)==0.0 ) {  
	            	found = true;
	            }
            }
        	if(!found){
        		result.add(t1);
        	}
        }  
        return result;  
    }  
	
	/**
	 * Consensus Clustering:
	 * It's an algorithm that runs a clustering algorithm using a set of numbers of centers desired and returns the best
	 * clustering based on connectivity of the clustered points. This method calculates consensus connectivity value for a
	 * given K value.
	 * 
	 * Algorithm:
	 * for K = K1, K2, ... Kmax
	 * Start with an empty matrix M (NXN)
	 * for h = 1, 2, ... H
	 * D(h) = Resample D (Bootstrapping or Subsampling)
	 * M(h) = KMeans(D, K)
	 * M = M union M(h)
	 * end
	 * Mk[K] = Consensus_Measure(M)
	 * end
	 * K^ = best Mk (Basically Max Mk)
	 * P = KMeans(D, K^)
	 * return P
	 * 
	 * Note:
	 * average of connectivity of all pairs i.e.  ( 1 if belong to same cluster 0 otherwise )
	 * The average connectivity = (Sum of M(h) connectivity)/Sum( I(h))
	 * where I(i,j) = 1 if ei and ej belong to the same D(h) or 0 otherwise)
	 * 
	 * @param data Dataset D={e1, e2, ... eN}
	 * @param sampleSize Sample Size
	 * @param K number of clusters
	 * @param convergeDist Convergence Distance
	 * @param maxIterations Number of maximum iterations h
	 * @return Best K value
	 */
	public static double calculateConsensus(JavaRDD<Vector> data, int sampleSize, int K, double convergeDist, int maxIterations) {
		// Random seed generator
		final Random random = new Random();
		// D: Generic Data set
		JavaRDD<Vector> D = data;
		// N: Number of Items in D
		long N = D.count();
		// P: Partition of D into K
		// Nk: Number of Items in Pk
		final int Nk = sampleSize;
		// H: Number of Resampling
		final double H = maxIterations;
		// Populate N x H with K value for each point
		JavaPairRDD<Vector, Integer> M = null;
		// Presence Indicator
		double I = 0.0;
		
		JavaPairRDD<Integer, Vector> mergedPoints = null;

		for (int i = 0; i < H; i++) {
			double tempDist;
			JavaPairRDD<Integer, Vector> clusteredPoints = null;

			// Dh: Dataset obtained by Resampling
			List<Vector> Dh = D.takeSample(false, Nk, random.nextInt());
			JavaRDD<Vector> DhRDD = sc.parallelize(Dh, 1);

			// M: Connectivity Matrix M, here we are summing the number of
			// similar points between two samples which is why algorithm
			// performs comparison between them.
			
			// Keeping track of D taken for each H i.e. <H,D>
			final int ii = i;
			JavaPairRDD<Integer, Vector> points = DhRDD.map(new PairFunction<Vector,Integer,Vector>(){

				@Override
				public Tuple2<Integer, Vector> call(Vector arg0)
						throws Exception {
					// TODO Auto-generated method stub
					return new Tuple2<Integer, Vector>(ii, arg0);
				}
				
			});
			
			// Building a matrix of all samples <H,D> i.e. <<h1,d1>,<h2,d2>,...>
			if(mergedPoints==null)
				mergedPoints = points;
			else
				mergedPoints = mergedPoints.union(points);
			
			do { // While divergent distance not reached perform normal K-Means

				// Taking random centroids
				final List<Vector> centroids = DhRDD.takeSample(false, K,random.nextInt());
				
				// allocate each vector to closest centroid
				JavaPairRDD<Integer, Vector> closest = DhRDD
						.map(new PairFunction<Vector, Integer, Vector>() {
							@Override
							public Tuple2<Integer, Vector> call(Vector vector) {
								return new Tuple2<Integer, Vector>(closestPoint(
										vector, centroids), vector);
							}
						});
				
				// group by cluster id and average the vectors within each cluster
				// to compute centroids
				JavaPairRDD<Integer, List<Vector>> pointsGroup = closest
						.groupByKey();
				
				// calculating new centroids
				Map<Integer, Vector> newCentroids = pointsGroup.mapValues(
						new Function<List<Vector>, Vector>() {
							@Override
							public Vector call(List<Vector> ps) {
								return average(ps);
							}
						}).collectAsMap();

				// Calculating divergent for current pass
				tempDist = 0.0;
				for (int j = 0; j < K; j++) {
					if (newCentroids.get(j) != null) {
						tempDist += centroids.get(j).squaredDist(
								newCentroids.get(j));
					}
				}

				// Updating clusters with new centroids
				for (Map.Entry<Integer, Vector> t : newCentroids.entrySet()) {
					centroids.set(t.getKey(), t.getValue());
				}
				clusteredPoints = closest;
			} while (tempDist > convergeDist);
						
			// Storing clustered points for current H i.e. Mh
			JavaPairRDD<Vector, Integer> Mh = clusteredPoints
					.map(new PairFunction<Tuple2<Integer, Vector>, Vector, Integer>() {

						@Override
						public Tuple2<Vector, Integer> call(
								Tuple2<Integer, Vector> arg0) throws Exception {
							return new Tuple2<Vector, Integer>(arg0._2, arg0._1);
						}

					});

			// Building a matrix M of all <Mh,h> i.e. <<<M1,1>,<M2,2>,...>
			if (M == null) {
				M = Mh;
			} else {
				M = M.union(Mh);
			}
		}
				
		// Transforming mergedPoints to <H,All_Samples>
		final JavaPairRDD<Integer, List<Vector>> groupedPoints = mergedPoints.groupByKey();
		
		// Changing to map
		Map<Integer, List<Vector>> pointsMap = groupedPoints.collectAsMap();
		
		// Building connectivity map <D,factor> i.e. <<d1,value>,<d2,value>,...> between all samples in pointsMap
		Map<Integer, Double> connectivityMap = new HashMap<Integer, Double>();
		for(int i = 0 ; i < pointsMap.size(); i++){
			if(i==0){
				connectivityMap.put(i, 1.0);
			}else{
				int currentDatasetSize = pointsMap.size();
				List<Vector> differentPoints = subtract(pointsMap.get(i-1), pointsMap.get(i));
				int NumberOfelementsInPreviousDatasetNotInCurrentDataset = differentPoints.size();
				int numberOfSimilarItems = currentDatasetSize - NumberOfelementsInPreviousDatasetNotInCurrentDataset;
				connectivityMap.put(i, (double) (numberOfSimilarItems/pointsMap.size()));
			}
		}
		final Map<Integer, Double> connectivity = connectivityMap;

		// Calculating Mu i.e. if M(i,j) then Mu is the factor where i and j belong to the same cluster
		JavaPairRDD<Vector, List<Integer>> groupedM = M.groupByKey();
		Map<Vector, List<Integer>> groupedMMap = groupedM.collectAsMap();
		Map<Vector, Double> Mu = new HashMap<Vector,Double>();
		Set<Vector> keyList = groupedMMap.keySet();
		double iSum = 0.0;
		for(int i = 0 ; i < keyList.size(); i++){
			Mu.put((Vector) keyList.toArray()[i], (double)groupedMMap.get(keyList.toArray()[i]).size()/H);
			iSum += ((double)groupedMMap.get(keyList.toArray()[i]).size()/H);
		}
//		JavaPairRDD<Vector, Double> completedMh = groupedM
//				.map(new PairFunction<Tuple2<Vector, List<Integer>>, Vector, Double>() {
//
//					@Override
//					public Tuple2<Vector, Double> call(
//							Tuple2<Vector, List<Integer>> arg0)
//							throws Exception {
//						double sum = 0;
//						Iterator<Integer> iterator = arg0._2.iterator();
//
//						while (iterator.hasNext()) {
//							sum += iterator.next();
//						}
//						return new Tuple2<Vector, Double>(arg0._1,
//								(sum / connectivity.get(arg0)));
//					}
//				});
//
//		double iSum = 0.0;
//		while(iterator.hasNext()){
//			iSum += iterator.next()._2;
//		}
		
		double CDF = (iSum/((N*(N-1))/2));
		return CDF; // connectivityIdentity(results, sampleSize);
	}

	// Sample Array: [(0,[(4.5, 5.0), (1.5, 2.0)]), (1,[(5.0, 7.0)]), (0,[(1.5,
	// 2.0)]), (1,[(4.5, 5.0), (5.0, 7.0)]), (0,[(1.5, 2.0)]), (1,[(4.5, 5.0),
	// (5.0, 7.0)])]
//	public static double connectivityIdentity(
//			ArrayList<Tuple2<Integer, List<Vector>>> array, int sampleSize) {
//
//		boolean[][] connectivity = new boolean[sampleSize][sampleSize];
//		int tuple = 0;
//		for (int i = 0; i < array.get(tuple)._2().size(); i++) {
//			for (int j = 0; j < array.get(tuple)._2().size(); j++) {
//				if (connectedOrNot(array.get(tuple)._2(), array.get(tuple)._2()
//						.get(i), array.get(tuple)._2().get(j))) {
//					connectivity[i][j] = true;
//				} else {
//					connectivity[i][j] = false;
//				}
//			}
//			if (array.get(tuple + 1)._2().size() != i + 1) {
//				break;
//			}
//			tuple++;
//		}
//
//		return ((double) countTrue(connectivity) / (double) connectivity.length);
//	}
//
//	public static int countTrue(boolean[][] connectivity) {
//		int count = 0;
//		for (int i = 0; i < connectivity.length; i++) {
//			for (int j = 0; j < connectivity[i].length; j++) {
//				if (connectivity[i][j]) {
//					count++;
//				}
//			}
//		}
//		return count;
//	}
//
//	public static boolean connectedOrNot(List<Vector> listOfVectors,
//			Vector point1, Vector point2) {
//		if (listOfVectors.contains(point1) && listOfVectors.contains(point2)) {
//			return true;
//		} else {
//			return false;
//		}
//	}

	public static List<Vector> kmeans(JavaRDD<Vector> data, int K,
			double convergeDist, long maxIterations) {
		final List<Vector> centroids = data.takeSample(false, K, 42);

		double tempDist;
		do {
			// allocate each vector to closest centroid
			JavaPairRDD<Integer, Vector> closest = data
					.map(new PairFunction<Vector, Integer, Vector>() {
						@Override
						public Tuple2<Integer, Vector> call(Vector vector) {
							return new Tuple2<Integer, Vector>(closestPoint(
									vector, centroids), vector);
						}
					});

			// group by cluster id and average the vectors within each cluster
			// to compute centroids
			JavaPairRDD<Integer, List<Vector>> pointsGroup = closest
					.groupByKey();
			Map<Integer, Vector> newCentroids = pointsGroup.mapValues(
					new Function<List<Vector>, Vector>() {
						@Override
						public Vector call(List<Vector> ps) {
							return average(ps);
						}
					}).collectAsMap();
			tempDist = 0.0;
			for (int j = 0; j < K; j++) {
				tempDist += centroids.get(j).squaredDist(newCentroids.get(j));
			}
			for (Map.Entry<Integer, Vector> t : newCentroids.entrySet()) {
				centroids.set(t.getKey(), t.getValue());
			}
		} while (tempDist > convergeDist);

		System.out.println("Final centers:");
		for (Vector c : centroids) {
			System.out.println(c);
		}

		return centroids;
	}

	public static ArrayList<Integer> readK(String string) throws Exception {
		String[] parameters = string.split(",");
		ArrayList<Integer> result = new ArrayList<Integer>();
		for (int i = 0; i < parameters.length; i++) {
			if (result.size() != 0 && parameters[i].equals("...")) {
				int startingValue = result.get(result.size() - 1);
				int lastValue = Integer.parseInt(parameters[i + 1]);
				if (startingValue < lastValue) {
					while (startingValue != lastValue) {
						result.add(++startingValue);
					}
					i++;
				} else if (startingValue > lastValue) {
					while (startingValue != lastValue) {
						result.add(--startingValue);
					}
					i++;
				} // else if equal then ignore
			} else if (result.size() == 0 && parameters[i].equals("...")) {
				throw new Exception();
			} else if (parameters[i].equals(".") || parameters[i].equals("..")
					|| !isInteger(parameters[i])) {
				System.err
						.println("Value "
								+ parameters[i]
								+ " was ignored. Only numeric values or range dots '...' seperated by a comma ',' are accepted!");
			} else if (!parameters[i].equals("") && isInteger(parameters[i])) {
				result.add(Integer.parseInt(parameters[i]));
			}
		}

		// Just sorting
		Set<Integer> noDubplicationsSet = new TreeSet<Integer>(result);
		result = new ArrayList<Integer>(noDubplicationsSet);
		Collections.sort(result);
		return result;
	}

	public static boolean isInteger(String input) {
		try {
			Integer.parseInt(input);
			return true;
		} catch (Exception e) {
			return false;
		}
	}

	/** Parses numbers split by whitespace to a vector */
	static Vector parseVector(String line) {
		String[] splits = SPACE.split(line);
		double[] data = new double[splits.length];
		int i = 0;
		for (String s : splits) {
			data[i] = Double.parseDouble(s);
			i++;
		}
		return new Vector(data);
	}

	/**
	 * Computes the vector to which the input vector is closest using squared
	 * distance
	 */
	static int closestPoint(Vector p, List<Vector> centers) {
		int bestIndex = 0;
		double closest = Double.POSITIVE_INFINITY;
		for (int i = 0; i < centers.size(); i++) {
			double tempDist = p.squaredDist(centers.get(i));
			if (tempDist < closest) {
				closest = tempDist;
				bestIndex = i;
			}
		}
		return bestIndex;
	}

	/** Computes the mean across all vectors in the input set of vectors */
	static Vector average(List<Vector> ps) {
		int numVectors = ps.size();
		Vector out = new Vector(ps.get(0).elements());
		// start from i = 1 since we already copied index 0 above
		for (int i = 1; i < numVectors; i++) {
			out.addInPlace(ps.get(i));
		}
		return out.divide(numVectors);
	}
}
