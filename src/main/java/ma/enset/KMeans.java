package ma.enset;


import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.evaluation.ClusteringEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.types.DataTypes;

public class KMeans {
    public static void main(String[] args) {
        SparkSession ss=SparkSession.builder().appName("tp spark ml").master("local[*]").getOrCreate();

        Dataset<Row> dataset=ss.read().option("inferSchema",true).option("header",true).csv("src/main/resources/Mall_Customers.csv");

        VectorAssembler vectorAssemble=new VectorAssembler().setInputCols(
                new String[]{"Age","Annual Income (k$)","Spending Score (1-100)"}
        ).setOutputCol("features");

        Dataset<Row> assembledDS=vectorAssemble.transform(dataset);

        org.apache.spark.ml.clustering.KMeans kmeans = new org.apache.spark.ml.clustering.KMeans()
                .setK(5)
                .setFeaturesCol("features")
                .setPredictionCol("prediction");

        KMeansModel model = kmeans.fit(assembledDS);
        Dataset<Row> transformed = model.transform(assembledDS);

        transformed.show(200);


// Evaluate model performance
        ClusteringEvaluator evaluator = new ClusteringEvaluator();
        double silhouette = evaluator.evaluate(transformed);
        System.out.println("Silhouette score: " + silhouette);


        // Visualize the results
        Dataset<Row> visualizationData = transformed.select("prediction", "features")
                .withColumn("features", functions.col("features"));

        visualizationData.show(); // Optional: view the data for plotting

// Create a scatter plot with different colors for each cluster
        String plotPath = "src/main/resources/plots/scatter_plot.png"; // Replace with your desired path
        visualizationData.write().format("png").save(plotPath);
    }
}