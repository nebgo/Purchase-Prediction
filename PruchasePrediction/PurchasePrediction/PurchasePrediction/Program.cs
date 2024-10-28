using System;
using System.IO;
using Microsoft.ML;
using PurchasePrediction;

namespace PurchasePrediction
{
    class Program
    {
        static readonly string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "data", "final_fitted_purchase_again_data.csv");
        static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "data", "final_fitted_purchase_again_data_eval.csv");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "data", "PurchaseModel.zip");

        static void Main(string[] args)
        {
            try
            {
                MLContext mlContext = new MLContext(seed: 0);
                var model = Train(mlContext, _trainDataPath);   
                Evaluate(mlContext, model, _testDataPath);     
                TestSinglePrediction(mlContext, model);        
            }
            catch (Exception ex)
            {
                Console.WriteLine($"An error occurred: {ex.Message}");
                Console.WriteLine($"Stack Trace: {ex.StackTrace}");
            }

            Console.WriteLine("Press any key to exit...");
            Console.ReadLine();
        }

        static ITransformer Train(MLContext mlContext, string dataPath)
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<PurchaseData>(dataPath, hasHeader: true, separatorChar: ',');

            var pipeline = mlContext.Transforms.Categorical.OneHotEncoding("GenderType") 
                .Append(mlContext.Transforms.Concatenate("Features", "AgeCount", "PrepurchaseAmount", "GenderType")) 
                .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "PurchaseAgain"));

            var model = pipeline.Fit(dataView);
            return model;
        }

        static void Evaluate(MLContext mlContext, ITransformer model, string testDataPath)
        {
            IDataView testDataView = mlContext.Data.LoadFromTextFile<PurchaseData>(testDataPath, hasHeader: true, separatorChar: ',');
            var predictions = model.Transform(testDataView);
            var metrics = mlContext.BinaryClassification.Evaluate(predictions, labelColumnName: "PurchaseAgain", scoreColumnName: "Score");

            Console.WriteLine();
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Model quality metrics evaluation         ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"*       Accuracy:      {metrics.Accuracy:0.##}");
            Console.WriteLine($"*       AUC:          {metrics.AreaUnderRocCurve:0.##}");
            Console.WriteLine($"*       F1 Score:     {metrics.F1Score:0.##}");
            Console.WriteLine($"*       Precision:     {metrics.PositivePrecision:0.##}");
            Console.WriteLine($"*       Recall:        {metrics.PositiveRecall:0.##}");
            Console.WriteLine($"*************************************************");
        }

        static void TestSinglePrediction(MLContext mlContext, ITransformer model)
        {
            var predictionEngine = mlContext.Model.CreatePredictionEngine<PurchaseData, PurchasePrediction>(model);
            var sampleData = new PurchaseData()
            {
                CustomerID = "499",
                AgeCount = 52,
                GenderType = "Male",
                PrepurchaseAmount = 3
            };

            var prediction = predictionEngine.Predict(sampleData);

            float probability = (float)(1 / (1 + Math.Exp(-prediction.Score)));

            Console.WriteLine($"**********************************************************************");
            Console.WriteLine($"Predicted Purchase Again: {prediction.PurchaseAgain} (Probability: {probability:0.##})");
            Console.WriteLine($"Score (Raw): {prediction.Score}");
            Console.WriteLine($"**********************************************************************");
        }
    }
}
