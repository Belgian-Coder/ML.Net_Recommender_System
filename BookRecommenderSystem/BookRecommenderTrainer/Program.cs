using BookRecommenderShared.Configuration;
using BookRecommenderShared.Models;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using System;
using System.IO;

namespace BookRecommenderTrainer
{
    class Program
    {
        static void Main(string[] args)
        {
            var mlContext = new MLContext();

            var reader = mlContext.Data.CreateTextReader(new TextLoader.Arguments()
            {
                Separator = ",",
                HasHeader = true,
                /* 
                 * Define the columns to read be from the csv.
                 * The indeces are based on the position in the csv file and will receive a new index.
                 * DataKind is defined as R4 (floating point) instead of integer
                 * Computers are optimized for processing floating point calculations 
                 * so this will not create performance issues but mitigates some possible issues
                 */
                Column = new[]
                {
                    new TextLoader.Column(Labels.Label, DataKind.R4, 1), // new index 0
                    new TextLoader.Column(Labels.User, DataKind.R4, 2),  // new index 1
                    new TextLoader.Column(Labels.BookId, DataKind.R4, 3) // new index 2
                }
            });

            var data = reader.Read(Path.Combine(Environment.CurrentDirectory, Filenames.DataFolder, Filenames.RatingsDataset));

            // Split our dataset in 80% training data and 20% testing data to evaluate our model
            var (trainData, testData) = mlContext.BinaryClassification.TrainTestSplit(data, testFraction: 0.2);

            var pipeline = mlContext.Transforms.Conversion.MapValueToKey(Labels.User, Labels.UserEncoded)
                .Append(mlContext.Transforms.Conversion.MapValueToKey(Labels.BookId, Labels.BookIdEncoded))
                .Append(new MatrixFactorizationTrainer(mlContext, Labels.UserEncoded, Labels.BookIdEncoded, Labels.Label,
                advancedSettings: s =>
                {
                    s.NumIterations = 50;
                    s.K = 500;
                    s.NumThreads = 1;
                }));

            Console.WriteLine("Creating and training the model");
            // Fit the data/train the model
            var model = pipeline.Fit(trainData);
            Console.WriteLine("Training finished");

            // Use the test data to evaluate the difference between the real value and the prediction
            var prediction = model.Transform(testData);
            var metrics = mlContext.Regression.Evaluate(prediction);
            Console.WriteLine("Model Metrics:");
            Console.WriteLine($"RMS: {metrics.Rms}");
            Console.WriteLine($"R^2: {metrics.RSquared}");
            Console.ReadLine();

            var predictionFunction = model.CreatePredictionEngine<BookRating, BookRatingPrediction>(mlContext);

            var bookPrediction = predictionFunction.Predict(new BookRating
            {
                user = 91,
                bookid = 10365
            });

            Console.WriteLine($"Predicted rating: {Math.Round(bookPrediction.Score, 1)}");
            Console.ReadLine();

            Console.WriteLine("Saving model");
            using (var stream = new FileStream(
                Path.Combine(Environment.CurrentDirectory, Filenames.TrainedModel), 
                FileMode.Create, 
                FileAccess.Write, 
                FileShare.Write))
            {
                mlContext.Model.Save(model, stream);
                Console.WriteLine($"The model is saved as {Filenames.TrainedModel}");
            }

            Console.ReadLine();
        }
    }
}
