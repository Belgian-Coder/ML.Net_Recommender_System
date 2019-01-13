using BookRecommenderShared.Configuration;
using BookRecommenderShared.Models;
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using System;
using System.IO;

namespace BookRecommenderPredictor
{
    class Program
    {
        static void Main(string[] args)
        {
            var mlContext = new MLContext();

            ITransformer loadedModel;
            
            Console.WriteLine("Reading the model from disk");
            // Load the model into memory
            using (var file = File.OpenRead(Path.Combine(Environment.CurrentDirectory, Filenames.TrainedModel)))
            {
                loadedModel = mlContext.Model.Load(file);
                Console.WriteLine($"The model was read from file: {Filenames.TrainedModel}");
            }

            // Create the prediction function from the loaded model
            var predictionFunction = loadedModel.CreatePredictionEngine<BookRating, BookRatingPrediction>(mlContext);

            // Predict the outcome
            var bookPrediction = predictionFunction.Predict(new BookRating
            {
                user = 91,
                bookid = 10365
            });

            Console.WriteLine($"Predicted rating: {Math.Round(bookPrediction.Score, 1)}");
            Console.ReadLine();
        }
    }
}
