using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.IO;


class Program
{
    static void Main(string[] args)
    {
        string _appPath = Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]) ?? ".";
        string _trainDataPath = Path.GetFullPath(Path.Combine(_appPath, "..", "..", "..", "Data", "issues_train.tsv"));
        string _testDataPath = Path.GetFullPath(Path.Combine(_appPath, "..", "..", "..", "Data", "issues_test.tsv"));
        string _modelPath = Path.GetFullPath(Path.Combine(_appPath, "..", "..", "..", "Models", "model.zip"));

        MLContext mlContext = new MLContext(seed: 0);

        // Ensure files exist
        ValidateFileExists(_trainDataPath);
        ValidateFileExists(_testDataPath);

        // Load data
        IDataView trainingDataView = mlContext.Data.LoadFromTextFile<GitHubIssue>(_trainDataPath, hasHeader: true);

        // Build and train model
        var pipeline = ProcessData(mlContext);
        var trainingPipeline = BuildAndTrainModel(mlContext, trainingDataView, pipeline, _modelPath);

        // Evaluate model
        Evaluate(mlContext, _testDataPath, trainingPipeline);

        // Test prediction with a single issue
        PredictIssue(mlContext, _modelPath);
    }

    static IEstimator<ITransformer> ProcessData(MLContext mlContext)
    {
        return mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Area", outputColumnName: "Label")
            .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Title", outputColumnName: "TitleFeaturized"))
            .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Description", outputColumnName: "DescriptionFeaturized"))
            .Append(mlContext.Transforms.Concatenate("Features", "TitleFeaturized", "DescriptionFeaturized"))
            .AppendCacheCheckpoint(mlContext);
    }

    static IEstimator<ITransformer> BuildAndTrainModel(MLContext mlContext, IDataView trainingDataView, IEstimator<ITransformer> pipeline, string modelPath)
    {
        var trainingPipeline = pipeline.Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
            .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

        Console.WriteLine("Training the model...");
        var trainedModel = trainingPipeline.Fit(trainingDataView);

        Console.WriteLine("Saving the trained model...");
        mlContext.Model.Save(trainedModel, trainingDataView.Schema, modelPath);

        return trainingPipeline;
    }

    static void Evaluate(MLContext mlContext, string testDataPath, IEstimator<ITransformer> trainingPipeline)
    {
        IDataView testDataView = mlContext.Data.LoadFromTextFile<GitHubIssue>(testDataPath, hasHeader: true);
        var trainedModel = trainingPipeline.Fit(testDataView);

        var testMetrics = mlContext.MulticlassClassification.Evaluate(trainedModel.Transform(testDataView));

        Console.WriteLine("Metrics for Multi-class Classification model - Test Data");
        Console.WriteLine($"MicroAccuracy:    {testMetrics.MicroAccuracy:0.###}");
        Console.WriteLine($"MacroAccuracy:    {testMetrics.MacroAccuracy:0.###}");
        Console.WriteLine($"LogLoss:          {testMetrics.LogLoss:#.###}");
        Console.WriteLine($"LogLossReduction: {testMetrics.LogLossReduction:#.###}");
    }

    static void PredictIssue(MLContext mlContext, string modelPath)
    {
        Console.WriteLine("Loading the model for prediction...");
        ITransformer loadedModel = mlContext.Model.Load(modelPath, out var modelInputSchema);

        var predEngine = mlContext.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(loadedModel);

        var singleIssue = new GitHubIssue
        {
            Title = "Entity Framework crashes",
            Description = "When connecting to the database, EF is crashing"
        };

        var prediction = predEngine.Predict(singleIssue);
        Console.WriteLine($"Single Prediction - Result: {prediction.Area}");
    }

    static void ValidateFileExists(string filePath)
    {
        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException($"The file was not found: {filePath}");
        }
    }
}
