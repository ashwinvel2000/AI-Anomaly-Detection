using System.Text.Json.Serialization;
using System.Collections.Generic;

namespace taqa.polaris.sit.Models
{
    public class MLServiceResponse
    {
        public string Status { get; set; } = string.Empty;
        public Dictionary<string, ModelPredictionResult> Predictions { get; set; } = new();
        public Dictionary<string, string> Errors { get; set; } = new();
        [JsonPropertyName("successful_models")]
        public List<string> SuccessfulModels { get; set; } = new();
        [JsonPropertyName("failed_models")]
        public List<string> FailedModels { get; set; } = new();
        [JsonPropertyName("total_models")]
        public int TotalModels { get; set; }
        [JsonPropertyName("success_count")]
        public int SuccessCount { get; set; }
        [JsonPropertyName("error_count")]
        public int ErrorCount { get; set; }
    }

    public class ModelPredictionResult
    {
        public List<AnomalyEvent> Anomalies { get; set; } = new();
        [JsonPropertyName("anomaly_count")]
        public int AnomalyCount { get; set; }
        public PreprocessInfo Preprocess { get; set; } = new();
        [JsonPropertyName("model_version")]
        public string? ModelVersion { get; set; }
        [JsonPropertyName("internal_model")]
        public InternalModelInfo InternalModel { get; set; } = new();
        public TimingInfo Timings { get; set; } = new();
    }

    public class AnomalyEvent
    {
        public string Timestamp { get; set; } = string.Empty;
        public string Detector { get; set; } = string.Empty;
        [JsonPropertyName("RawValue")]
        public double RawValue { get; set; }
        public double Score { get; set; }
        public double Threshold { get; set; }
        public string Severity { get; set; } = string.Empty;
        [JsonPropertyName("ObservedValue")]
        public double? ObservedValue { get; set; }
        [JsonPropertyName("PredictedValue")]
        public double? PredictedValue { get; set; }
        public double? Residual { get; set; }
        public string? Tag { get; set; }
    }

    public class PreprocessInfo
    {
        [JsonPropertyName("dq_report")]
        public DataQualityReport DqReport { get; set; } = new();
        public ProcessingStats Stats { get; set; } = new();
    }

    public class DataQualityReport
    {
        [JsonPropertyName("total_rows")]
        public int TotalRows { get; set; }
        [JsonPropertyName("valid_rows")]
        public int ValidRows { get; set; }
        [JsonPropertyName("invalid_rows")]
        public int InvalidRows { get; set; }
        [JsonPropertyName("missing_values_by_column")]
        public Dictionary<string, int> MissingValuesByColumn { get; set; } = new();
        [JsonPropertyName("quality_issues")]
        public List<string> QualityIssues { get; set; } = new();
    }

    public class ProcessingStats
    {
        [JsonPropertyName("n_rows")]
        public int NRows { get; set; }
        [JsonPropertyName("n_tags")]
        public int NTags { get; set; }
    }

    public class InternalModelInfo
    {
        public string? Name { get; set; }
        public List<double>? Predictions { get; set; }
        public List<double>? Scores { get; set; }
    }

    public class TimingInfo
    {
        [JsonPropertyName("total_ms")]
        public double TotalMs { get; set; }
    }
}
