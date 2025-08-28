using System.ComponentModel.DataAnnotations;
using System.Text.Json.Serialization;

namespace taqa.polaris.sit.Models
{
    public class AnomalyReportData
    {
        [JsonPropertyName("severityChart")]
        public string SeverityChart { get; set; } = string.Empty;
        
        [JsonPropertyName("timelineChart")]
        public string TimelineChart { get; set; } = string.Empty;
        
        [JsonPropertyName("tableData")]
        public List<AnomalyTableRow> TableData { get; set; } = new();
        
        [JsonPropertyName("totalAnomalies")]
        public int TotalAnomalies { get; set; }
        
        [JsonPropertyName("summaryData")]
        public Dictionary<string, int> SummaryData { get; set; } = new();
        
        [JsonPropertyName("batchState")]
        public Dictionary<string, int> BatchState { get; set; } = new();
        
        [JsonPropertyName("generatedAt")]
        public string GeneratedAt { get; set; } = string.Empty;

        // Add deployment metrics for PDF report
        [JsonPropertyName("inferenceLatencyMs")]
        public double? InferenceLatencyMs { get; set; }
        [JsonPropertyName("throughput")]
        public double? Throughput { get; set; }
        [JsonPropertyName("memoryUsedMb")]
        public double? MemoryUsedMb { get; set; }
        [JsonPropertyName("cpuUsedMs")]
        public double? CpuUsedMs { get; set; }
    }

    public class AnomalyTableRow
    {
        [JsonPropertyName("number")]
        public string Number { get; set; } = string.Empty;
        
        [JsonPropertyName("timestamp")]
        public string Timestamp { get; set; } = string.Empty;
        
        [JsonPropertyName("detector")]
        public string Detector { get; set; } = string.Empty;
        
        [JsonPropertyName("rawValue")]
        public string RawValue { get; set; } = string.Empty;
        
        [JsonPropertyName("score")]
        public string Score { get; set; } = string.Empty;
        
        [JsonPropertyName("threshold")]
        public string Threshold { get; set; } = string.Empty;
        
        [JsonPropertyName("severity")]
        public string Severity { get; set; } = string.Empty;
        
        [JsonPropertyName("observed")]
        public string Observed { get; set; } = string.Empty;
        
        [JsonPropertyName("predicted")]
        public string Predicted { get; set; } = string.Empty;
    }
}