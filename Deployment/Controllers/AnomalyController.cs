using Microsoft.AspNetCore.Mvc;
using taqa.polaris.sit.Services.Anomaly;
using AnomalyRouter = taqa.polaris.sit.Services.Anomaly.AnomalyRouter.AnomalyRouter;
using CsvStreamReader = taqa.polaris.sit.Services.Anomaly.CsvStreamReader.CsvStreamReader;
using MultiCsvJoiner = taqa.polaris.sit.Services.Anomaly.MultiCsvJoiner.MultiCsvJoiner;
using taqa.polaris.sit.Models;
using QuestPDF.Fluent;
using QuestPDF.Helpers;
using QuestPDF.Infrastructure;
using System.IO;
using taqa.polaris.sit.Configuration;
using taqa.polaris.sit.DataHub;
using taqa.polaris.sit.ControllerStats;
using taqa.polaris.sit.NamingModel;
using taqa.polaris.sit.Utilities;
using taqa.polaris.sit.Types;
using taqa.polaris.sit.CsvMaps;
using System.Text;
using System.Diagnostics;
using CsvHelper;
using CsvHelper.Configuration;
using taqa.polaris.sit.Services;
using taqa.polaris.sit.Models;

namespace taqa.polaris.sit.Controllers
{
    [Route("Anomaly")]
    public class AnomalyController : Controller
    {
        private readonly IWebHostEnvironment _env;
        private readonly AnomalyRouter _router;
        private readonly ILogger<AnomalyController> _logger;
        private readonly APIConfig _apiConfig;
        private DataHubServices _datahubServices;
        private StreamStatus _streamStatus;
        private readonly CsvHandler _csvHandler;
        private readonly MLServiceClient _mlServiceClient;
        private readonly IConfiguration _config;

        static AnomalyController()
        {
            // Configure QuestPDF to use free license
            QuestPDF.Settings.License = LicenseType.Community;
        }

        public AnomalyController(IWebHostEnvironment env, AnomalyRouter router, ILogger<AnomalyController> logger, APIConfig apiConfig, MLServiceClient mlServiceClient, IConfiguration config)
        {
            _env = env;
            _router = router;
            _logger = logger;
            _apiConfig = apiConfig;
            _csvHandler = new CsvHandler(_logger);
            _mlServiceClient = mlServiceClient;
            _config = config;
        }

        // Feature flag for ML API usage
        private bool UseMLApi => _config.GetValue<bool>("MLService:UseApi", true);

        // ─── GET  /Anomaly/Upload 
        [HttpGet("Upload")]
        public IActionResult Upload() 
        {
            // Clean up any leftover temp files
            CleanupTempFiles();
            return View();
        }

        // ─── API: Get tools for namespace ────────────────────────────────────
        [HttpGet("GetToolsForNamespace")]
        public async Task<IActionResult> GetToolsForNamespace(string @namespace)
        {
            try
            {
                AppConfig.assignNamespaceChoice(namespaceId: @namespace);
                _apiConfig.updateApiConfig();
                
                _streamStatus = new(_logger, _apiConfig.metadataService);
                var toolIds = await _streamStatus.GetIdAndStreamTypes(type: "SIT");
                
                return Json(toolIds.Keys.ToList());
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error fetching tools for namespace {Namespace}", @namespace);
                return StatusCode(500, new { error = "Failed to fetch tools" });
            }
        }

        // ─── API: Get streams for tool ────────────────────────────────────
        [HttpGet("GetStreamsForTool")
        ]
        public async Task<IActionResult> GetStreamsForTool(string @namespace, string toolId)
        {
            try
            {
                AppConfig.assignNamespaceChoice(namespaceId: @namespace);
                _apiConfig.updateApiConfig();

                // Get the required streams configuration for anomaly detection
                var requiredStreams = new[]
                {
                    "BatteryVoltage", "UpstreamPressure", "DownstreamPressure",
                    "UpstreamTemperature", "DownstreamTemperature", "ChokePosition",
                    "TargetPosition", "ToolState", "DownstreamUpstreamDifference"
                };

                var streamConfig = StreamConfig.GetSitStreamConfig(toolId);
                var availableStreams = new Dictionary<string, string>();

                // Check which required streams are available
                foreach (var streamKey in requiredStreams)
                {
                    if (streamConfig.ContainsKey(streamKey))
                    {
                        availableStreams[streamKey] = streamConfig[streamKey].StreamId;
                    }
                }

                return Json(availableStreams);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error fetching streams for tool {ToolId}", toolId);
                return StatusCode(500, new { error = "Failed to fetch streams" });
            }
        }

        // ─── POST /Anomaly/DetectFromDataHub ──────────────────────────────────────────
        [HttpPost("DetectFromDataHub")]
        public async Task<IActionResult> DetectFromDataHub(string selectedNamespace, string toolId, DateTime startDate, DateTime? endDate = null)
        {
            var overallStopwatch = Stopwatch.StartNew();
            _logger.LogInformation("=== ANOMALY DETECTION START (DataHub) === Tool: {ToolId}, Namespace: {Namespace}", toolId, selectedNamespace);
            
            try
            {
                // Set up DataHub connection
                var setupStopwatch = Stopwatch.StartNew();
                AppConfig.assignNamespaceChoice(namespaceId: selectedNamespace);
                _apiConfig.updateApiConfig();
                _datahubServices = new DataHubServices(_logger, _apiConfig.dataService);
                setupStopwatch.Stop();
                _logger.LogInformation("DataHub setup completed in {ElapsedMs}ms", setupStopwatch.ElapsedMilliseconds);

                // Use end date or default to now
                var actualEndDate = endDate ?? DateTime.Now;
                var startDateStr = startDate.ToString("yyyy-MM-dd");

                // Get stream configuration for the tool
                var streamConfig = StreamConfig.GetSitStreamConfig(toolId);
                
                // Define the required streams for anomaly detection
                var requiredStreamKeys = new[]
                {
                    "BatteryVoltage", "UpstreamPressure", "DownstreamPressure",
                    "UpstreamTemperature", "DownstreamTemperature", "ChokePosition",
                    "TargetPosition", "ToolState", "DownstreamUpstreamDifference"
                };

                // Download and process each stream
                var downloadStopwatch = Stopwatch.StartNew();
                var tagStreams = new Dictionary<string, IEnumerable<(DateTime, double)>>();
                var downloadedStreams = new List<string>();

                foreach (var streamKey in requiredStreamKeys)
                {
                    if (streamConfig.ContainsKey(streamKey))
                    {
                        try
                        {
                            var streamDownloadTimer = Stopwatch.StartNew();
                            var streamId = streamConfig[streamKey].StreamId;
                            _logger.LogInformation("Downloading stream {StreamKey} with ID {StreamId}", streamKey, streamId);

                            // Download stream data based on stream type
                            var streamData = await DownloadStreamData(streamId, streamKey, startDateStr);
                            streamDownloadTimer.Stop();
                            
                            if (streamData != null && streamData.Any())
                            {
                                // Convert stream name to match the expected tag names
                                var tagName = ConvertStreamKeyToTagName(streamKey);
                                tagStreams[tagName] = streamData;
                                downloadedStreams.Add(streamKey);
                                _logger.LogInformation("Successfully downloaded {Count} records for {StreamKey} in {ElapsedMs}ms", 
                                    streamData.Count(), streamKey, streamDownloadTimer.ElapsedMilliseconds);
                            }
                            else
                            {
                                _logger.LogWarning("No data found for stream {StreamKey} (download time: {ElapsedMs}ms)", 
                                    streamKey, streamDownloadTimer.ElapsedMilliseconds);
                            }
                        }
                        catch (Exception ex)
                        {
                            _logger.LogWarning(ex, "Failed to download stream {StreamKey}, continuing without it", streamKey);
                        }
                    }
                    else
                    {
                        _logger.LogWarning("Stream {StreamKey} not found in configuration for tool {ToolId}", streamKey, toolId);
                    }
                }
                
                downloadStopwatch.Stop();
                _logger.LogInformation("Total download time: {ElapsedMs}ms for {Count} streams", 
                    downloadStopwatch.ElapsedMilliseconds, downloadedStreams.Count);

                if (!tagStreams.Any())
                {
                    overallStopwatch.Stop();
                    _logger.LogError("No data streams downloaded - total time: {ElapsedMs}ms", overallStopwatch.ElapsedMilliseconds);
                    TempData["Msg"] = "No data streams could be downloaded from DataHub.";
                    return RedirectToAction("Upload");
                }

                _logger.LogInformation("Downloaded {Count} streams: {Streams}", downloadedStreams.Count, string.Join(", ", downloadedStreams));

                // Process the data using the same logic as CSV upload
                return await ProcessAnomalyData(tagStreams, isDataHubData: true, overallStopwatch);
            }
            catch (Exception ex)
            {
                overallStopwatch.Stop();
                _logger.LogError(ex, "Error in DetectFromDataHub after {ElapsedMs}ms", overallStopwatch.ElapsedMilliseconds);
                TempData["Msg"] = $"Error downloading data from DataHub: {ex.Message}";
                return RedirectToAction("Upload");
            }
        }

        // ─── Helper: Download stream data ─────────────────────────────────────
        private async Task<IEnumerable<(DateTime, double)>?> DownloadStreamData(string streamId, string streamKey, string startDateStr)
        {
            try
            {
                // Map stream key to the appropriate data type and download method
                return streamKey switch
                {
                    "BatteryVoltage" => await DownloadTypedStream<BatteryVoltageType>(streamId, startDateStr, data => data.Voltage),
                    "UpstreamPressure" => await DownloadTypedStream<UpstreamPressureType>(streamId, startDateStr, data => data.Pressure),
                    "DownstreamPressure" => await DownloadTypedStream<DownstreamPressureType>(streamId, startDateStr, data => data.Pressure),
                    "UpstreamTemperature" => await DownloadTypedStream<UpstreamTemperatureType>(streamId, startDateStr, data => data.Temperature),
                    "DownstreamTemperature" => await DownloadTypedStream<DownstreamTemperatureType>(streamId, startDateStr, data => data.Temperature),
                    "ChokePosition" => await DownloadTypedStream<PositionChokeType>(streamId, startDateStr, data => data.Position),
                    "TargetPosition" => await DownloadTypedStream<PositionTargetType>(streamId, startDateStr, data => data.Position),
                    "ToolState" => await DownloadTypedStream<ToolStateType>(streamId, startDateStr, data => (double)data.State),
                    "DownstreamUpstreamDifference" => await DownloadTypedStream<DownstreamUpstreamDifferenceType>(streamId, startDateStr, data => data.Pressure),
                    _ => null
                };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error downloading stream {StreamId} of type {StreamKey}", streamId, streamKey);
                return null;
            }
        }

        // ─── Helper: Download typed stream ─────────────────────────────────────
        private async Task<IEnumerable<(DateTime, double)>> DownloadTypedStream<T>(string streamId, string startDateStr, Func<T, double> valueSelector)
            where T : class
        {
            var data = await _datahubServices.DownloadStreamToMemory<T>(streamId: streamId, startDateStr: startDateStr);
            
            return data.Select(item =>
            {
                // Extract timestamp - handle both DateTime and DateTimeOffset
                var timestampProperty = item.GetType().GetProperty("Timestamp");
                var timestampValue = timestampProperty?.GetValue(item);
                
                DateTime timestamp;
                if (timestampValue is DateTimeOffset dateTimeOffset)
                {
                    timestamp = dateTimeOffset.DateTime;
                }
                else if (timestampValue is DateTime dateTime)
                {
                    timestamp = dateTime;
                }
                else
                {
                    // Try to parse as string if it's neither DateTime nor DateTimeOffset
                    if (DateTime.TryParse(timestampValue?.ToString(), out var parsedDateTime))
                    {
                        timestamp = parsedDateTime;
                    }
                    else
                    {
                        throw new InvalidOperationException($"Unable to extract timestamp from {timestampValue?.GetType().Name}: {timestampValue}");
                    }
                }
                
                var value = valueSelector(item);
                return (timestamp, value);
            }).Where(item => !double.IsNaN(item.value) && double.IsFinite(item.value));
        }

        // ─── Helper: Align timestamps for DataHub data ─────────────────────────────────────
        private Dictionary<string, IEnumerable<(DateTime, double)>> AlignTimestamps(Dictionary<string, IEnumerable<(DateTime, double)>> tagStreams)
        {
            var alignStopwatch = Stopwatch.StartNew();
            
            if (!tagStreams.Any()) return tagStreams;

            _logger.LogInformation("Aligning timestamps for DataHub data across {Count} streams", tagStreams.Count);

            var alignedStreams = new Dictionary<string, IEnumerable<(DateTime, double)>>();

            foreach (var stream in tagStreams)
            {
                // Round timestamps to nearest minute for alignment
                var alignedData = stream.Value
                    .Select(item => (
                        timestamp: new DateTime(
                            item.Item1.Year, 
                            item.Item1.Month, 
                            item.Item1.Day, 
                            item.Item1.Hour, 
                            item.Item1.Minute, 
                            0), // Round down to minute
                        value: item.Item2))
                    .GroupBy(item => item.timestamp)
                    .Select(group => (
                        timestamp: group.Key,
                        value: group.Average(g => g.value))) // Average values within the same minute
                    .OrderBy(item => item.timestamp)
                    .ToList();

                alignedStreams[stream.Key] = alignedData;
                _logger.LogInformation("Stream {StreamName}: {OriginalCount} → {AlignedCount} records after timestamp alignment", 
                    stream.Key, stream.Value.Count(), alignedData.Count);
            }

            // Find common timestamps across all streams
            var commonTimestamps = alignedStreams.Values
                .Select(stream => stream.Select(item => item.Item1).ToHashSet())
                .Aggregate((set1, set2) => 
                {
                    set1.IntersectWith(set2);
                    return set1;
                })
                .OrderBy(ts => ts)
                .ToList();

            _logger.LogInformation("Found {Count} common timestamps after alignment", commonTimestamps.Count);

            // Filter each stream to only include common timestamps
            var finalStreams = new Dictionary<string, IEnumerable<(DateTime, double)>>();
            foreach (var stream in alignedStreams)
            {
                var filteredData = stream.Value
                    .Where(item => commonTimestamps.Contains(item.Item1))
                    .ToList();

                finalStreams[stream.Key] = filteredData;
                _logger.LogInformation("Stream {StreamName}: {FilteredCount} records after common timestamp filtering", 
                    stream.Key, filteredData.Count);
            }

            if (commonTimestamps.Count == 0)
            {
                _logger.LogWarning("No common timestamps found across all streams after alignment");
            }

            alignStopwatch.Stop();
            _logger.LogInformation("Timestamp alignment completed in {ElapsedMs}ms", alignStopwatch.ElapsedMilliseconds);

            return finalStreams;
        }

        // ─── Helper: Convert stream key to tag name ─────────────────────────────────────
        private static string ConvertStreamKeyToTagName(string streamKey)
        {
            return streamKey switch
            {
                "BatteryVoltage" => "Battery-Voltage",
                "UpstreamPressure" => "Upstream-Pressure",
                "DownstreamPressure" => "Downstream-Pressure",
                "UpstreamTemperature" => "Upstream-Temperature",
                "DownstreamTemperature" => "Downstream-Temperature",
                "ChokePosition" => "Choke-Position",
                "TargetPosition" => "Target-Position",
                "ToolState" => "Tool-State",
                "DownstreamUpstreamDifference" => "Downstream-Upstream-Difference",
                _ => streamKey.Replace("_", "-")
            };
        }

        // ─── POST /Anomaly/Detect ──────────────────────────────────────────
        [HttpPost("Detect")]
        public async Task<IActionResult> Detect(List<IFormFile> files)
        {
            var overallStopwatch = Stopwatch.StartNew();
            _logger.LogInformation("=== ANOMALY DETECTION START (CSV) === Files: {Count}", files.Count);

            if (files.Count == 0)
            {
                TempData["Msg"] = "No file selected.";
                return RedirectToAction("Upload");
            }

            if (UseMLApi)
            {
                // Ensure ML service is running
                if (!await EnsureMLServiceRunningAsync(60))
                {
                    TempData["Msg"] = "ML service could not be started. Please check your Python environment.";
                    return RedirectToAction("Upload");
                }
                // Validate required files
                var requiredFiles = new []
                {
                    "Tool.P8-36.SIT.Battery-Voltage.csv",
                    "Tool.P8-36.SIT.Choke-Position.csv",
                    "Tool.P8-36.SIT.Downstream-Pressure.csv",
                    "Tool.P8-36.SIT.Downstream-Temperature.csv",
                    "Tool.P8-36.SIT.Downstream-Upstream-Difference.csv",
                    "Tool.P8-36.SIT.Target-Position.csv",
                    "Tool.P8-36.SIT.Tool-State.csv",
                    "Tool.P8-36.SIT.Upstream-Pressure.csv",
                    "Tool.P8-36.SIT.Upstream-Temperature.csv"
                };
                var uploadedFileNames = files.Select(f => f.FileName).ToList();
                var missingFiles = requiredFiles.Except(uploadedFileNames).ToList();
                if (missingFiles.Any())
                {
                    TempData["Msg"] = $"Missing required files: {string.Join(", ", missingFiles)}";
                    return RedirectToAction("Upload");
                }
                try
                {
                    var mlResults = await _mlServiceClient.RunAllModelsAnalysisAsync(files);
                    if (mlResults.FailedModels?.Any() == true)
                    {
                        foreach (var error in mlResults.Errors)
                        {
                            _logger.LogWarning($"Model {error.Key} failed: {error.Value}");
                        }
                    }
                    // Transform API results to dashboard format
                    var dashboardData = TransformToDashboardData(mlResults);
                    // TODO: Map dashboardData to ViewBag and model for Detect.cshtml
                    // For now, just show a summary page
                    ViewBag.ApiResults = dashboardData;
                    ViewBag.TotalAnomalies = mlResults.Predictions.Values.Sum(p => p.AnomalyCount);
                    ViewBag.ProcessedModels = mlResults.SuccessfulModels;
                    ViewBag.FailedModels = mlResults.FailedModels;
                    ViewBag.ProcessingTimeMs = mlResults.Predictions.Values.Sum(p => p.Timings.TotalMs);
                    // You may want to map anomalies to the expected model for Detect.cshtml
                    // For now, just pass the anomalies from all models
                    var allAnomalies = mlResults.Predictions.Values.SelectMany(p => p.Anomalies).ToList();
                    return View("Detect", allAnomalies);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error during ML API analysis");
                    TempData["Msg"] = $"ML API error: {ex.Message}";
                    return RedirectToAction("Upload");
                }
            }

            // --------------------------------------------------------------------
            // 1.  Process each upload directly in memory and build tag ⇒ stream map
            // --------------------------------------------------------------------
            var fileProcessingStopwatch = Stopwatch.StartNew();
            var tagStreams = new Dictionary<string, IEnumerable<(DateTime, double)>>();

            foreach (var f in files)
            {
                var fileStopwatch = Stopwatch.StartNew();
                
                try
                {
                    // Process file directly from memory stream to avoid file lock issues
                    var tagName = InferTagName(f.FileName);
                    
                    using var memoryStream = new MemoryStream();
                    await f.CopyToAsync(memoryStream);
                    memoryStream.Position = 0; // Reset position to beginning
                    
                    // Read CSV data directly from memory stream
                    var csvData = ReadCsvFromStream(memoryStream).ToList(); // Materialize to avoid multiple enumeration
                    tagStreams[tagName] = csvData;
                    
                    fileStopwatch.Stop();
                    _logger.LogInformation("Processed file {FileName} → {TagName}, {Count} rows in {ElapsedMs}ms", 
                        f.FileName, tagName, csvData.Count, fileStopwatch.ElapsedMilliseconds);
                }
                catch (Exception ex)
                {
                    fileStopwatch.Stop();
                    _logger.LogError(ex, "Error processing file {FileName} after {ElapsedMs}ms", f.FileName, fileStopwatch.ElapsedMilliseconds);
                    TempData["Msg"] = $"Error processing file {f.FileName}: {ex.Message}";
                    return RedirectToAction("Upload");
                }
            }
            
            fileProcessingStopwatch.Stop();
            _logger.LogInformation("File processing completed in {ElapsedMs}ms, {StreamCount} streams loaded", 
                fileProcessingStopwatch.ElapsedMilliseconds, tagStreams.Count);

            return await ProcessAnomalyData(tagStreams, isDataHubData: false, overallStopwatch);
        }

        // ─── Helper: Read CSV data directly from stream ─────────────────────────────────────
        private static List<(DateTime ts, double val)> ReadCsvFromStream(Stream stream)
        {
            var results = new List<(DateTime ts, double val)>();
            var cfg = new CsvHelper.Configuration.CsvConfiguration(System.Globalization.CultureInfo.InvariantCulture)
            {
                HasHeaderRecord = true,
                BadDataFound = null,   // ignore junk lines
            };

            using var reader = new StreamReader(stream);
            using var csv = new CsvHelper.CsvReader(reader, cfg);

            csv.Read();
            csv.ReadHeader(); // skip header line

            while (csv.Read())
            {
                try
                {
                    var ts = csv.GetField<DateTime>(0);
                    var val = csv.GetField<double>(1);
                    results.Add((ts, val));
                }
                catch (Exception)
                {
                    // Skip invalid rows
                    continue;
                }
            }

            return results;
        }

        // ─── Helper: Process anomaly data (common for both CSV and DataHub) ─────────────────────────────────────
        private async Task<IActionResult> ProcessAnomalyData(Dictionary<string, IEnumerable<(DateTime, double)>> tagStreams, bool isDataHubData = false, Stopwatch? overallTimer = null)
        {
            var processingStopwatch = Stopwatch.StartNew();
            _logger.LogInformation("Starting anomaly data processing (DataHub: {IsDataHub}), {StreamCount} streams", 
                isDataHubData, tagStreams.Count);

            try
            {
                // Apply timestamp alignment for DataHub data
                if (isDataHubData)
                {
                    tagStreams = AlignTimestamps(tagStreams);
                    
                    if (!tagStreams.Any() || !tagStreams.Values.Any(stream => stream.Any()))
                    {
                        processingStopwatch.Stop();
                        overallTimer?.Stop();
                        _logger.LogError("No aligned data available after timestamp processing (total time: {TotalMs}ms, processing: {ProcessingMs}ms)", 
                            overallTimer?.ElapsedMilliseconds ?? 0, processingStopwatch.ElapsedMilliseconds);
                        TempData["Msg"] = "No aligned data available after timestamp processing. The streams may not have overlapping time periods.";
                        return RedirectToAction("Upload");
                    }
                }

                // --------------------------------------------------------------------
                // 2.  Join on timestamp
                // --------------------------------------------------------------------
                var joinStopwatch = Stopwatch.StartNew();
                var joined = MultiCsvJoiner.Join(tagStreams);
                var joinedList = joined.ToList(); // Materialize to avoid multiple enumeration
                joinStopwatch.Stop();
                
                var first = joinedList.FirstOrDefault(); // get first row for logging
                if (first != null)
                    _logger.LogInformation("Data join completed in {ElapsedMs}ms. First joined row has {Cnt} tags: {Names}",
                        joinStopwatch.ElapsedMilliseconds, first.Values.Count, string.Join(", ", first.Values.Keys));

                // --------------------------------------------------------------------
                // 3. OPTIMIZED: Process rows and run anomaly detection with batching
                // --------------------------------------------------------------------
                var detectionStopwatch = Stopwatch.StartNew();
                var anomalies = new List<taqa.polaris.sit.Services.Anomaly.AnomalyEvent>();

                var recentRows = new Queue<JoinedRow>();
                var rowsOk = new List<JoinedRow>();  // for ML scoring
                var dqTimeline = new List<IDictionary<string, DqState>>();
                var badCounts = new Dictionary<string, int>();
                var allTags = joinedList.First().Values.Keys.ToHashSet(StringComparer.OrdinalIgnoreCase);
                int totalRows = 0;
                var lastProgressLog = Stopwatch.StartNew();

                // OPTIMIZATION: Process DQ in larger batches for better performance
                const int dqBatchSize = 100;
                var dqBatch = new List<JoinedRow>();

                foreach (var row in joinedList)
                {
                    totalRows++;
                    recentRows.Enqueue(row);
                    if (recentRows.Count > 5) recentRows.Dequeue();

                    dqBatch.Add(row);
                    
                    // Process DQ in batches instead of one by one
                    if (dqBatch.Count >= dqBatchSize || totalRows == joinedList.Count)
                    {
                        foreach (var batchRow in dqBatch)
                        {
                            var dqMap = DataQualityWatchdog.ScanWindow(recentRows.ToList());
                            dqTimeline.Add(dqMap);

                            foreach (var (tag, state) in dqMap)
                                if (state != DqState.Good)
                                    badCounts[tag] = badCounts.GetValueOrDefault(tag, 0) + 1;

                            var breaches = DataQualityWatchdog.GetBreaches(recentRows.ToList()).ToList();
                            if (breaches.Any())
                            {
                                foreach (var b in breaches)
                                    // Fix ambiguous reference by using full namespace for local AnomalyEvent
                                    anomalies.Add(new taqa.polaris.sit.Services.Anomaly.AnomalyEvent(
                                         batchRow.Timestamp, $"dq_{b.tag.ToLower()}_{b.rule}",
                                         batchRow.Values[b.tag], double.NaN, 0,
                                         b.state == DqState.Bad ? "High" : "Medium"));
                            }
                            else
                            {
                                rowsOk.Add(batchRow);            // keep for batched ML
                            }
                        }
                        dqBatch.Clear();
                    }
                    
                    // Progress logging every 5000 rows or every 10 seconds for better performance
                    if (totalRows % 5000 == 0 || lastProgressLog.ElapsedMilliseconds > 10000)
                    {
                        _logger.LogInformation("Processed {Rows}/{Total} rows ({Percent:F1}%)…", 
                            totalRows, joinedList.Count, (double)totalRows / joinedList.Count * 100);
                        lastProgressLog.Restart();
                    }
                }

                _logger.LogInformation("Row processing completed: {TotalRows} rows, {OkRows} rows for ML, {DqAnomalies} DQ anomalies", 
                    totalRows, rowsOk.Count, anomalies.Count);

                // OPTIMIZATION: Run ML scoring with larger batch sizes for better GPU/CPU utilization
                var mlStopwatch = Stopwatch.StartNew();
                long memBefore = GC.GetTotalMemory(false);
                var cpuBefore = Process.GetCurrentProcess().TotalProcessorTime;
                var mlAnomalies = _router.ScoreBatch(rowsOk, batchSize: 2048); // Increased from default 1024
                long memAfter = GC.GetTotalMemory(false);
                var cpuAfter = Process.GetCurrentProcess().TotalProcessorTime;
                mlStopwatch.Stop();
                
                // Metrics calculation
                var inferenceLatencyMs = mlStopwatch.ElapsedMilliseconds;
                var throughput = rowsOk.Count > 0 && inferenceLatencyMs > 0 ? (rowsOk.Count * 1000.0 / inferenceLatencyMs) : 0;
                var memoryUsedMb = (memAfter - memBefore) / 1024.0 / 1024.0;
                var cpuUsedMs = (cpuAfter - cpuBefore).TotalMilliseconds;
                
                // Pass metrics to view
                ViewBag.InferenceLatencyMs = inferenceLatencyMs;
                ViewBag.Throughput = throughput;
                ViewBag.MemoryUsedMb = memoryUsedMb;
                ViewBag.CpuUsedMs = cpuUsedMs;
                
                anomalies.AddRange(mlAnomalies);
                _logger.LogInformation("ML scoring completed in {ElapsedMs}ms, {MlAnomalies} ML anomalies detected", 
                    mlStopwatch.ElapsedMilliseconds, mlAnomalies.Count());

                detectionStopwatch.Stop();
                _logger.LogInformation("Anomaly detection completed in {ElapsedMs}ms, total anomalies: {Count}", 
                    detectionStopwatch.ElapsedMilliseconds, anomalies.Count);

                foreach (var t in allTags)               // fill missing tags
                    if (!badCounts.ContainsKey(t))
                        badCounts[t] = 0;

                // ❶ Batch health ratio per tag
                var batchHealth = badCounts.ToDictionary(
                    kv => kv.Key,
                    kv => (double)kv.Value / totalRows);          // e.g. 0.12 ⇒ 12 %

                // ❂ Decide colour state from ratio
                var batchState = batchHealth.ToDictionary(
                    kv => kv.Key,
                    kv => kv.Value >= 0.10 ? 2        // ≥10 % bad = red
                           : kv.Value >= 0.01 ? 1      // 1-10 %    = yellow
                           : 0);                      // <1 %      = green

                
                // Pass to view
                ViewBag.BatchState = batchState;      // dict<tag,int>
                ViewBag.Summary = anomalies.GroupBy(a => a.Detector)
                                              .ToDictionary(g => g.Key, g => g.Count());
                var ordered = anomalies.OrderBy(a => a.Timestamp).ToList();

                // --------------------------------------------------------------------
                //  4. OPTIMIZED: Build chart data with parallel processing for large datasets
                // --------------------------------------------------------------------
                var chartDataStopwatch = Stopwatch.StartNew();
                
                // OPTIMIZATION: Use parallel processing for large datasets
                object severityMix;
                if (anomalies.Count > 10000)
                {
                    severityMix = anomalies
                        .AsParallel()
                        .GroupBy(a => a.Detector)
                        .OrderByDescending(g => g.Count())
                        .Take(8)  // Top 8 detectors
                        .ToDictionary(
                            g => g.Key,
                            g => new
                            {
                                Low = g.Count(a => a.Severity == "Low"),
                                Medium = g.Count(a => a.Severity == "Medium"),
                                High = g.Count(a => a.Severity == "High")
                            }
                        );
                }
                else
                {
                    severityMix = anomalies
                        .GroupBy(a => a.Detector)
                        .OrderByDescending(g => g.Count())
                        .Take(8)  // Top 8 detectors
                        .ToDictionary(
                            g => g.Key,
                            g => new
                            {
                                Low = g.Count(a => a.Severity == "Low"),
                                Medium = g.Count(a => a.Severity == "Medium"),
                                High = g.Count(a => a.Severity == "High")
                            }
                        );
                }

                ViewBag.SeverityMix = severityMix;

                /* -----------------------------------------------------------------
                 *  5. OPTIMIZED: Build chart payload with efficient grouping
                 * ----------------------------------------------------------------*/
                var chartBuckets = ordered
                    .GroupBy(a => a.Timestamp                           // round ↓ to minute
                                     .ToUniversalTime()
                                     .AddSeconds(-a.Timestamp.Second)
                                     .AddMilliseconds(-a.Timestamp.Millisecond))
                    .OrderBy(g => g.Key)
                    .Select(g =>
                    {
                        int dq = g.Count(x => x.Detector.StartsWith("dq_",
                                          StringComparison.OrdinalIgnoreCase));
                        int ml = g.Count() - dq;
                        return new { t = g.Key.ToString("yyyy-MM-dd HH:mm"), dq, ml };
                    }).ToList();

                ViewBag.Chart = chartBuckets;

                // --------------------------------------------------------------------
                //  6. OPTIMIZED: Build drill-down data with better memory management
                // --------------------------------------------------------------------

                double Clean(double x) => double.IsFinite(x) ? x : 0.0;    // or null

                var drill = new Dictionary<string, List<object>>();

                // OPTIMIZATION: Pre-allocate capacity for better performance
                foreach (var detectorGroup in anomalies.GroupBy(a => a.Detector))
                {
                    var list = new List<object>(detectorGroup.Count());
                    
                    foreach (var a in detectorGroup)
                    {
                        var row = new
                        {
                            t = a.Timestamp.ToString("o"),
                            raw = Clean(a.RawValue),
                            obs = Clean(a.Observed ?? double.NaN),
                            pred = Clean(a.Predicted ?? double.NaN),
                            score = Clean(a.Score),
                            thr = Clean(a.Threshold)
                        };
                        list.Add(row);
                    }
                    
                    drill[detectorGroup.Key] = list;
                }

                ViewBag.Drill = System.Text.Json.JsonSerializer.Serialize(drill);
                
                chartDataStopwatch.Stop();
                processingStopwatch.Stop();
                overallTimer?.Stop();
                
                _logger.LogInformation("Chart data preparation completed in {ElapsedMs}ms", chartDataStopwatch.ElapsedMilliseconds);
                _logger.LogInformation("=== ANOMALY DETECTION COMPLETE === Total time: {TotalMs}ms, Processing: {ProcessingMs}ms, {AnomalyCount} anomalies", 
                    overallTimer?.ElapsedMilliseconds ?? processingStopwatch.ElapsedMilliseconds, 
                    processingStopwatch.ElapsedMilliseconds, ordered.Count);
                
                return View("Detect", ordered);
            }
            catch (Exception ex)
            {
                processingStopwatch.Stop();
                overallTimer?.Stop();
                _logger.LogError(ex, "Error in ProcessAnomalyData after {ProcessingMs}ms (total: {TotalMs}ms)", 
                    processingStopwatch.ElapsedMilliseconds, overallTimer?.ElapsedMilliseconds ?? 0);
                throw;
            }
        }

        // ─── GET /Anomaly/Status - Simple diagnostic endpoint ────────────────────────────────────
        [HttpGet("Status")]
        public IActionResult Status()
        {
            var status = new
            {
                Timestamp = DateTime.UtcNow,
                Server = Environment.MachineName,
                Memory = new
                {
                    WorkingSet = GC.GetTotalMemory(false),
                    WorkingSetMB = Math.Round(GC.GetTotalMemory(false) / 1024.0 / 1024.0, 2)
                },
                Performance = new
                {
                    ProcessorCount = Environment.ProcessorCount,
                    UpTime = Environment.TickCount64 / 1000 // seconds
                }
            };
            
            return Json(status);
        }

        // ─── Test endpoint to verify routing ────────────────────────────────────
        [HttpGet("TestReport")]
        public IActionResult TestReport()
        {
            return Json(new { 
                Message = "Report endpoint is working", 
                Timestamp = DateTime.UtcNow,
                ControllerName = "AnomalyController",
                ActionName = "TestReport",
                Routes = new {
                    GetTest = "/Anomaly/TestReport",
                    PostTest = "/Anomaly/TestReport (POST)",
                    GenerateReport = "/Anomaly/GenerateReport (POST)",
                    Status = "/Anomaly/Status"
                }
            });
        }

        [HttpPost("TestReport")]
        public IActionResult TestReportPost([FromBody] object data)
        {
            return Json(new { 
                Message = "POST endpoint is working", 
                Timestamp = DateTime.UtcNow,
                DataReceived = data != null,
                DataContent = data?.ToString(),
                ControllerName = "AnomalyController",
                ActionName = "TestReportPost"
            });
        }

        // ─── Troubleshooting endpoint ────────────────────────────────────
        [HttpGet("Debug")]
        public IActionResult Debug()
        {
            var debugInfo = new
            {
                Timestamp = DateTime.UtcNow,
                Server = Environment.MachineName,
                Environment = _env.EnvironmentName,
                ContentRoot = _env.ContentRootPath,
                WebRoot = _env.WebRootPath,
                ControllerType = GetType().FullName,
                AvailableEndpoints = new[]
                {
                    "GET /Anomaly/Upload",
                    "POST /Anomaly/Detect",
                    "POST /Anomaly/DetectFromDataHub", 
                    "POST /Anomaly/GenerateReport",
                    "GET /Anomaly/Status",
                    "GET /Anomaly/TestReport",
                    "POST /Anomaly/TestReport",
                    "GET /Anomaly/Debug"
                },
                QuestPDFLicense = QuestPDF.Settings.License.ToString(),
                Memory = new
                {
                    WorkingSet = GC.GetTotalMemory(false),
                    WorkingSetMB = Math.Round(GC.GetTotalMemory(false) / 1024.0 / 1024.0, 2)
                }
            };
            
            return Json(debugInfo);
        }

        // ─── Helper: Clean up temporary files ─────────────────────────────────────
        private void CleanupTempFiles()
        {
            try
            {
                var tempDir = System.IO.Path.Combine(_env.ContentRootPath, "TempUploads");
                if (Directory.Exists(tempDir))
                {
                    var files = Directory.GetFiles(tempDir);
                    var deletedCount = 0;
                    
                    foreach (var file in files)
                    {
                        try
                        {
                            // Only delete files older than 1 hour to avoid deleting active files
                            var fileInfo = new FileInfo(file);
                            if (fileInfo.CreationTime < DateTime.Now.AddHours(-1))
                            {
                                System.IO.File.Delete(file);
                                deletedCount++;
                            }
                        }
                        catch (Exception ex)
                        {
                            _logger.LogWarning(ex, "Failed to delete temp file {FilePath}", file);
                        }
                    }
                    
                    if (deletedCount > 0)
                    {
                        _logger.LogInformation("Cleaned up {Count} old temporary files", deletedCount);
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Error during temp file cleanup");
            }
        }

        // ─── Helper: Infer tag name from filename ─────────────────────────────────────
        // crude helper: "Tool.P8-41.SIT.Battery-Voltage.csv" → "Battery-Voltage"
        private static string InferTagName(string file) =>
            file.Split('.', StringSplitOptions.RemoveEmptyEntries)[^2]
                .Replace('_', ' ').Trim();

        // ─── Helper methods for professional formatting ─────────────────────────────────────
        private static int CalculateSystemHealth(Dictionary<string, int>? batchState)
        {
            if (batchState == null || !batchState.Any()) return 100;
            
            var totalSensors = batchState.Count;
            var healthySensors = batchState.Count(kv => kv.Value == 0);
            return (int)Math.Round((double)healthySensors / totalSensors * 100);
        }

        private static string DetermineRiskLevel(int totalAnomalies, int systemHealth)
        {
            if (totalAnomalies > 1000 || systemHealth < 70) return "HIGH";
            if (totalAnomalies > 100 || systemHealth < 90) return "MEDIUM";
            return "LOW";
        }

        private static string FormatDetectorName(string detector)
        {
            if (string.IsNullOrEmpty(detector)) return "";
            
            return detector
                .Replace("dq_", "DQ: ")
                .Replace("residual_", "")
                .Replace("_", " ")
                .Replace("-", " ")
                .Split(' ')
                .Select(word => word.Length > 0 ? char.ToUpper(word[0]) + (word.Length > 1 ? word[1..].ToLower() : "") : "")
                .Where(word => !string.IsNullOrEmpty(word))
                .Aggregate((a, b) => a + " " + b);
        }

        private static string FormatSensorName(string sensor)
        {
            if (string.IsNullOrEmpty(sensor)) return "";
            
            return sensor
                .Replace("-", " ")
                .Replace("_", " ")
                .Split(' ')
                .Select(word => word.Length > 0 ? char.ToUpper(word[0]) + (word.Length > 1 ? word[1..].ToLower() : "") : "")
                .Where(word => !string.IsNullOrEmpty(word))
                .Aggregate((a, b) => a + " " + b);
        }

        private static string FormatTimestamp(string timestamp)
        {
            if (DateTime.TryParse(timestamp, out var dt))
            {
                return dt.ToString("MM/dd HH:mm");
            }
            return timestamp;
        }

        private static (string Status, string Percentage, Color Color) GetHealthInfo(int healthValue)
        {
            return healthValue switch
            {
                0 => ("OPTIMAL", "99%+", Colors.Green.Medium),
                1 => ("GOOD", "95-99%", Colors.Orange.Medium),
                _ => ("DEGRADED", "<95%", Colors.Red.Medium)
            };
        }

        private static (string, Color) GetSeverityLevel(int count)
        {
            return count switch
            {
                > 500 => ("CRITICAL", Colors.Red.Medium),
                > 100 => ("HIGH", Colors.Orange.Medium),
                > 10 => ("MEDIUM", Colors.Blue.Medium),
                _ => ("LOW", Colors.Green.Medium)
            };
        }

        // Helper method to generate dynamic recommendations
        private static List<string> GenerateRecommendations(AnomalyReportData reportData, int totalAnomalies, int systemHealth, string riskLevel)
        {
            var recommendations = new List<string>();

            // Risk-based recommendations
            switch (riskLevel)
            {
                case "HIGH":
                    recommendations.Add("IMMEDIATE ACTION REQUIRED: Conduct emergency system inspection within 24 hours");
                    recommendations.Add("Implement emergency monitoring protocols for all critical sensors");
                    recommendations.Add("Multiple simultaneous failure patterns detected - investigate root cause");
                    break;
                case "MEDIUM":
                    recommendations.Add("Schedule comprehensive system inspection within 72 hours");
                    recommendations.Add("Increase monitoring frequency for affected sensors");
                    recommendations.Add("Trend analysis indicates potential degradation - preventive action recommended");
                    break;
                case "LOW":
                    recommendations.Add("Continue routine monitoring and maintenance schedule");
                    recommendations.Add("System operating within acceptable parameters");
                    break;
            }

            // System health recommendations
            if (systemHealth < 70)
            {
                recommendations.Add("Multiple sensors showing degraded performance - consider system-wide maintenance");
            }
            else if (systemHealth < 90)
            {
                recommendations.Add("Monitor sensor health trends and plan preventive maintenance");
            }

            // Detector-specific recommendations
            if (reportData.SummaryData != null)
            {
                var topIssues = reportData.SummaryData.Where(kv => kv.Value > 0).OrderByDescending(kv => kv.Value).Take(3);
                
                foreach (var issue in topIssues)
                {
                    var detectorRecommendation = GetDetectorSpecificRecommendation(issue.Key, issue.Value);
                    if (!string.IsNullOrEmpty(detectorRecommendation))
                    {
                        recommendations.Add(detectorRecommendation);
                    }
                }
            }

            // Sensor health recommendations
            if (reportData.BatchState != null)
            {
                var degradedSensors = reportData.BatchState.Where(kv => kv.Value >= 2).ToList();
                if (degradedSensors.Any())
                {
                    var sensorNames = degradedSensors.Select(s => FormatSensorName(s.Key)).Take(3);
                    recommendations.Add($"Priority maintenance required for: {string.Join(", ", sensorNames)}");
                }
            }

            // Anomaly volume recommendations
            if (totalAnomalies > 1000)
            {
                recommendations.Add("High anomaly volume detected - review detection thresholds and investigate root causes");
            }
            else if (totalAnomalies > 100)
            {
                recommendations.Add("Moderate anomaly activity - review trending patterns and adjust monitoring parameters");
            }

            // Data quality recommendations
            var dqIssues = reportData.SummaryData?.Where(kv => kv.Key.StartsWith("dq_", StringComparison.OrdinalIgnoreCase)).Sum(kv => kv.Value) ?? 0;
            if (dqIssues > totalAnomalies * 0.3) // More than 30% are data quality issues
            {
                recommendations.Add("Significant data quality issues detected - review sensor calibration and data collection systems");
            }

            // Ensure we always have at least basic recommendations
            if (!recommendations.Any())
            {
                recommendations.Add("System operating within normal parameters - maintain current monitoring protocols");
                recommendations.Add("Continue regular preventive maintenance schedule");
            }

            return recommendations.Take(8).ToList(); // Limit to 8 recommendations for readability
        }

        // Helper method for detector-specific recommendations
        private static string GetDetectorSpecificRecommendation(string detector, int count)
        {
            var lowerDetector = detector.ToLower();
            
            return lowerDetector switch
            {
                var d when d.Contains("battery") && d.Contains("voltage") => 
                    $"Battery voltage anomalies ({count}) - Check power supply and battery health",
                    
                var d when d.Contains("pressure") && d.Contains("upstream") => 
                    $"Upstream pressure issues ({count}) - Inspect upstream sensors and pressure lines",
                    
                var d when d.Contains("pressure") && d.Contains("downstream") => 
                    $"Downstream pressure anomalies ({count}) - Check downstream system and flow restrictions",
                    
                var d when d.Contains("temperature") => 
                    $"Temperature anomalies ({count}) - Verify thermal management and sensor calibration",
                    
                var d when d.Contains("choke") && d.Contains("position") => 
                    $"Choke position irregularities ({count}) - Inspect choke mechanism and position sensors",
                    
                var d when d.Contains("target") && d.Contains("position") => 
                    $"Target position deviations ({count}) - Review control system and position feedback",
                    
                var d when d.Contains("tool") && d.Contains("state") => 
                    $"Tool state anomalies ({count}) - Check tool operational status and control systems",
                    
                var d when d.Contains("residual") => 
                    $"Model prediction deviations ({count}) - Review {FormatDetectorName(detector)} sensor accuracy",
                    
                _ => count > 50 ? $"High anomaly count in {FormatDetectorName(detector)} ({count}) - Requires investigation" : ""
            };
        }

        // ─── POST /Anomaly/GenerateReport ──────────────────────────────────────────
        [HttpPost("GenerateReport")]
        public async Task<IActionResult> GenerateReport([FromBody] AnomalyReportData reportData)
        {
            var reportStopwatch = Stopwatch.StartNew();
            
            try
            {
                _logger.LogInformation("=== PDF REPORT GENERATION START === Received report generation request");
                
                if (reportData == null)
                {
                    _logger.LogError("Report data is null");
                    return BadRequest(new { error = "Invalid report data", details = "No data received" });
                }

                _logger.LogInformation("Report data validation - Total anomalies: {Total}, Table rows: {Rows}, Charts: Severity={SeveritySize}, Timeline={TimelineSize}", 
                    reportData.TotalAnomalies, 
                    reportData.TableData?.Count ?? 0,
                    reportData.SeverityChart?.Length ?? 0,
                    reportData.TimelineChart?.Length ?? 0);

                // Calculate key metrics
                var totalAnomalies = reportData.TotalAnomalies;
                var criticalAnomalies = reportData.SummaryData?.Values.Sum() ?? 0;
                var systemHealth = CalculateSystemHealth(reportData.BatchState);
                var riskLevel = DetermineRiskLevel(totalAnomalies, systemHealth);

                // Deployment metrics from reportData (fix for PDF)
                double? inferenceLatencyMs = reportData.InferenceLatencyMs;
                double? throughput = reportData.Throughput;
                double? memoryUsedMb = reportData.MemoryUsedMb;
                double? cpuUsedMs = reportData.CpuUsedMs;
                
                _logger.LogInformation("Report metrics calculated - Health: {Health}%, Risk: {Risk}, Critical: {Critical}",
                    systemHealth, riskLevel, criticalAnomalies);

                // Define TAQA brand colors based on newsletter
                var taqaTeal = "#00A6A6";        // Primary teal from TAQA logo
                var taqaDarkTeal = "#008080";    // Darker teal for headers
                var taqaOrange = "#FF6B35";      // Orange accent color
                var taqaLightTeal = "#E0F2F1";  // Light teal background
                var taqaNeutralGray = "#546E7A"; // Professional gray

                // Define reusable colors for helper functions
                var headerTeal = taqaDarkTeal;
                var cellBorder = "#E0E0E0";

                // Generate PDF using QuestPDF with TAQA branding
                byte[] pdfBytes;
                
                try
                {
                    _logger.LogInformation("Starting PDF generation with TAQA brand styling...");
                    
                    pdfBytes = Document.Create(container =>
                    {
                        container.Page(page =>
                        {
                            page.Size(PageSizes.A4);
                            page.Margin(1.5f, Unit.Centimetre);
                            page.PageColor(Colors.White);
                            page.DefaultTextStyle(x => x.FontFamily("Arial").FontSize(11));

                            // TAQA-branded header
                            page.Header().Row(row =>
                            {
                                row.RelativeItem().Column(column =>
                                {
                                    // TAQA logo section
                                    var logoPath = Path.Combine(_env.ContentRootPath, "wwwroot", "img", "logo-tq.jpg");
                                    if (System.IO.File.Exists(logoPath))
                                    {
                                        try
                                        {
                                            column.Item().Width(100).Image(logoPath);
                                        }
                                        catch (Exception ex)
                                        {
                                            _logger.LogWarning(ex, "Could not load logo from {LogoPath}", logoPath);
                                            column.Item().Text("TAQA").FontSize(28).SemiBold()
                                                .FontColor(taqaDarkTeal);
                                        }
                                    }
                                    else
                                    {
                                        column.Item().Text("TAQA").FontSize(28).SemiBold()
                                            .FontColor(taqaDarkTeal);
                                    }
                                });

                                row.RelativeItem().Column(column =>
                                {
                                    column.Item().AlignRight().Text("SYSTEMS INTEGRITY")
                                        .FontSize(14).SemiBold().FontColor(taqaNeutralGray);
                                    column.Item().AlignRight().Text("Anomaly Detection Report")
                                        .FontSize(22).SemiBold().FontColor(taqaDarkTeal);
                                    column.Item().AlignRight().Text($"Generated: {reportData.GeneratedAt}")
                                        .FontSize(10).FontColor(taqaNeutralGray);
                                    column.Item().AlignRight().Text("CONFIDENTIAL")
                                        .FontSize(10).SemiBold().FontColor(taqaOrange);
                                });
                            });

                            page.Content()
                                .PaddingVertical(1, Unit.Centimetre)
                                .Column(x =>
                                {
                                    x.Spacing(20);

                                    // Executive Summary with TAQA styling
                                    x.Item().Background(taqaLightTeal).BorderLeft(4, Unit.Point).BorderColor(taqaTeal)
                                        .Padding(20).Column(exec =>
                                        {
                                            exec.Item().Text("EXECUTIVE SUMMARY").FontSize(16).SemiBold().FontColor(taqaDarkTeal);
                                            exec.Item().PaddingTop(10).Row(row =>
                                            {
                                                row.RelativeItem().Column(col =>
                                                {
                                                    col.Item().Text($"Total Anomalies: {totalAnomalies:N0}").FontSize(14).SemiBold().FontColor(taqaNeutralGray);
                                                    col.Item().Text($"System Health: {systemHealth}%").FontSize(14).SemiBold()
                                                        .FontColor(systemHealth >= 90 ? "#2E7D32" : 
                                                                  systemHealth >= 70 ? taqaOrange : "#C62828");
                                                });
                                                row.RelativeItem().Column(col =>
                                                {
                                                    col.Item().Text($"Risk Level: {riskLevel}").FontSize(14).SemiBold()
                                                        .FontColor(riskLevel == "LOW" ? "#2E7D32" :
                                                                  riskLevel == "MEDIUM" ? taqaOrange : "#C62828");
                                                    col.Item().Text($"Critical Issues: {criticalAnomalies:N0}").FontSize(12)
                                                        .FontColor(taqaNeutralGray);
                                                });
                                            });

                                            // Deployment Metrics Section
                                            exec.Item().PaddingTop(15).Text("MODEL DEPLOYMENT METRICS").FontSize(13).SemiBold().FontColor(taqaDarkTeal);
                                            exec.Item().Row(row =>
                                            {
                                                row.RelativeItem().Text($"Inference Latency: {(inferenceLatencyMs.HasValue ? inferenceLatencyMs.Value.ToString("N0") : "-")} ms").FontSize(12).FontColor(taqaNeutralGray);
                                                row.RelativeItem().Text($"Throughput: {(throughput.HasValue ? throughput.Value.ToString("N1") : "-")} events/sec").FontSize(12).FontColor(taqaNeutralGray);
                                                row.RelativeItem().Text($"Memory Used: {(memoryUsedMb.HasValue ? memoryUsedMb.Value.ToString("N1") : "-")} MB").FontSize(12).FontColor(taqaNeutralGray);
                                                row.RelativeItem().Text($"CPU Time: {(cpuUsedMs.HasValue ? cpuUsedMs.Value.ToString("N0") : "-")} ms").FontSize(12).FontColor(taqaNeutralGray);
                                            });
                                        });

                                    // Key Performance Indicators with TAQA card styling
                                    x.Item().Text("KEY PERFORMANCE INDICATORS").FontSize(16).SemiBold().FontColor(taqaDarkTeal);
                                    x.Item().Row(row =>
                                    {
                                        if (reportData.SummaryData != null && reportData.SummaryData.Any())
                                        {
                                            var topDetectors = reportData.SummaryData.OrderByDescending(kv => kv.Value).Take(3);
                                            foreach (var detector in topDetectors)
                                            {
                                                row.RelativeItem().Background("#F8F9FA").Border(1).BorderColor(taqaTeal)
                                                    .Padding(15).Column(col =>
                                                {
                                                    col.Item().Text(detector.Value.ToString("N0")).FontSize(20).SemiBold().FontColor(taqaOrange);
                                                    col.Item().Text("Anomalies").FontSize(12).FontColor(taqaNeutralGray);
                                                    col.Item().Text(FormatDetectorName(detector.Key)).FontSize(10).SemiBold().FontColor(taqaDarkTeal);
                                                });
                                            }
                                        }
                                    });

                                    // Add charts with TAQA styling if available
                                    if (!string.IsNullOrEmpty(reportData.SeverityChart))
                                    {
                                        try
                                        {
                                            x.Item().PaddingTop(10).Text("SEVERITY ANALYSIS").FontSize(16).SemiBold().FontColor(taqaDarkTeal);
                                            var chartBytes = Convert.FromBase64String(reportData.SeverityChart.Split(',')[1]);
                                            x.Item().Border(1).BorderColor(taqaTeal).Padding(10).Image(chartBytes).FitWidth();
                                        }
                                        catch (Exception ex)
                                        {
                                            _logger.LogWarning(ex, "Failed to include severity chart in PDF");
                                            x.Item().Text("Severity Chart: Unable to load").FontSize(12).FontColor("#C62828");
                                        }
                                    }

                                    if (!string.IsNullOrEmpty(reportData.TimelineChart))
                                    {
                                        try
                                        {
                                            x.Item().PaddingTop(10).Text("TIMELINE ANALYSIS").FontSize(16).SemiBold().FontColor(taqaDarkTeal);
                                            var chartBytes = Convert.FromBase64String(reportData.TimelineChart.Split(',')[1]);
                                            x.Item().Border(1).BorderColor(taqaTeal).Padding(10).Image(chartBytes).FitWidth();
                                        }
                                        catch (Exception ex)
                                        {
                                            _logger.LogWarning(ex, "Failed to include timeline chart in PDF");
                                            x.Item().Text("Timeline Chart: Unable to load").FontSize(12).FontColor("#C62828");
                                        }
                                    }

                                    // Anomaly Distribution Table with TAQA styling
                                    if (reportData.SummaryData != null && reportData.SummaryData.Any())
                                    {
                                        x.Item().PaddingTop(10).Text("ANOMALY DISTRIBUTION BY DETECTOR").FontSize(16).SemiBold().FontColor(taqaDarkTeal);
                                        x.Item().Table(table =>
                                        {
                                            table.ColumnsDefinition(columns =>
                                            {
                                                columns.RelativeColumn(3);
                                                columns.RelativeColumn(1);
                                                columns.RelativeColumn(2);
                                            });

                                            table.Header(header =>
                                            {
                                                header.Cell().Element(HeaderStyle).Text("Detector System");
                                                header.Cell().Element(HeaderStyle).Text("Count").AlignCenter();
                                                header.Cell().Element(HeaderStyle).Text("Severity").AlignCenter();
                                            });

                                            foreach (var item in reportData.SummaryData.Where(kv => kv.Value > 0).OrderByDescending(kv => kv.Value))
                                            {
                                                var severity = GetSeverityLevel(item.Value);
                                                table.Cell().Element(CellStyle).Text(FormatDetectorName(item.Key)).FontColor(taqaNeutralGray);
                                                table.Cell().Element(CellStyle).Text(item.Value.ToString("N0")).AlignCenter().SemiBold().FontColor(taqaDarkTeal);
                                                table.Cell().Element(CellStyle).Text(severity.Item1).AlignCenter()
                                                    .FontColor(severity.Item2);
                                            }
                                        });
                                    }

                                    // Data Table with TAQA styling if available
                                    if (reportData.TableData != null && reportData.TableData.Any())
                                    {
                                        x.Item().PaddingTop(10).Text("ANOMALY DETAILS (TOP 10)").FontSize(16).SemiBold().FontColor(taqaDarkTeal);
                                        x.Item().Table(table =>
                                        {
                                            table.ColumnsDefinition(columns =>
                                            {
                                                columns.RelativeColumn(1); // #
                                                columns.RelativeColumn(2); // Timestamp
                                                columns.RelativeColumn(2); // Detector
                                                columns.RelativeColumn(1); // Severity
                                            });

                                            table.Header(header =>
                                            {
                                                header.Cell().Element(HeaderStyle).Text("#");
                                                header.Cell().Element(HeaderStyle).Text("Timestamp");
                                                header.Cell().Element(HeaderStyle).Text("Detector");
                                                header.Cell().Element(HeaderStyle).Text("Severity");
                                            });

                                            foreach (var row in reportData.TableData.Take(10))
                                            {
                                                table.Cell().Element(CellStyle).Text(row.Number).FontColor(taqaNeutralGray);
                                                table.Cell().Element(CellStyle).Text(FormatTimestamp(row.Timestamp)).FontColor(taqaNeutralGray);
                                                table.Cell().Element(CellStyle).Text(FormatDetectorName(row.Detector)).FontColor(taqaNeutralGray);
                                                table.Cell().Element(CellStyle).Text(row.Severity).AlignCenter()
                                                    .FontColor(row.Severity == "High" ? "#C62828" :
                                                              row.Severity == "Medium" ? taqaOrange : "#2E7D32");
                                            }
                                        });
                                    }

                                    // Recommendations Section with TAQA styling
                                    x.Item().Background("#FFF3E0").BorderLeft(4, Unit.Point).BorderColor(taqaOrange)
                                        .Padding(15).Column(rec =>
                                        {
                                            rec.Item().Text("RECOMMENDATIONS").FontSize(16).SemiBold().FontColor(taqaOrange);
                                            rec.Item().PaddingTop(5).Text("Based on anomaly detection analysis and system health assessment")
                                                .FontSize(10).FontColor(taqaNeutralGray).Italic();
                                            rec.Item().PaddingTop(10);
                                            
                                            // Generate dynamic recommendations based on the data
                                            var recommendations = GenerateRecommendations(reportData, totalAnomalies, systemHealth, riskLevel);
                                            
                                            foreach (var recommendation in recommendations)
                                            {
                                                rec.Item().Text($"• {recommendation}").FontSize(11).FontColor(taqaNeutralGray);
                                            }
                                        });

                                    // Helper functions for TAQA styling
                                    IContainer HeaderStyle(IContainer container)
                                    {
                                        return container.Background(headerTeal).Padding(8).DefaultTextStyle(x => x.FontColor(Colors.White).SemiBold());
                                    }

                                    IContainer CellStyle(IContainer container)
                                    {
                                        return container.Padding(6).BorderBottom(0.5f).BorderColor(cellBorder);
                                    }
                                });

                            // TAQA-branded footer
                            page.Footer()
                                .Background(taqaDarkTeal)
                                .Padding(10)
                                .Row(row =>
                                {
                                    row.RelativeItem().Text("TAQA - Industrialization & Energy Services Company")
                                        .FontSize(10).FontColor(Colors.White);
                                    row.RelativeItem().AlignCenter().Text("Systems Integrity & Anomaly Detection")
                                        .FontSize(8).FontColor(taqaLightTeal);
                                    row.RelativeItem().AlignRight().Text("Page 1")
                                        .FontSize(10).FontColor(Colors.White);
                                });
                        });
                    }).GeneratePdf();
                    
                    _logger.LogInformation("TAQA-branded PDF generation completed successfully, size: {Size} bytes", pdfBytes.Length);
                }
                catch (Exception pdfEx)
                {
                    _logger.LogError(pdfEx, "Error during PDF generation with QuestPDF");
                    throw;
                }

                reportStopwatch.Stop();
                _logger.LogInformation("=== PDF REPORT GENERATION COMPLETE === Total time: {ElapsedMs}ms, Size: {Size} bytes", 
                    reportStopwatch.ElapsedMilliseconds, pdfBytes.Length);
                
                return File(pdfBytes, "application/pdf", $"TAQA_Anomaly_Detection_Report_{DateTime.Now:yyyy-MM-dd}.pdf");
            }
            catch (Exception ex)
            {
                reportStopwatch.Stop();
                _logger.LogError(ex, "=== PDF REPORT GENERATION FAILED === After {ElapsedMs}ms: {Message}", 
                    reportStopwatch.ElapsedMilliseconds, ex.Message);
                
                return StatusCode(500, new { 
                    error = "Error generating report", 
                    details = ex.Message, 
                    type = ex.GetType().Name,
                    timestamp = DateTime.UtcNow,
                    elapsed = reportStopwatch.ElapsedMilliseconds
                });
            }
        }

        private object TransformToDashboardData(MLServiceResponse mlResults)
        {
            var anomaliesByModel = new Dictionary<string, object>();
            var timeSeriesData = new List<object>();
            var summaryStats = new Dictionary<string, object>();
            foreach (var prediction in mlResults.Predictions)
            {
                var modelName = prediction.Key;
                var result = prediction.Value;
                anomaliesByModel[modelName] = new
                {
                    count = result.AnomalyCount,
                    anomalies = result.Anomalies.Select(a => new
                    {
                        timestamp = a.Timestamp,
                        severity = a.Severity,
                        value = a.RawValue,
                        score = a.Score,
                        detector = a.Detector,
                        observedValue = a.ObservedValue,
                        predictedValue = a.PredictedValue,
                        residual = a.Residual,
                        tag = a.Tag
                    }).ToList()
                };
                foreach (var anomaly in result.Anomalies)
                {
                    timeSeriesData.Add(new
                    {
                        timestamp = anomaly.Timestamp,
                        model = modelName,
                        severity = anomaly.Severity,
                        value = anomaly.RawValue,
                        score = anomaly.Score,
                        detector = anomaly.Detector
                    });
                }
                summaryStats[modelName] = new
                {
                    totalAnomalies = result.AnomalyCount,
                    dataQuality = new
                    {
                        totalRows = result.Preprocess?.Stats?.NRows ?? 0,
                        validRows = result.Preprocess?.DqReport?.ValidRows ?? 0,
                        processingTimeMs = result.Timings?.TotalMs ?? 0
                    }
                };
            }
            return new
            {
                anomaliesByModel,
                timeSeriesData = timeSeriesData.OrderBy(x => ((dynamic)x).timestamp),
                summaryStats,
                overallStats = new
                {
                    totalModels = mlResults.TotalModels,
                    successfulModels = mlResults.SuccessCount,
                    failedModels = mlResults.ErrorCount,
                    totalAnomalies = anomaliesByModel.Values.Sum(v => ((dynamic)v).count)
                }
            };
        }

        private async Task<bool> IsMLServiceRunningAsync()
        {
            try
            {
                return await _mlServiceClient.IsHealthyAsync();
            }
            catch
            {
                return false;
            }
        }

        private bool StartMLServiceProcess()
        {
            try
            {
                // Go up two levels from the project directory to get the solution root
                var solutionRoot = Directory.GetParent(Directory.GetParent(_env.ContentRootPath).FullName).FullName;
                var mlServiceDir = Path.Combine(solutionRoot, "ml-service");
                var scriptPath = Path.Combine(mlServiceDir, "run_dev.py");

                if (!Directory.Exists(mlServiceDir))
                {
                    _logger.LogError($"ML service directory does not exist: {mlServiceDir}");
                    return false;
                }
                if (!System.IO.File.Exists(scriptPath))
                {
                    _logger.LogError($"ML service script does not exist: {scriptPath}");
                    return false;
                }

                _logger.LogInformation($"Starting ML service from {mlServiceDir}");
                _logger.LogInformation($"Script path: {scriptPath}");

                var psi = new ProcessStartInfo
                {
                    FileName = "python",
                    Arguments = $"\"{scriptPath}\"",
                    WorkingDirectory = mlServiceDir,
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = false
                };

                var process = new Process { StartInfo = psi };
                process.OutputDataReceived += (sender, args) => {
                    if (args.Data != null)
                        _logger.LogInformation($"ML Service: {args.Data}");
                };
                process.ErrorDataReceived += (sender, args) => {
                    if (args.Data != null)
                        _logger.LogError($"ML Service Error: {args.Data}");
                };
                process.Start();
                process.BeginOutputReadLine();
                process.BeginErrorReadLine();
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to start ML service process");
                return false;
            }
        }

        private async Task<bool> EnsureMLServiceRunningAsync(int timeoutSeconds = 60)
        {
            if (await IsMLServiceRunningAsync()) 
            {
                _logger.LogInformation("ML service is already running");
                return true;
            }
            
            _logger.LogInformation("ML service not running, attempting to start...");
            if (!StartMLServiceProcess()) return false;
            
            // Wait for service to start
            var sw = Stopwatch.StartNew();
            while (sw.Elapsed.TotalSeconds < timeoutSeconds)
            {
                if (await IsMLServiceRunningAsync())
                {
                    _logger.LogInformation($"ML service started successfully after {sw.Elapsed.TotalSeconds:F1} seconds");
                    return true;
                }
                _logger.LogInformation("Waiting for ML service to start...");
                await Task.Delay(1000);
            }
            
            _logger.LogError($"ML service failed to start after {timeoutSeconds} seconds");
            return false;
        }
    }
}
