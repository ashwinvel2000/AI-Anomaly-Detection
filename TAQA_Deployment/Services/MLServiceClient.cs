using System.Text.Json;
using System.Net.Http.Headers;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using System.Net.Http;
using System.Threading.Tasks;
using taqa.polaris.sit.Models;

namespace taqa.polaris.sit.Services
{
    public class MLServiceClient
    {
        private readonly HttpClient _httpClient;
        private readonly string _baseUrl;
        private readonly ILogger<MLServiceClient> _logger;

        public MLServiceClient(HttpClient httpClient, IConfiguration config, ILogger<MLServiceClient> logger)
        {
            _httpClient = httpClient;
            _httpClient.Timeout = TimeSpan.FromSeconds(300); // Force timeout to 300 seconds
            _baseUrl = config.GetValue<string>("MLService:BaseUrl", "http://localhost:8000");
            _logger = logger;
            _logger.LogInformation($"MLServiceClient HttpClient timeout (forced): {_httpClient.Timeout.TotalSeconds} seconds");
        }

        public async Task<bool> IsHealthyAsync()
        {
            try
            {
                // Use GET request to /docs for FastAPI readiness
                var response = await _httpClient.GetAsync($"{_baseUrl}/docs");
                return response.IsSuccessStatusCode;
            }
            catch (Exception ex)
            {
                _logger.LogDebug($"Health check failed: {ex.Message}");
                return false;
            }
        }

        public async Task<MLServiceResponse> RunAllModelsAnalysisAsync(List<IFormFile> csvFiles)
        {
            _logger.LogInformation($"Starting ML analysis with {csvFiles.Count} CSV files");
            using var formData = new MultipartFormDataContent();
            foreach (var file in csvFiles)
            {
                var fileContent = new StreamContent(file.OpenReadStream());
                fileContent.Headers.ContentType = new MediaTypeHeaderValue("text/csv");
                formData.Add(fileContent, "files", file.FileName);
                _logger.LogDebug($"Added file: {file.FileName}");
            }
            try
            {
                // Use the correct endpoint path that matches the FastAPI route
                var response = await _httpClient.PostAsync($"{_baseUrl}/predict/all", formData);
                if (!response.IsSuccessStatusCode)
                {
                    var error = await response.Content.ReadAsStringAsync();
                    _logger.LogError($"ML Service Error: {response.StatusCode} - {error}");
                    throw new Exception($"ML Service Error: {response.StatusCode} - {error}");
                }
                var jsonContent = await response.Content.ReadAsStringAsync();
                var result = JsonSerializer.Deserialize<MLServiceResponse>(jsonContent, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
                _logger.LogInformation($"ML analysis completed. Successful models: {result.SuccessCount}, Failed: {result.ErrorCount}");
                return result;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error calling ML service");
                throw;
            }
        }
    }
}
