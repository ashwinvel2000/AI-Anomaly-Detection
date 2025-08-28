# Anomaly Detection Pipeline Documentation

This document explains the core scripts needed for the anomaly detection pipeline. Note that the system is transitioning from local model inference to a FastAPI-based service architecture.

## Current Implementation (FastAPI Service)

### Directory Structure

```
taqa.polaris.sit/
├── Controllers/
│ ├── AnomalyController.cs # Handles API requests
│ ├── DataExplorationController.cs # Handles stream exploration
│ └── ReportLandingController.cs # Handles PDF generation
├── Models/
│ ├── MLServiceModels.cs # API response models
│ ├── AnomalyReportData.cs # Report data structure
│ └── ReportInputModel.cs # Report generation inputs
├── Services/
│ ├── MLServiceClient.cs # FastAPI client
│ └── Anomaly/
│ ├── AnomalyRouter.cs # Legacy detection logic
│ └── AnomalyEvent.cs # Anomaly data structure
├── Scripts/
│ └── DataHub.cs # DataHub integration services
└── Views/
├── Anomaly/
│ ├── Upload.cshtml # File upload interface
│ └── Detect.cshtml # Results dashboard
├── DataExploration/
│ └── Index.cshtml # Stream exploration UI
├── Home/
│ └── DataHub.cshtml # DataHub management UI
└── ReportLanding/
└── Index.cshtml # Report generation form
```

### Active Components

1. **Data Input**
   - Upload.cshtml: File upload interface
   - AnomalyController.cs: Handles file validation and API calls
   - MLServiceClient.cs: Communicates with FastAPI service
   - DataHub.cs: Handles DataHub integration

2. **Results Processing**
   - MLServiceModels.cs: Structures for API responses
   - Detect.cshtml: Dashboard display
   - AnomalyReportData.cs: Report data models

3. **Report Generation**
   - ReportLandingController.cs: Generates PDF reports
   - ReportInputModel.cs: Report parameters
   - Index.cshtml: Report form

### DataHub Integration

The DataHub functionality is implemented through:
- **Scripts/DataHub.cs**: Core DataHub services
  - Stream data management
  - Data upload/download
  - Batch processing
  - Error handling

- **Views/Home/DataHub.cshtml**: User interface for
  - Stream management
  - Data upload
  - Configuration
  
- **wwwroot/js/Pages/DataHub.js**: Client-side logic for
  - UI interactions
  - DataHub operations
  - Error handling

### How It Works

1. Data Upload Path:
   - CSV Files:
     1. Upload through Upload.cshtml
     2. Files validated by AnomalyController
     3. MLServiceClient sends to FastAPI service
     4. Results shown in Detect.cshtml

   - DataHub Path:
     1. Configure in DataHub.cshtml
     2. DataHub.cs fetches stream data
     3. Data sent to FastAPI service
     4. Results shown in Detect.cshtml

2. Processing:
   - FastAPI service runs all 10 models
   - Returns comprehensive results
   - Results displayed in dashboard
   - Option to generate PDF report

3. Report Generation:
   - User inputs parameters in ReportLanding/Index.cshtml
   - System captures charts and data
   - Generates PDF using QuestPDF

This older implementation used:
- Local ONNX runtime
- Model files stored in the application
- Direct row-by-row processing

The system is moving to the FastAPI service instead, which provides:
- Centralized model management
- Better scalability
- Easier updates
- Consistent processing

## Required Files

### CSV Files Needed
1. Tool.P8-XX.SIT.Battery-Voltage.csv
2. Tool.P8-XX.SIT.Choke-Position.csv
3. Tool.P8-XX.SIT.Downstream-Pressure.csv
4. Tool.P8-XX.SIT.Downstream-Temperature.csv
5. Tool.P8-XX.SIT.Downstream-Upstream-Difference.csv
6. Tool.P8-XX.SIT.Target-Position.csv
7. Tool.P8-XX.SIT.Tool-State.csv
8. Tool.P8-XX.SIT.Upstream-Pressure.csv
9. Tool.P8-XX.SIT.Upstream-Temperature.csv

## Machine Learning Models (Now on FastAPI Service)

### Isolation Forest Models
- choke_position: Finds unusual choke positions
- delta_temp_open: Checks temperature differences
- full_vectors_if: Looks at multiple measurements
- pressure_pair_open: Analyzes pressure patterns

### Residual Models
- residual_battery: Checks battery behavior
- residual_downP: Monitors downstream pressure
- residual_downT: Tracks downstream temperature
- residual_upP: Watches upstream pressure
- residual_upT: Monitors upstream temperature
- target_pos_residual: Checks target positions

## PDF Report Generation

Uses QuestPDF package with these components:

1. **ReportLanding/Index.cshtml**
   - Form for report parameters
   - Tool ID and date selection
   - Namespace choice

2. **ReportLandingController.cs**
   - Handles report generation
   - Gathers data for PDF
   - Creates document structure

3. **AnomalyReportData.cs**
   - Report data models
   - Chart and table structures
   - Summary statistics

4. **ReportInputModel.cs**
   - Input parameters
   - Validation rules
   - Default values

## Error Handling

The system handles:
- Wrong file uploads
- Missing data
- API connection issues
- Processing errors
- Report generation problems

## Report Contents

PDF reports include:
- Severity distribution charts
- Anomaly timelines
- Top anomalies table
- Statistics summary
- Data quality metrics
- Processing details


The system is designed to handle both the current FastAPI service implementation while maintaining compatibility with any remaining legacy components during the transition.


