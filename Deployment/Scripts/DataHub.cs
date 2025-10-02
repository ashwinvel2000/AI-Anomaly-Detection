using Microsoft.AspNetCore.Routing;
using OSIsoft.Data;
using OSIsoft.Data.Http;
using System.Globalization;
using System.Reflection;
using taqa.polaris.sit.Configuration;
using taqa.polaris.sit.Entries;
using taqa.polaris.sit.Models;
using taqa.polaris.sit.Types;

namespace taqa.polaris.sit.DataHub
{
    public class DataHubServices
    {
        private readonly ILogger _logger;
        private readonly ISdsDataService _dataService;
        public readonly APIConfig _apiCfg;
        private const int BatchSize = 50000;
        private const int EmptyPageLimit = 50;

        private static readonly string[] _tsNames = { "Timestamp", "timestamp", "Time", "DateTime" };

        private static DateTime? TryGetTimestamp(dynamic? obj)
        {
            if (obj is null) return null;

            foreach (var name in _tsNames)
            {
                try
                {
                    var prop = obj.GetType().GetProperty(
                                   name,
                                   BindingFlags.Public | BindingFlags.Instance |
                                   BindingFlags.IgnoreCase);
                    if (prop == null) continue;

                    var val = prop.GetValue(obj);

                    if (val is DateTime dt)
                        return dt.ToUniversalTime();

                    if (val is DateTimeOffset dto)
                        return dto.UtcDateTime;
                }
                catch { /* keep trying other names */ }
            }

            // Expando / dictionary fallback
            if (obj is IDictionary<string, object> dict)
            {
                foreach (var n in _tsNames)
                    if (dict.TryGetValue(n, out var v))
                    {
                        if (v is DateTime dt) return dt.ToUniversalTime();
                        if (v is DateTimeOffset dto) return dto.UtcDateTime;
                    }
            }

            // 3️⃣  JObject path  (Data Hub sometimes returns first/last as JSON objects)
            if (obj is Newtonsoft.Json.Linq.JObject jobj)
            {
                foreach (var n in _tsNames)
                {
                    if (jobj.TryGetValue(n, StringComparison.OrdinalIgnoreCase, out var tok) &&
                        DateTime.TryParse(tok.ToString(), out var dt))
                        return dt.ToUniversalTime();
                }
            }
            return null;
        }

        public DataHubServices(ILogger logger,
                               ISdsDataService dataService,
                               APIConfig apiCfg)
        {
            _logger = logger;
            _dataService = dataService;
            _apiCfg = apiCfg;
        }

        public DataHubServices(ILogger logger,
                               ISdsDataService dataService)
            : this(logger, dataService, new APIConfig())
        {

        }

        public async Task<List<T>> DownloadStreamToMemory<T>(string streamId, string startDateStr)
        {
            var formats = new[] { "o", "yyyy-MM-ddTHH:mm:ssK", "yyyy-MM-ddTHH:mm:ss.fffK", "yyyy-MM-dd" };
            var startDate = DateTimeOffset.ParseExact(startDateStr, formats, CultureInfo.InvariantCulture, DateTimeStyles.None); // Fixed ParseExact usage
            var cursor = DateTimeOffset.UtcNow; // start with “now”
            int emptyStreak = 0;
            var rows = new List<T>();

            while (cursor > startDate)
            {
                var windowStart = cursor.AddDays(-30); // 10-day slices
                if (windowStart < startDate) windowStart = startDate;

                _logger.LogDebug("Reading stream {StreamId} ({Namespace}) {From:u} — {To:u}",streamId, AppConfig.selectedNamespace, windowStart, cursor);

                var page = await _dataService.GetWindowValuesAsync<T>(
                               streamId,
                               startIndex : $"{windowStart:O}",
                               endIndex   : $"{cursor:O}",
                               boundaryType : SdsBoundaryType.Outside);  

                _logger.LogDebug("Fetched {Count} pts  {From:u} – {To:u}", page?.Count() ?? 0, windowStart, cursor);

                int pageCount = page?.Count() ?? 0;

                if (pageCount == 0)
                {
                    emptyStreak++;
                    if (emptyStreak >= EmptyPageLimit) break;
                }
                else
                {
                    rows.AddRange(page!);             // page is not null here
                    emptyStreak = 0;
                }

                cursor = windowStart; // slide the window back
            }

            rows.Reverse(); // chronological order
            return rows;
        }


        public async Task<StreamWindowInfo?> GetStreamWindowInfo(
                string streamId,
                DateTime fromUtc,
                DateTime toUtc,
                CancellationToken ct = default)
        {
            // SDK returns null if no data
            var firstObj = await _dataService.GetFirstValueAsync<dynamic>(streamId);
            var lastObj = await _dataService.GetLastValueAsync<dynamic>(streamId);


            var keys = firstObj is null
                       ? "null"
                       : string.Join(", ",
                             ((object)firstObj).GetType()      
                                               .GetProperties()
                                               .Select(p => p.Name));

            DateTime? firstUtc = TryGetTimestamp(firstObj);
            DateTime? lastUtc = TryGetTimestamp(lastObj);

            if (firstUtc is null || lastUtc is null)
            {
                _logger.LogDebug("Stream {Id} has no data between {From} and {To}", streamId, fromUtc, toUtc);
                return null;
            }                    //  EARLY EXIT for empty streams

            var http = _apiCfg.CreateAuthedClient();
            http.DefaultRequestHeaders.Accept.Add(new System.Net.Http.Headers.MediaTypeWithQualityHeaderValue("text/plain"));
            var tenant = AppConfig.tenantId;
            var ns = AppConfig.selectedNamespace;

            var url = $"/api/v1-preview/Tenants/{tenant}/Namespaces/{ns}/Streams/" +
                      $"{streamId}/Data/Summaries" + 
                      $"?startIndex={firstUtc.Value:O}" +
                      $"&endIndex={lastUtc.Value:O}"+              
                      $"&summaryType=Count" +
                      $"&count=1";

            var resp = await http.GetAsync(url, ct);

            int total = 0;

            if (resp.IsSuccessStatusCode)
            {
                var txt = await resp.Content.ReadAsStringAsync(ct);
                var root = Newtonsoft.Json.Linq.JArray.Parse(txt)[0];
                var countObj = root["Summaries"]?["Count"] as Newtonsoft.Json.Linq.JObject;

                if (countObj != null)
                {
                    // take the first numeric property inside "Count"
                    total = countObj.Properties()
                                   .Select(p => (int?)p.Value)   // value is JToken → int?
                                   .FirstOrDefault() ?? 0;
                }

                // optional debug
                _logger.LogDebug("Raw body => {Body}", txt);
            }
            else
            {
                _logger.LogWarning("Count-summary call {Code} for {Id}",
                                   resp.StatusCode, streamId);
            }

            _logger.LogDebug("Count URL  => {Url}", url);
            _logger.LogDebug("HTTP code => {Code}", resp.StatusCode);


            return new StreamWindowInfo(streamId, firstUtc.Value, lastUtc.Value, total);
        }

        public StreamEntry CreateStreamEntry((string StreamId, string StreamName, string StreamDescription, string StreamTypeId, string PropertyOverrideTypeId, string PropertyOverrideUom, SdsInterpolationMode PropertyOverrideInterpolationMode) streamConfig)
        {
            var streamEntry = new StreamEntry()
            {
                StreamId = streamConfig.StreamId,
                StreamName = streamConfig.StreamName,
                StreamTypeId = streamConfig.StreamTypeId,
                StreamDescription = streamConfig.StreamDescription,
                PropertyOverrideTypeId = streamConfig.PropertyOverrideTypeId,
                PropertyOverrideInterpolationMode = streamConfig.PropertyOverrideInterpolationMode,
                PropertyOverrideUom = streamConfig.PropertyOverrideUom,
            };
            return streamEntry;
        }

        public SdsStream CreateStream(StreamEntry streamEntry)
        {
            SdsStream sdsStream = new()
            {
                Id = streamEntry.StreamId,
                Name = streamEntry.StreamName,
                TypeId = streamEntry.StreamTypeId,
                Description = streamEntry.StreamDescription,
            };

            if (!(string.IsNullOrEmpty(streamEntry.PropertyOverrideTypeId)) && !(string.IsNullOrEmpty(streamEntry.PropertyOverrideUom)))
            {
                var propertyOverride = new SdsStreamPropertyOverride()
                {
                    SdsTypePropertyId = streamEntry.PropertyOverrideTypeId,
                    Uom = streamEntry.PropertyOverrideUom,
                    InterpolationMode = streamEntry.PropertyOverrideInterpolationMode
                };

                var propertyOverrides = new List<SdsStreamPropertyOverride>() { propertyOverride };
                sdsStream.PropertyOverrides = propertyOverrides;
            }

            return sdsStream;
        }

		public async Task<int> PushPropertyData<T>(PropertyInfo Property, List<T> EntryData, string streamId)
		{
			try
			{
				List<object> Batch = new();
				int RowCount = EntryData.Count; // Get total row count
				int RowIndex = 0; // Initialize row index counter

				int ValuesPushedCounter = 0;
				string PropertyName = Property.Name;

				PropertyInfo TimestampProperty = typeof(T).GetProperty("Timestamp");
				if (TimestampProperty == null)
				{
					throw new ArgumentException("The type T does not have a 'Timestamp' property.");
				}

				for (int i = 0; i < RowCount; i++)
				{
					var Entry = EntryData[i];
					var Value = Property.GetValue(Entry);
					var Timestamp = (DateTimeOffset)TimestampProperty.GetValue(Entry);

                    if ((Value != null) && (Value != DBNull.Value))
					{
						switch (PropertyName)
						{
							case "BatteryVoltage":
								Batch.Add(new BatteryVoltageType() { Voltage = Convert.ToDouble(Value), Timestamp = Timestamp });
								break;

							case "UpstreamPressure":
                                Batch.Add(new UpstreamPressureType() { Pressure = Convert.ToDouble(Value), Timestamp = Timestamp });
                                break;

                            case "DownstreamPressure":
                                Batch.Add(new DownstreamPressureType() { Pressure = Convert.ToDouble(Value), Timestamp = Timestamp });
                                break;

                            case "UpstreamTemperature":
                                Batch.Add(new UpstreamTemperatureType() { Temperature = Convert.ToDouble(Value), Timestamp = Timestamp });
                                break;

                            case "DownstreamTemperature":
                                Batch.Add(new DownstreamTemperatureType() { Temperature = Convert.ToDouble(Value), Timestamp = Timestamp });
                                break;

                            case "CPUTemperature":
                                Batch.Add(new CPUTemperatureType() { Temperature = Convert.ToDouble(Value), Timestamp = Timestamp });
                                break;

                            case "ChokePosition":
                                Batch.Add(new PositionChokeType() { Position = Convert.ToDouble(Value), Timestamp = Timestamp });
                                break;
                            
                            case "PowerCyclesSIT":
                                Batch.Add(new PowerCyclesSitType() { Cycles = Convert.ToUInt32(Value), Timestamp = Timestamp });
                                break;

                            case "PowerCycles":
                                Batch.Add(new PowerCyclesType() { Cycles = Convert.ToInt16(Value), Timestamp = Timestamp });
                                break;

                            case "WatchdogResets":
                                Batch.Add(new WatchdogResetType() { Resets = Convert.ToInt16(Value), Timestamp = Timestamp });
                                break;

                            case "WatchdogResetsSIT":
                                Batch.Add(new WatchdogResetSitType() { Resets = Convert.ToUInt32(Value), Timestamp = Timestamp });
                                break;

                            case "LeadscrewRuntime":
                                Batch.Add(new LeadscrewRuntimeType() { Runtime = Convert.ToInt32(Value), Timestamp = Timestamp });
                                break;

                            case "LeadscrewRuntimeSIT":
                                Batch.Add(new LeadscrewRuntimeSitType() { Runtime = Convert.ToUInt32(Value), Timestamp = Timestamp });
                                break;

                            case "HydraulicRuntime":
                                Batch.Add(new HydraulicRuntimeType() { Runtime = Convert.ToInt32(Value), Timestamp = Timestamp });
                                break;

                            case "HydraulicRuntimeSIT":
                                Batch.Add(new HydraulicRuntimeSitType() { Runtime = Convert.ToUInt32(Value), Timestamp = Timestamp });
                                break;

                            case "BatteryLife":
                                Batch.Add(new BatteryLifetimeType() { Lifetime = Convert.ToInt16(Value), Timestamp = Timestamp });
                                break;

                            case "BatteryLifeSIT":
                                Batch.Add(new BatteryLifetimeSitType() { Lifetime = Convert.ToUInt32(Value), Timestamp = Timestamp });
                                break;

                            case "Events":
                                Batch.Add(new EventType() { Event = Convert.ToString(Value), Timestamp = Timestamp });
                                break;

                            case "DownstreamDifferential":
                                Batch.Add(new DownstreamDifferentialType() { Pressure = Convert.ToDouble(Value), Timestamp = Timestamp });
                                break;

                            case "UpstreamDifferential":
                                Batch.Add(new UpstreamDifferentialType() { Pressure = Convert.ToDouble(Value), Timestamp = Timestamp });
                                break;

                            case "AverageVoltage":
                                Batch.Add(new AverageVoltageType() { Voltage = Convert.ToDouble(Value), Timestamp = Timestamp });
                                break;

                            case "BoreTracker":
                                Batch.Add(new BoreTrackerType() { Pressure = Convert.ToDouble(Value), Timestamp = Timestamp });
                                break;

                            case "PulseDetect":
                                Batch.Add(new PulseDetectType() { PulseDetect = Convert.ToDouble(Value), Timestamp = Timestamp });
                                break;

                            case "NegativePulseDetect":
                                Batch.Add(new NegativePulseDetectType() { PulseDetect = Convert.ToDouble(Value), Timestamp = Timestamp });
                                break;

                            case "DownstreamUpstreamDifference":
                                Batch.Add(new DownstreamUpstreamDifferenceType() { Pressure = Convert.ToDouble(Value), Timestamp = Timestamp });
                                break;

                            case "BoreLive":
                                Batch.Add(new BoreLiveType() { Pressure = Convert.ToDouble(Value), Timestamp = Timestamp });
                                break;

                            case "FlashDataCount":
                                Batch.Add(new FlashDataCountType() { Count = Convert.ToInt32(Value), Timestamp = Timestamp });
                                break;

                            case "LogCache":
                                Batch.Add(new LogCacheType() { Bytes = Convert.ToUInt16(Value), Timestamp = Timestamp });
                                break;

                            case "ToolState":
                                Batch.Add(new ToolStateType() { State = Convert.ToInt16(Value), Timestamp = Timestamp });
                                break;

                            case "PulseCount":
                                Batch.Add(new PulseCountType() { Count = Convert.ToUInt16(Value), Timestamp = Timestamp });
                                break;

                            case "MotorStates":
                                Batch.Add(new MotorStateType() { State = Convert.ToUInt16(Value), Timestamp = Timestamp });
                                break;

                            case "LocalMotorSpeed":
                                Batch.Add(new LocalMotorSpeedType() { Speed = Convert.ToUInt16(Value), Timestamp = Timestamp });
                                break;

                            case "RemoteMotorSpeed":
                                Batch.Add(new RemoteMotorSpeedType() { Speed = Convert.ToUInt16(Value), Timestamp = Timestamp });
                                break;

                            case "RawPosition":
                                Batch.Add(new PositionRawType() { Position = Convert.ToUInt16(Value), Timestamp = Timestamp });
                                break;

                            case "TargetPosition":
                                Batch.Add(new PositionTargetType() { Position = Convert.ToUInt16(Value), Timestamp = Timestamp });
                                break;

                            case "Position0":
                                Batch.Add(new Position0Type() { Position = Convert.ToUInt16(Value), Timestamp = Timestamp });
                                break;

                            case "Position100":
                                Batch.Add(new Position100Type() { Position = Convert.ToUInt16(Value), Timestamp = Timestamp });
                                break;

                            case "LocalMotorCurrent":
                                Batch.Add(new LocalMotorCurrentType() { Current = Convert.ToUInt16(Value), Timestamp = Timestamp });
                                break;

                            case "RemoteMotorCurrent":
                                Batch.Add(new RemoteMotorCurrentType() { Current = Convert.ToUInt16(Value), Timestamp = Timestamp });
                                break;

                            case "LocalMotorPulses":
                                Batch.Add(new LocalMotorPulsesType() { MotorPulses = Convert.ToUInt16(Value), Timestamp = Timestamp });
                                break;

                            case "RemoteMotorPulses":
                                Batch.Add(new RemoteMotorPulsesType() { MotorPulses = Convert.ToUInt16(Value), Timestamp = Timestamp });
                                break;

                            case "MasterFirmwareVersion":
                                Batch.Add(new MasterFirmwareVersionType() { Version = Convert.ToInt32(Value), Timestamp = Timestamp });
                                break;

                            case "MasterFirmwareBuild":
                                Batch.Add(new MasterFirmwareBuildType() { Build = Convert.ToInt32(Value), Timestamp = Timestamp });
                                break;

                            case "SlaveHeartbeat":
                                Batch.Add(new SlaveHeartbeatType() { Heartbeat = Convert.ToInt16(Value), Timestamp = Timestamp });
                                break;

                            case "SlaveFirmwareVersion":
                                Batch.Add(new SlaveFirmwareVersionType() { Version = Convert.ToInt32(Value), Timestamp = Timestamp });
                                break;

                            case "SlaveFirmwareBuild":
                                Batch.Add(new SlaveFirmwareBuildType() { Build = Convert.ToInt32(Value), Timestamp = Timestamp });
                                break;

                            case "MinutesToPulse":
                                Batch.Add(new MinutesToPulseType() { Minutes = Convert.ToInt32(Value), Timestamp = Timestamp });
                                break;

                            case "SecondsToPulse":
                                Batch.Add(new SecondsToPulseType() { Seconds = Convert.ToInt32(Value), Timestamp = Timestamp });
                                break;

                            case "ToolStatus":
                                Batch.Add(new ToolStatusType() { Status = Convert.ToUInt16(Value), Timestamp = Timestamp });
                                break;

                            case "MotorStallSpeed":
                                Batch.Add(new MotorStallSpeedType() { Speed = Convert.ToUInt16(Value), Timestamp = Timestamp });
                                break;

                            case "LiveCRC":
                                Batch.Add(new LiveCRCType() { CRC = Convert.ToUInt16(Value), Timestamp = Timestamp });
                                break;

                            case "DeviceStatus":
                                Batch.Add(new DeviceStatusType() { Status = Convert.ToUInt16(Value), Timestamp = Timestamp });
                                break;

                            case "ToolType":
                                Batch.Add(new ToolType() { Type = Convert.ToUInt16(Value), Timestamp = Timestamp });
                                break;

                            case "StartupTime":
                                Batch.Add(new StartupTimeType() { Time = Convert.ToInt64(Value), Timestamp = Timestamp });
                                break;

                            case "StackAllocation":
                                Batch.Add(new StackAllocationType() { Bytes = Convert.ToUInt16(Value), Timestamp = Timestamp });
                                break;

                            case "HeapAllocation":
                                Batch.Add(new HeapAllocationType() { Bytes = Convert.ToUInt16(Value), Timestamp = Timestamp });
                                break;

                            case "SendPulse":
                                Batch.Add(new SendPulseType() { Count = Convert.ToUInt16(Value), Timestamp = Timestamp });
                                break;

                            case "PulseCommand":
                                Batch.Add(new PulseCommandType() { Command = Convert.ToUInt16(Value), Timestamp = Timestamp });
                                break;

                            case "Debug1":
                                Batch.Add(new Debug1Type() { Debug = Convert.ToInt16(Value), Timestamp = Timestamp });
                                break;

                            case "Debug2":
                                Batch.Add(new Debug2Type() { Debug = Convert.ToInt16(Value), Timestamp = Timestamp });
                                break;

                            case "Debug3":
                                Batch.Add(new Debug3Type() { Debug = Convert.ToInt16(Value), Timestamp = Timestamp });
                                break;

                            case "Debug4":
                                Batch.Add(new Debug4Type() { Debug = Convert.ToInt16(Value), Timestamp = Timestamp });
                                break;

                            case "Debug5":
                                Batch.Add(new Debug5Type() { Debug = Convert.ToInt16(Value), Timestamp = Timestamp });
                                break;

                            case "Debug6":
                                Batch.Add(new Debug6Type() { Debug = Convert.ToInt16(Value), Timestamp = Timestamp });
                                break;

                            case "Debug7":
                                Batch.Add(new Debug7Type() { Debug = Convert.ToInt16(Value), Timestamp = Timestamp });
                                break;

                            case "Debug8":
                                Batch.Add(new Debug8Type() { Debug = Convert.ToInt16(Value), Timestamp = Timestamp });
                                break;

                            case "Debug9":
                                Batch.Add(new Debug9Type() { Debug = Convert.ToInt16(Value), Timestamp = Timestamp });
                                break;

                            case "Debug10":
                                Batch.Add(new Debug10Type() { Debug = Convert.ToInt16(Value), Timestamp = Timestamp });
                                break;

                            case "DifferentialPressure":
                                Batch.Add(new DownstreamDifferentialType() { Pressure = Convert.ToDouble(Value), Timestamp = Timestamp });
                                break;

                            case "PulseQuality":
                                Batch.Add(new PulseQualityType() { Quality = Convert.ToInt32(Value), Timestamp = Timestamp });
                                break;

                            case "DecodedBatteryLife":
                                Batch.Add(new DecodedBatteryLifetimeType() { Lifetime = Convert.ToDouble(Value), Timestamp = Timestamp });
                                break;

                            case "FlashUse":
                                Batch.Add(new FlashUseType() { Usage = Convert.ToDouble(Value), Timestamp = Timestamp });
                                break;

                            case "ShutInPressure":
                                Batch.Add(new ShutInPressureType() { Pressure = Convert.ToDouble(Value), Timestamp = Timestamp });
                                break;

                            default:
                                throw new Exception("Could not get the mapped type of the parameter");

                        }
					}

                    if (Batch.Count >= BatchSize || i == RowCount - 1)
                    {
                        await _dataService.UpdateValuesAsync(streamId, Batch);
                        ValuesPushedCounter += Batch.Count;
                        Batch.Clear();
                    }
                }
				return ValuesPushedCounter;
			}
			catch (Exception ex)
			{
				_logger.LogError("Failed to push column to DataHub. Error: {errorMessage}", ex.Message);
				throw;
			}
		}

		public async Task PushDataToHub<T>(List<T> records, string streamId)
        {
            try
            {
                int totalRecords = records.Count;
                List<T> batch = new();

                for (int startIndex = 0; startIndex < totalRecords; startIndex += BatchSize)
                {
                    int endIndex = Math.Min(startIndex + BatchSize, totalRecords);

                    Console.WriteLine($"Processing records {startIndex} to {endIndex - 1}");

                    for (int i = startIndex; i < endIndex; i++)
                    {
                        var record = records[i];
                        batch.Add(record);
                    }

                    await _dataService.UpdateValuesAsync(streamId, batch);

                    batch.Clear();
                }
            }
            catch (Exception ex)
            {
                _logger.LogError("Failed to push data to the Hub. Error: {errorMessage}", ex.Message);
                throw;
            }
        }
    }
}