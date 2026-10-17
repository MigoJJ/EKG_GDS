# Eclipse Java EMR (`GDSEMREKG`) Integration Specification

**Version:** 2.0 (Phase 5 Step 5 - Async + Input Formats)  
**Last Updated:** 2026-10-17

This document describes how to integrate the Python EKG AI Assistant service (`http://localhost:8000`) into the existing Eclipse Java EMR application (`com.emr.gds`). Updated to support:
- ✅ Asynchronous analysis (background processing)
- ✅ DICOM file integration
- ✅ HL7 message parsing
- ✅ SCP-ECG device files
- ✅ API key authentication
- ✅ Progress monitoring & cancellation

---

## 1. Architecture Overview

```text
[JavaFX EMR Client (Eclipse)]
      │
      │  HTTP POST /api/v1/ecg/analyze (OkHttp / Gson)
      ▼
[FastAPI Python Microservice (Port 8000)]
      │
      ├─► 1. Preprocessor & Feature Extractor (SciPy / NeuroKit2)  [<10ms]
      ├─► 2. 1D-ResNet18 Diagnostic Classifier (ONNX Runtime CPU)  [<5ms]
      └─► 3. Ollama Clinical Interpretation (Qwen 2.5 REST)        [10-20s]
      │
      ▼
[Unified JSON Response with Legal Disclaimer & KCD Codes]
```

---

## 2. API Contract

### Request: `POST http://localhost:8000/api/v1/ecg/analyze`
```json
{
  "signal": [[0.12, 0.45, ...], ...],  // (1000, 12) samples in mV, or null if ecg_id is sent
  "ecg_id": 63,                         // Optional PTB-XL record ID for simulation
  "sampling_rate": 100,
  "patient_age": 62,
  "patient_sex": "M",
  "symptoms": "급성 흉통 호소",
  "output_format": "both"
}
```

### Response (JSON):
```json
{
  "status": "success",
  "patient_info": { "age": 62, "sex": "M" },
  "quantitative_features": {
    "heart_rate_bpm": 60.5,
    "rhythm_regular": true,
    "pr_interval_ms": 180.0,
    "qrs_duration_ms": 60.0,
    "qtc_interval_ms": 321.4,
    "heart_axis_deg": 45.5,
    "heart_axis_category": "Normal Axis",
    "sokolow_lyon_mv": 2.46,
    "lvh_voltage_criteria": false,
    "stemi_suspected": true,
    "st_elevation_leads": ["V1", "V2", "V3"]
  },
  "deep_learning_classification": {
    "probabilities": { "NORM": 0.05, "MI": 0.88, "AFIB": 0.02, ... },
    "top_findings": [["MI", 0.88], ["STTC", 0.72]],
    "clinical_urgency": "EMERGENCY"
  },
  "ollama_narrative_report": "### 1. [기본 리듬 및 심전도 간격] ...",
  "emr_structured_json": {
    "primary_diagnosis": "급성 전중격 심근경색증 (ASMI)",
    "kcd_code": "I21.0",
    "urgency": "EMERGENCY"
  },
  "disclaimer": "※ 본 판독문은 인공지능 심전도(ECG) 판독 보조 소프트웨어의 분석 결과이며..."
}
```

---

## 3. Java Client Implementation (`com.emr.gds.features.ekg.service`)

Copy this service class into your Eclipse workspace:

```java
package com.emr.gds.features.ekg.service;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import okhttp3.*;
import java.io.IOException;
import java.util.concurrent.TimeUnit;

public class EkgAiService {
    private static final String API_URL = "http://localhost:8000/api/v1/ecg/analyze";
    private final OkHttpClient client;
    private final Gson gson;

    public EkgAiService() {
        this.client = new OkHttpClient.Builder()
                .connectTimeout(5, TimeUnit.SECONDS)
                .readTimeout(60, TimeUnit.SECONDS)
                .build();
        this.gson = new Gson();
    }

    /**
     * Sends ECG record ID and patient info to the AI service.
     */
    public JsonObject analyzeEcgById(int ecgId, double age, String sex, String symptoms) throws IOException {
        JsonObject req = new JsonObject();
        req.addProperty("ecg_id", ecgId);
        req.addProperty("patient_age", age);
        req.addProperty("patient_sex", sex);
        req.addProperty("symptoms", symptoms);
        req.addProperty("output_format", "both");

        RequestBody body = RequestBody.create(
                req.toString(),
                MediaType.parse("application/json; charset=utf-8")
        );

        Request request = new Request.Builder()
                .url(API_URL)
                .post(body)
                .build();

        try (Response response = client.newCall(request).execute()) {
            if (!response.isSuccessful()) {
                throw new IOException("AI Server Error: " + response.code() + " " + response.message());
            }
            return gson.fromJson(response.body().string(), JsonObject.class);
        }
    }
}
```

---

## 4. JavaFX UI Integration Example

```java
// Asynchronous call using JavaFX Task to prevent UI freezing
Task<JsonObject> aiTask = new Task<>() {
    @Override
    protected JsonObject call() throws Exception {
        return ekgAiService.analyzeEcgById(selectedEcgId, patient.getAge(), patient.getSex(), "흉통");
    }
};

aiTask.setOnSucceeded(e -> {
    JsonObject result = aiTask.getValue();
    JsonObject feats = result.getAsJsonObject("quantitative_features");
    JsonObject dl = result.getAsJsonObject("deep_learning_classification");
    String report = result.get("ollama_narrative_report").getAsString();

    // 1. Populate UI text fields
    hrLabel.setText(feats.get("heart_rate_bpm").getAsString() + " bpm");
    qrsLabel.setText(feats.get("qrs_duration_ms").getAsString() + " ms");
    axisLabel.setText(feats.get("heart_axis_deg").getAsString() + "°");
    reportTextArea.setText(report);

    // 2. Emergency STEMI / High-Risk Alert
    boolean isStemi = feats.get("stemi_suspected").getAsBoolean();
    String urgency = dl.get("clinical_urgency").getAsString();
    if (isStemi || "EMERGENCY".equals(urgency)) {
        Alert alert = new Alert(Alert.AlertType.WARNING);
        alert.setTitle("임상 응급 경보 (EMERGENCY)");
        alert.setHeaderText("급성 심근경색(STEMI) 의심 소견 감지!");
        alert.setContentText("ST 분절 상승 유도: " + feats.get("st_elevation_leads") + "\n즉시 응급 진료 처치를 고려하세요.");
        alert.show();
    }
});

new Thread(aiTask).start();
```

---

## 5. Phase 5 Step 5: Asynchronous API & Input Formats

### 5.1 Async Endpoints Overview

**Problem:** Synchronous analysis blocks EMR UI for 4–5 seconds (LLM generation).

**Solution:** Background job queue with progress polling.

```
Old (Phase 4):  POST /analyze → 5000ms wait → response
New (Phase 5):  POST /analyze-async → 200ms → job_id
                GET /jobs/{job_id} → progress (0–100%)
                GET /jobs/{job_id} → complete with result
```

### 5.2 Extended Java Service (`EkgAiService` v2)

```java
package com.emr.gds.features.ekg.service;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.JsonArray;
import okhttp3.*;
import java.io.File;
import java.io.IOException;
import java.util.concurrent.TimeUnit;

public class EkgAiService {
    private static final String BASE_URL = "http://localhost:8000/api/v1";
    private static final String API_KEY = "Bearer sk_prod_real_abcdefghij";  // TODO: Load from config
    private final OkHttpClient client;
    private final Gson gson;

    public EkgAiService() {
        this.client = new OkHttpClient.Builder()
                .connectTimeout(5, TimeUnit.SECONDS)
                .readTimeout(60, TimeUnit.SECONDS)
                .build();
        this.gson = new Gson();
    }

    // ========== SYNCHRONOUS ANALYSIS (Original) ==========

    /**
     * Old method: Blocks until complete (5s for LLM).
     * Use for quick analysis when blocking is acceptable.
     */
    public JsonObject analyzeEcgById(int ecgId, double age, String sex, String symptoms) throws IOException {
        JsonObject req = new JsonObject();
        req.addProperty("ecg_id", ecgId);
        req.addProperty("patient_age", age);
        req.addProperty("patient_sex", sex);
        req.addProperty("symptoms", symptoms);
        req.addProperty("output_format", "both");

        return executeAnalysis(BASE_URL + "/ecg/analyze", req);
    }

    /**
     * Synchronous analysis with raw signal data.
     */
    public JsonObject analyzeEcgSignal(double[][] signal, double age, String sex, String symptoms) throws IOException {
        JsonObject req = new JsonObject();
        req.add("signal", gson.toJsonTree(signal));
        req.addProperty("patient_age", age);
        req.addProperty("patient_sex", sex);
        req.addProperty("symptoms", symptoms);
        req.addProperty("output_format", "both");

        return executeAnalysis(BASE_URL + "/ecg/analyze", req);
    }

    // ========== ASYNCHRONOUS ANALYSIS (Phase 5 Step 1) ==========

    /**
     * Start async analysis → Returns job_id immediately.
     * Use for EMR UI to avoid blocking (recommended).
     * 
     * @return Job ID for polling
     */
    public String analyzeEcgAsync(int ecgId, double age, String sex, String symptoms) throws IOException {
        JsonObject req = new JsonObject();
        req.addProperty("ecg_id", ecgId);
        req.addProperty("patient_age", age);
        req.addProperty("patient_sex", sex);
        req.addProperty("symptoms", symptoms);
        req.addProperty("output_format", "both");

        JsonObject resp = executeAnalysis(BASE_URL + "/ecg/analyze-async", req);
        return resp.get("job_id").getAsString();
    }

    /**
     * Check job progress.
     * 
     * @return {"status": "queued|in_progress|complete|failed|cancelled",
     *          "progress": 0–100,
     *          "result": {...} (if complete)}
     */
    public JsonObject getJobStatus(String jobId) throws IOException {
        Request request = new Request.Builder()
                .url(BASE_URL + "/jobs/" + jobId)
                .addHeader("Authorization", API_KEY)
                .get()
                .build();

        try (Response response = client.newCall(request).execute()) {
            if (!response.isSuccessful()) {
                throw new IOException("Job query failed: " + response.code());
            }
            return gson.fromJson(response.body().string(), JsonObject.class);
        }
    }

    /**
     * Poll job until complete or timeout.
     * 
     * @param jobId Job ID
     * @param maxWaitMs Maximum wait time (60000 = 60s)
     * @param pollIntervalMs Poll interval (1000 = 1s)
     * @param onProgress Callback for progress updates (optional)
     * @return Complete analysis result
     */
    public JsonObject waitForResult(String jobId, long maxWaitMs, long pollIntervalMs,
                                    ProgressCallback onProgress) throws IOException, InterruptedException {
        long startTime = System.currentTimeMillis();

        while (System.currentTimeMillis() - startTime < maxWaitMs) {
            JsonObject status = getJobStatus(jobId);
            String statusStr = status.get("status").getAsString();
            int progress = status.get("progress").getAsInt();

            // Fire progress callback
            if (onProgress != null) {
                onProgress.onProgress(progress, statusStr);
            }

            if ("complete".equals(statusStr)) {
                return status.getAsJsonObject("result");
            } else if ("failed".equals(statusStr) || "cancelled".equals(statusStr)) {
                throw new IOException("Job " + statusStr + ": " + status.get("error").getAsString());
            }

            Thread.sleep(pollIntervalMs);
        }

        throw new IOException("Job timeout after " + maxWaitMs + "ms");
    }

    /**
     * Cancel running or queued job.
     */
    public void cancelJob(String jobId) throws IOException {
        Request request = new Request.Builder()
                .url(BASE_URL + "/jobs/" + jobId)
                .addHeader("Authorization", API_KEY)
                .delete()
                .build();

        try (Response response = client.newCall(request).execute()) {
            if (!response.isSuccessful()) {
                throw new IOException("Cancel failed: " + response.code());
            }
        }
    }

    // ========== INPUT FORMATS (Phase 5 Step 4) ==========

    /**
     * Analyze DICOM file.
     * Use for PACS integration or DICOM ECG files.
     */
    public String analyzeDicomAsync(File dicomFile, String outputFormat) throws IOException {
        MultipartBody.Builder builder = new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("file", dicomFile.getName(),
                        RequestBody.create(dicomFile, MediaType.parse("application/octet-stream")))
                .addFormDataPart("output_format", outputFormat);

        RequestBody body = builder.build();

        Request request = new Request.Builder()
                .url(BASE_URL + "/ecg/analyze-dicom")
                .addHeader("Authorization", API_KEY)
                .post(body)
                .build();

        try (Response response = client.newCall(request).execute()) {
            if (!response.isSuccessful()) {
                throw new IOException("DICOM analysis failed: " + response.code());
            }
            JsonObject result = gson.fromJson(response.body().string(), JsonObject.class);
            return result.get("request_id").getAsString();  // For future tracking
        }
    }

    /**
     * Analyze HL7 message with ECG signal.
     * Use for EMR/HIS integration.
     * 
     * @param hl7Message HL7 v2.5 message (MSH|PID|OBX segments)
     * @param ecgSignal 12-lead signal as double[][]
     */
    public String analyzeHl7Async(String hl7Message, double[][] ecgSignal, String outputFormat) throws IOException {
        JsonObject req = new JsonObject();
        req.addProperty("message", hl7Message);
        req.add("signal", gson.toJsonTree(ecgSignal));
        req.addProperty("output_format", outputFormat);

        RequestBody body = RequestBody.create(
                req.toString(),
                MediaType.parse("application/json; charset=utf-8")
        );

        Request request = new Request.Builder()
                .url(BASE_URL + "/ecg/analyze-hl7")
                .addHeader("Authorization", API_KEY)
                .post(body)
                .build();

        try (Response response = client.newCall(request).execute()) {
            if (!response.isSuccessful()) {
                throw new IOException("HL7 analysis failed: " + response.code());
            }
            JsonObject result = gson.fromJson(response.body().string(), JsonObject.class);
            return result.get("request_id").getAsString();
        }
    }

    /**
     * Analyze SCP-ECG file.
     * Use for direct ECG device integration (GE, Philips, Mortara).
     */
    public String analyzeScpAsync(File scpFile, String outputFormat) throws IOException {
        MultipartBody.Builder builder = new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("file", scpFile.getName(),
                        RequestBody.create(scpFile, MediaType.parse("application/octet-stream")))
                .addFormDataPart("output_format", outputFormat);

        RequestBody body = builder.build();

        Request request = new Request.Builder()
                .url(BASE_URL + "/ecg/analyze-scp")
                .addHeader("Authorization", API_KEY)
                .post(body)
                .build();

        try (Response response = client.newCall(request).execute()) {
            if (!response.isSuccessful()) {
                throw new IOException("SCP analysis failed: " + response.code());
            }
            JsonObject result = gson.fromJson(response.body().string(), JsonObject.class);
            return result.get("request_id").getAsString();
        }
    }

    // ========== HELPER METHODS ==========

    private JsonObject executeAnalysis(String url, JsonObject request) throws IOException {
        RequestBody body = RequestBody.create(
                request.toString(),
                MediaType.parse("application/json; charset=utf-8")
        );

        Request httpRequest = new Request.Builder()
                .url(url)
                .addHeader("Authorization", API_KEY)
                .post(body)
                .build();

        try (Response response = client.newCall(httpRequest).execute()) {
            if (!response.isSuccessful()) {
                throw new IOException("API Error: " + response.code() + " " + response.body().string());
            }
            return gson.fromJson(response.body().string(), JsonObject.class);
        }
    }

    /**
     * Callback interface for progress updates.
     */
    public interface ProgressCallback {
        void onProgress(int progress, String status);
    }
}
```

---

## 6. JavaFX UI Integration Examples

### 6.1 Async Analysis with Progress Monitoring

```java
// Start async analysis
String jobId = ekgAiService.analyzeEcgAsync(selectedEcgId, patient.getAge(), patient.getSex(), "흉통");

// Poll in background thread
Task<JsonObject> pollTask = new Task<>() {
    @Override
    protected JsonObject call() throws Exception {
        return ekgAiService.waitForResult(
            jobId,
            60000,  // 60s timeout
            1000,   // Poll every 1s
            (progress, status) -> {
                updateProgress(progress, 100);
                updateMessage(status);  // Show "queued", "in_progress", etc.
            }
        );
    }
};

// Update UI when done
pollTask.setOnSucceeded(e -> {
    JsonObject result = pollTask.getValue();
    displayAnalysisResult(result);
});

// Show cancel button during analysis
cancelButton.setOnAction(e -> {
    try {
        ekgAiService.cancelJob(jobId);
        pollTask.cancel();
    } catch (IOException ex) {
        showError("Cancel failed: " + ex.getMessage());
    }
});

new Thread(pollTask).start();
```

### 6.2 DICOM File Analysis

```java
// User selects DICOM file from PACS
FileChooser chooser = new FileChooser();
chooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("DICOM Files", "*.dcm"));
File dicomFile = chooser.showOpenDialog(window);

if (dicomFile != null) {
    Task<JsonObject> dicomTask = new Task<>() {
        @Override
        protected JsonObject call() throws Exception {
            String jobId = ekgAiService.analyzeDicomAsync(dicomFile, "both");
            return ekgAiService.waitForResult(jobId, 60000, 1000, (p, s) -> {
                updateProgress(p, 100);
                updateMessage("DICOM Analysis: " + s);
            });
        }
    };

    dicomTask.setOnSucceeded(e -> {
        JsonObject result = dicomTask.getValue();
        displayAnalysisResult(result);
    });

    new Thread(dicomTask).start();
}
```

### 6.3 HL7 EMR Integration

```java
// Receive HL7 message from EMR system
String hl7Message = emrSystem.getPatientEcgMessage(patientId);
double[][] ecgSignal = emrSystem.getEcgSignal(patientId);

Task<JsonObject> hl7Task = new Task<>() {
    @Override
    protected JsonObject call() throws Exception {
        String jobId = ekgAiService.analyzeHl7Async(hl7Message, ecgSignal, "both");
        return ekgAiService.waitForResult(jobId, 60000, 1000, null);
    }
};

hl7Task.setOnSucceeded(e -> {
    JsonObject result = hl7Task.getValue();
    // Patient info already extracted from HL7
    String patientName = result.getAsJsonObject("patient_info").get("patient_name").getAsString();
    displayAnalysisResult(result);
});

new Thread(hl7Task).start();
```

---

## 7. Configuration & Deployment

### 7.1 Settings (application.properties or config class)

```properties
# EKG AI Service Configuration
ekg.api.url=http://localhost:8000/api/v1
ekg.api.key=sk_prod_real_abcdefghij
ekg.async.enabled=true
ekg.async.timeout.ms=60000
ekg.async.poll.interval.ms=1000
ekg.dicom.support=true
ekg.hl7.support=true
ekg.scp.support=true
```

### 7.2 Maven Dependencies

```xml
<!-- OkHttp -->
<dependency>
    <groupId>com.squareup.okhttp3</groupId>
    <artifactId>okhttp</artifactId>
    <version>4.11.0</version>
</dependency>

<!-- Gson -->
<dependency>
    <groupId>com.google.code.gson</groupId>
    <artifactId>gson</artifactId>
    <version>2.10.1</version>
</dependency>

<!-- For DICOM file handling (optional) -->
<dependency>
    <groupId>org.dcm4che</groupId>
    <artifactId>dcm4che-core</artifactId>
    <version>5.29.1</version>
</dependency>

<!-- For HL7 message handling (optional) -->
<dependency>
    <groupId>ca.uhn.hapi</groupId>
    <artifactId>hapi-structures-v25</artifactId>
    <version>2.4</version>
</dependency>
```

---

## 8. Error Handling & Resilience

```java
try {
    // Start analysis
    String jobId = ekgAiService.analyzeEcgAsync(ecgId, age, sex, symptoms);

    // Wait for result with timeout
    JsonObject result = ekgAiService.waitForResult(
        jobId,
        60000,  // 60s max
        1000,   // 1s poll
        null
    );

    displayAnalysisResult(result);

} catch (InterruptedException e) {
    // User cancelled or timeout
    handleAnalysisCancelled();
} catch (IOException e) {
    // Network or server error
    showError("Analysis failed: " + e.getMessage());
    // Optionally retry with exponential backoff
}
```

---

## 9. Compliance & Security

### Authentication
- All requests include `Authorization: Bearer <API_KEY>`
- API key loaded from secure configuration (not hardcoded)

### HIPAA Compliance
- Patient data never stored in logs
- Analysis results include legal disclaimer
- Audit trail maintained server-side (JSONL)

### Rate Limiting
- 60 requests/minute per API key
- 1000 requests/day per API key
- Remaining quota returned in response headers

---

## 10. Migration Path (Phase 4 → Phase 5)

| Phase | Synchronous | Asynchronous | DICOM | HL7 | SCP |
|-------|-------------|--------------|-------|-----|-----|
| Phase 4 | ✅ | ❌ | ❌ | ❌ | ❌ |
| Phase 5 Step 1 | ✅ | ✅ | ❌ | ❌ | ❌ |
| Phase 5 Step 4 | ✅ | ✅ | ✅ | ✅ | ✅ |
| Phase 5 Step 5 | ✅ | ✅ | ✅ | ✅ | ✅ |

**Recommendation:** Use async API for EMR (non-blocking), keep sync for quick demos.

---

## 11. Support & Contact

For integration issues:
1. Check API health: `GET http://localhost:8000/api/v1/health`
2. Review logs: `logs/audit/YYYYMMDD.jsonl`
3. Test endpoints: Use Postman or cURL
4. Contact: ekg-assistant project team
