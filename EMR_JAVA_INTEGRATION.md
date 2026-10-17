# Eclipse Java EMR (`GDSEMREKG`) Integration Specification

This document describes how to integrate the Python EKG AI Assistant service (`http://localhost:8000`) into the existing Eclipse Java EMR application (`com.emr.gds`).

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
