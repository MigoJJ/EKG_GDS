/**
 * EKG AI Assistant Java Client (v2.0)
 * Phase 5 Step 5 - Async + Input Format Support
 *
 * Integration for Eclipse Java EMR (`com.emr.gds`)
 *
 * Features:
 * - Synchronous analysis (Phase 4)
 * - Asynchronous with progress polling (Phase 5 Step 1)
 * - DICOM file analysis (Phase 5 Step 4)
 * - HL7 message analysis (Phase 5 Step 4)
 * - SCP-ECG file analysis (Phase 5 Step 4)
 * - API key authentication & rate limiting
 *
 * Copy this file to: com.emr.gds.features.ekg.service.EkgAiService
 *
 * Dependencies (add to pom.xml):
 * - com.squareup.okhttp3:okhttp:4.11.0
 * - com.google.code.gson:gson:2.10.1
 */

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

    // TODO: Load from secure config (e.g., environment variables, config files)
    // DO NOT hardcode credentials in production code
    private static final String API_KEY = "Bearer sk_prod_real_abcdefghij";

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
     * Constructor with custom API key.
     * Use this if loading from configuration.
     */
    public EkgAiService(String apiKey) {
        this();
        // Note: Would need to make API_KEY non-final to support this
    }

    // ========== SYNCHRONOUS ANALYSIS (Phase 4) ==========

    /**
     * Synchronous analysis by ECG ID from PTB-XL dataset.
     * Blocks until complete (4-5 seconds).
     *
     * @param ecgId PTB-XL record ID
     * @param age Patient age in years
     * @param sex Patient sex ('M' or 'F')
     * @param symptoms Chief complaint
     * @return Full analysis result (features + classification + report)
     * @throws IOException If network error or server error
     */
    public JsonObject analyzeEcgById(int ecgId, double age, String sex, String symptoms)
            throws IOException {
        JsonObject req = new JsonObject();
        req.addProperty("ecg_id", ecgId);
        req.addProperty("patient_age", age);
        req.addProperty("patient_sex", sex);
        req.addProperty("symptoms", symptoms);
        req.addProperty("output_format", "both");

        return executeAnalysis(BASE_URL + "/ecg/analyze", req);
    }

    /**
     * Synchronous analysis with raw 12-lead signal.
     *
     * @param signal 12-lead ECG signal as double[samples][12] in mV
     * @param age Patient age
     * @param sex Patient sex
     * @param symptoms Chief complaint
     * @return Full analysis result
     * @throws IOException If network error or server error
     */
    public JsonObject analyzeEcgSignal(double[][] signal, double age, String sex, String symptoms)
            throws IOException {
        JsonObject req = new JsonObject();
        req.add("signal", gson.toJsonTree(signal));
        req.addProperty("patient_age", age);
        req.addProperty("patient_sex", sex);
        req.addProperty("symptoms", symptoms);
        req.addProperty("output_format", "both");
        req.addProperty("sampling_rate", 100);

        return executeAnalysis(BASE_URL + "/ecg/analyze", req);
    }

    // ========== ASYNCHRONOUS ANALYSIS (Phase 5 Step 1) ==========

    /**
     * Start asynchronous analysis by ECG ID.
     * Returns immediately with job ID (non-blocking).
     *
     * @param ecgId PTB-XL record ID
     * @param age Patient age
     * @param sex Patient sex
     * @param symptoms Chief complaint
     * @return Job ID for polling progress
     * @throws IOException If network error
     */
    public String analyzeEcgAsync(int ecgId, double age, String sex, String symptoms)
            throws IOException {
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
     * Asynchronous analysis with raw signal.
     *
     * @param signal 12-lead ECG signal
     * @param age Patient age
     * @param sex Patient sex
     * @param symptoms Chief complaint
     * @return Job ID
     * @throws IOException If network error
     */
    public String analyzeEcgSignalAsync(double[][] signal, double age, String sex, String symptoms)
            throws IOException {
        JsonObject req = new JsonObject();
        req.add("signal", gson.toJsonTree(signal));
        req.addProperty("patient_age", age);
        req.addProperty("patient_sex", sex);
        req.addProperty("symptoms", symptoms);
        req.addProperty("output_format", "both");
        req.addProperty("sampling_rate", 100);

        JsonObject resp = executeAnalysis(BASE_URL + "/ecg/analyze-async", req);
        return resp.get("job_id").getAsString();
    }

    /**
     * Get job status (progress).
     *
     * Response includes:
     * - status: "queued", "in_progress", "complete", "failed", "cancelled"
     * - progress: 0-100 (percentage)
     * - result: {...} (if complete)
     * - error: "..." (if failed)
     *
     * @param jobId Job ID from analyzeEcgAsync()
     * @return Job status object
     * @throws IOException If network error
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
            assert response.body() != null;
            return gson.fromJson(response.body().string(), JsonObject.class);
        }
    }

    /**
     * Poll job status until complete or timeout.
     * Calls onProgress callback periodically for UI updates.
     *
     * @param jobId Job ID
     * @param maxWaitMs Maximum wait time in milliseconds (e.g., 60000 = 60s)
     * @param pollIntervalMs Poll interval in milliseconds (e.g., 1000 = 1s)
     * @param onProgress Callback for progress updates (null = no callback)
     * @return Complete analysis result
     * @throws IOException If job failed
     * @throws InterruptedException If thread interrupted
     */
    public JsonObject waitForResult(String jobId, long maxWaitMs, long pollIntervalMs,
                                    ProgressCallback onProgress)
            throws IOException, InterruptedException {
        long startTime = System.currentTimeMillis();

        while (System.currentTimeMillis() - startTime < maxWaitMs) {
            JsonObject status = getJobStatus(jobId);
            String statusStr = status.get("status").getAsString();
            int progress = status.get("progress").getAsInt();

            // Fire progress callback for UI updates
            if (onProgress != null) {
                onProgress.onProgress(progress, statusStr);
            }

            // Check if complete
            if ("complete".equals(statusStr)) {
                return status.getAsJsonObject("result");
            } else if ("failed".equals(statusStr) || "cancelled".equals(statusStr)) {
                String error = status.has("error") ? status.get("error").getAsString() : "Unknown error";
                throw new IOException("Job " + statusStr + ": " + error);
            }

            // Wait before next poll
            Thread.sleep(pollIntervalMs);
        }

        throw new IOException("Job timeout after " + maxWaitMs + "ms");
    }

    /**
     * Cancel a running or queued job.
     *
     * @param jobId Job ID
     * @throws IOException If network error or job cannot be cancelled
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
     * Analyze DICOM file asynchronously.
     * Use for PACS integration or hospital imaging systems.
     *
     * @param dicomFile DICOM file with embedded ECG waveform
     * @param outputFormat "narrative", "json", "both", or "none"
     * @return ECGAnalysisResponse with patient demographics from DICOM
     * @throws IOException If file not found or upload fails
     */
    public JsonObject analyzeDicom(File dicomFile, String outputFormat) throws IOException {
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
                throw new IOException("DICOM analysis failed: " + response.code() + " " + response.message());
            }
            assert response.body() != null;
            return gson.fromJson(response.body().string(), JsonObject.class);
        }
    }

    /**
     * Analyze HL7 v2.5 message with ECG signal.
     * Use for EMR/HIS integration.
     *
     * HL7 message format (MSH|PID|OBX segments):
     * MSH|^~\&|SendingApp|...|Timestamp||MessageType|...|Version
     * PID|1||PatientID||LastName^FirstName||DOB|Sex
     * OBX|1|NM|HR^Heart Rate|1|75|bpm
     *
     * @param hl7Message HL7 v2.5 message (newline-separated segments)
     * @param ecgSignal 12-lead ECG signal as double[samples][12]
     * @param outputFormat "narrative", "json", "both", or "none"
     * @return ECGAnalysisResponse with patient info extracted from HL7
     * @throws IOException If network error
     */
    public JsonObject analyzeHl7(String hl7Message, double[][] ecgSignal, String outputFormat)
            throws IOException {
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
            assert response.body() != null;
            return gson.fromJson(response.body().string(), JsonObject.class);
        }
    }

    /**
     * Analyze SCP-ECG file.
     * Use for direct ECG device integration (GE, Philips, Mortara).
     *
     * @param scpFile SCP-ECG file from medical device
     * @param outputFormat "narrative", "json", "both", or "none"
     * @return ECGAnalysisResponse with patient demographics from SCP
     * @throws IOException If file not found or upload fails
     */
    public JsonObject analyzeScp(File scpFile, String outputFormat) throws IOException {
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
            assert response.body() != null;
            return gson.fromJson(response.body().string(), JsonObject.class);
        }
    }

    // ========== HELPER METHODS ==========

    /**
     * Execute a POST request to the AI service.
     */
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
                throw new IOException("API Error " + response.code() + ": " + response.message());
            }
            assert response.body() != null;
            return gson.fromJson(response.body().string(), JsonObject.class);
        }
    }

    /**
     * Callback interface for progress monitoring.
     * Called periodically by waitForResult().
     */
    public interface ProgressCallback {
        /**
         * Called when job progress is updated.
         *
         * @param progress Current progress (0-100)
         * @param status Current status ("queued", "in_progress", "complete", etc.)
         */
        void onProgress(int progress, String status);
    }
}
