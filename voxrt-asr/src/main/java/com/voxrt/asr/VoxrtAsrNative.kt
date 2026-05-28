package com.voxrt.asr

/**
 * Kotlin facade for the VoxRT ASR native library.
 *
 * The companion `init` block calls `System.loadLibrary("voxrt_asr")`
 * once per process — Android resolves to `libvoxrt_asr.so` under the
 * appropriate ABI directory in the APK. The `external fun`
 * declarations match the JNI symbols exported by
 * `crates/asr/src/lib.rs`.
 *
 * Two parallel pipelines:
 *   - **offline**: one-shot transcription on a full audio buffer
 *     (`offlineCreate` / `offlineTranscribe` / `offlineDestroy`).
 *   - **streaming**: cache-aware live inference
 *     (`streamingCreate` / `streamingPushAudio` / `streamingStop` /
 *     `streamingReset` / `streamingDestroy`).
 *
 * Both surface return `0` for "no handle" / errors; `null` returned
 * from text-returning methods signals an error. Empty string is a
 * valid "no text emitted yet" result for streaming chunks.
 *
 * State (K/V caches, LSTM, audio_tail, cumulative tokens) lives
 * **inside** the native handle (deviating from Silero's stateless
 * pattern; see `docs/decisions/0031-voxrt-asr-c-abi.md`).
 *
 * Caller MUST serialise calls against a given handle.
 */
object VoxrtAsrNative {
    init {
        System.loadLibrary("voxrt_asr")
    }

    /** SDK version string. */
    external fun voxrtAsrVersion(): String

    // ── Decode mode constants (match `crates/asr/src/c_api.rs`) ─────────────

    const val DECODE_CTC = 0
    const val DECODE_RNNT = 1
    const val DECODE_TDT = 2

    // ── Offline batch transcription ─────────────────────────────────────────

    /** Build offline ASR handle from `.vxrt` bytes. Returns 0 on any error. */
    external fun offlineCreate(modelBytes: ByteArray): Long

    /** Build offline ASR handle from an open file-descriptor pointing at
     *  a `.vxrt` blob (typically `AssetFileDescriptor.parcelFileDescriptor`
     *  for an uncompressed asset). The native side `mmap`s the slice
     *  `[offset, offset+length)` directly — no copy through the Java
     *  heap and no temp buffer. Caller retains ownership of `fd` and
     *  must close it after this call returns. Returns 0 on any error. */
    external fun offlineCreateFromFd(fd: Int, offset: Long, length: Long): Long

    /** Reclaim the offline handle. Idempotent on 0. */
    external fun offlineDestroy(handle: Long)

    /** Transcribe a full PCM buffer (f32 mono 16 kHz). Returns the
     *  full transcript on success, or `null` on error (bad handle,
     *  unsupported decode mode, native panic). */
    external fun offlineTranscribe(handle: Long, pcm: FloatArray, decodeMode: Int): String?

    /** Snapshot offline stage timings — same 8-element layout as
     *  [streamingStageTimings]. Accumulates across transcribe calls. */
    external fun offlineStageTimings(handle: Long): LongArray?

    /** Snapshot offline encoder per-sub-block timings — same 8-element
     *  layout as [streamingEncoderSubTimings]. Counts are
     *  `layers × transcribe_calls`. */
    external fun offlineEncoderSubTimings(handle: Long): LongArray?

    /** Zero the offline per-stage + per-sub-block accumulators. */
    external fun offlineResetStageTimings(handle: Long)

    // ── True streaming session ──────────────────────────────────────────────

    /** Build a streaming session. `decodeMode` must be DECODE_CTC or
     *  DECODE_RNNT — TDT is rejected on this model. Returns 0 on
     *  error. */
    external fun streamingCreate(modelBytes: ByteArray, decodeMode: Int): Long

    /** Build a streaming session from an open file-descriptor pointing
     *  at a `.vxrt` blob (typically `AssetFileDescriptor.parcelFileDescriptor`
     *  for an uncompressed asset). Same mmap zero-copy semantics as
     *  [offlineCreateFromFd]. Caller retains ownership of `fd` and
     *  must close it after this call. Returns 0 on error (incl. TDT
     *  rejected on this model). */
    external fun streamingCreateFromFd(fd: Int, offset: Long, length: Long, decodeMode: Int): Long

    /** Reclaim the streaming handle. Idempotent on 0. */
    external fun streamingDestroy(handle: Long)

    /** Zero the per-utterance streaming state (caches + LSTM +
     *  audio_tail + cumulative tokens). Use before a new utterance
     *  on the same handle. */
    external fun streamingReset(handle: Long)

    /** Push a block of f32 mono 16 kHz PCM. Returns the incremental
     *  text emitted by chunks that completed during this call (may
     *  be empty if no chunk completed yet), or `null` on error. */
    external fun streamingPushAudio(handle: Long, pcm: FloatArray): String?

    /** Flush remaining buffered audio and emit final text. */
    external fun streamingStop(handle: Long): String?

    /** Audio samples consumed per steady-state streaming chunk
     *  (= 17920 for streaming-medium-pc). Informational. */
    external fun streamingChunkAudioSamples(): Int

    /** Look-ahead samples beyond each chunk (= 96 = 6 ms).
     *  Informational. */
    external fun streamingLookAheadSamples(): Int

    // ── Per-stage timing (mel / subsampling / encoder / decoder) ────

    /** Snapshot the per-stage timing accumulators. Returns a
     *  [LongArray] of length 8:
     *
     *      [0] mel_total_us         [1] mel_count
     *      [2] subsampling_total_us [3] subsampling_count
     *      [4] encoder_total_us     [5] encoder_count
     *      [6] decoder_total_us     [7] decoder_count
     *
     *  Returns `null` on invalid handle. Counters accumulate over the
     *  session lifetime — call [streamingResetStageTimings] (or
     *  [streamingReset]) to zero them. */
    external fun streamingStageTimings(handle: Long): LongArray?

    /** Snapshot encoder per-sub-block timing accumulators. Returns a
     *  [LongArray] of length 8:
     *
     *      [0] ffn1_total_us  [1] ffn1_count
     *      [2] mha_total_us   [3] mha_count
     *      [4] conv_total_us  [5] conv_count
     *      [6] ffn2_total_us  [7] ffn2_count
     *
     *  Counts are layers × chunks (avg per-layer-per-call =
     *  total / count). Pairs map to Conformer sub-blocks: FFN1 (Macaron
     *  half), rel-pos MHA, ConvModule, FFN2 (Macaron half + LN_out). */
    external fun streamingEncoderSubTimings(handle: Long): LongArray?

    /** Zero the per-stage timing accumulators only. Does not reset
     *  inference state. */
    external fun streamingResetStageTimings(handle: Long)
}
