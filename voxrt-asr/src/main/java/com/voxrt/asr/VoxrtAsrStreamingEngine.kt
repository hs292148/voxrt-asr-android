package com.voxrt.asr

import android.content.res.AssetFileDescriptor
import java.io.Closeable

/**
 * Idiomatic Kotlin wrapper around the [VoxrtAsrNative] streaming
 * surface — modelled on `VoxrtSileroVadEngine` in the sibling VAD
 * library.
 *
 * The engine is a **stateful synchronous function** — not an actor
 * with its own worker thread. The caller decides where to run the
 * inference loop (mic-capture thread, IO-bound replay thread, etc.)
 * and pumps PCM into [processPcm]. Text deltas come back as the
 * return value of the same call.
 *
 * Typical live-mic usage:
 *
 * ```kotlin
 * VoxrtAsrStreamingEngine
 *     .fromAssetFd(context.assets.openFd("streaming_medium_pc.vxrt"))
 *     .use { engine ->
 *         while (recording) {
 *             val pcm = readMicAsF32()
 *             val delta = engine.processPcm(pcm)
 *             if (delta.isNotEmpty()) onUiThread { transcript += delta }
 *         }
 *         val tail = engine.stop()
 *         if (tail.isNotEmpty()) onUiThread { transcript += tail }
 *     }
 * ```
 *
 * Instances are built through the [fromAssetFd] / [fromBytes]
 * factory methods on the companion (matches `VoxrtSileroVadEngine`).
 * The primary constructor is `private` — there's no way to construct
 * an engine without a successful native session.
 *
 * For file-replay, the same shape with the PCM chunked into the
 * loop. See [transcribeAll] for a convenience helper.
 *
 * **Audio contract:** mono 16 kHz `FloatArray` in `[-1, 1]`. No
 * automatic resampling — feeding 44.1 kHz / 48 kHz yields gibberish
 * transcripts. See the public README.
 *
 * **Threading:** per-instance, **not** thread-safe. Serialise
 * [processPcm] / [stop] / [reset] / [close] against each other on a
 * given instance. `@Synchronized` here only protects against
 * accidental double-free / double-close on race conditions — it
 * does not make concurrent transcription correct.
 *
 * **Lifecycle:** one instance handles a stream of utterances. Call
 * [reset] between utterances to zero the K/V cache + LSTM state
 * without paying weight-load cost again. Call [close] (or use
 * `.use { }`) when done — leaks the native session otherwise.
 */
class VoxrtAsrStreamingEngine private constructor(
    private var handle: Long,
) : Closeable {

    /**
     * Feed a block of mono 16 kHz f32 PCM. Returns the text emitted
     * by any chunks that completed during this call — often the
     * empty string while the internal buffer fills up to the next
     * 1.12 s chunk boundary.
     *
     * `pcm` may be any length. The native session accumulates into
     * its steady-state chunk size (17 920 samples) and drives
     * inference at chunk boundaries.
     *
     * Throws [IllegalStateException] if the engine is closed or the
     * native call returned a sentinel error.
     */
    @Synchronized
    fun processPcm(pcm: FloatArray): String {
        check(handle != 0L) { "VoxrtAsrStreamingEngine is closed" }
        return VoxrtAsrNative.streamingPushAudio(handle, pcm)
            ?: throw IllegalStateException(
                "VoxrtAsrNative.streamingPushAudio returned null (native error)"
            )
    }

    /**
     * Flush remaining buffered audio and emit the final tail.
     * Call once per utterance, AFTER the last [processPcm]. Returns
     * the leftover text — may be empty if the last chunk emitted
     * everything.
     *
     * After [stop] the engine is in a "stopped" state — call
     * [reset] before pushing audio for the next utterance, or
     * [close] to release the native session.
     */
    @Synchronized
    fun stop(): String {
        check(handle != 0L) { "VoxrtAsrStreamingEngine is closed" }
        return VoxrtAsrNative.streamingStop(handle) ?: ""
    }

    /**
     * Zero the per-utterance state (K/V cache, LSTM, audio tail,
     * cumulative tokens) so the same handle can transcribe a new
     * utterance. Keeps the loaded weights in memory — no model
     * re-load cost.
     */
    @Synchronized
    fun reset() {
        check(handle != 0L) { "VoxrtAsrStreamingEngine is closed" }
        VoxrtAsrNative.streamingReset(handle)
    }

    /** Release the native session. Idempotent. */
    @Synchronized
    override fun close() {
        if (handle != 0L) {
            VoxrtAsrNative.streamingDestroy(handle)
            handle = 0L
        }
    }

    // ── Optional diagnostics surface ───────────────────────────────────────

    /**
     * Snapshot of per-stage timing accumulators since the session
     * started (or since the last [resetStageTimings] /
     * [reset] call). Eight-element layout:
     *
     *     [0] mel_total_us         [1] mel_count
     *     [2] subsampling_total_us [3] subsampling_count
     *     [4] encoder_total_us     [5] encoder_count
     *     [6] decoder_total_us     [7] decoder_count
     *
     * Returns null if the engine is closed.
     */
    @Synchronized
    fun stageTimings(): LongArray? =
        if (handle != 0L) VoxrtAsrNative.streamingStageTimings(handle) else null

    /**
     * Snapshot of per-Conformer-sub-block encoder timings. Same
     * eight-element layout shape as [stageTimings] but the four
     * pairs are FFN1 / MHA / Conv / FFN2 instead of mel / sub /
     * enc / dec.
     */
    @Synchronized
    fun encoderSubTimings(): LongArray? =
        if (handle != 0L) VoxrtAsrNative.streamingEncoderSubTimings(handle) else null

    /** Zero the per-stage timing accumulators without touching
     *  inference state. */
    @Synchronized
    fun resetStageTimings() {
        if (handle != 0L) VoxrtAsrNative.streamingResetStageTimings(handle)
    }

    /**
     * Convenience: drain a full `FloatArray` (mono 16 kHz f32)
     * through the engine in fixed-size blocks. Returns the complete
     * transcript (deltas concatenated + the tail from [stop]).
     * Useful for file replay and unit tests — for live mic, drive
     * [processPcm] directly from the capture loop instead.
     */
    @JvmOverloads
    fun transcribeAll(
        pcm: FloatArray,
        blockSamples: Int = 3_200,  // 200 ms @ 16 kHz
    ): String {
        require(blockSamples > 0)
        val sb = StringBuilder()
        var off = 0
        while (off < pcm.size) {
            val end = minOf(off + blockSamples, pcm.size)
            val block = if (off == 0 && end == pcm.size) pcm
                        else pcm.copyOfRange(off, end)
            sb.append(processPcm(block))
            off = end
        }
        sb.append(stop())
        return sb.toString()
    }

    companion object {
        /** The only sample rate the engine accepts. */
        const val SAMPLE_RATE_HZ = 16_000

        /** Internal chunk size. Inference fires every time the
         *  accumulator reaches this many samples (~1.12 s). */
        val CHUNK_AUDIO_SAMPLES: Int get() = VoxrtAsrNative.streamingChunkAudioSamples()

        /**
         * Build an engine by `mmap`'ing an open Android asset
         * file-descriptor. **Recommended** for bundled-in-APK
         * `.vxrt` models — zero managed-heap copy, ~63 MB less
         * peak memory at session start than the bytes path.
         *
         * Caller retains ownership of [assetFd] and may close it
         * as soon as this function returns — the native session
         * has copied weights out of the mmap by then.
         *
         * Requires `androidResources { noCompress.add("vxrt") }`
         * in the consumer app's `build.gradle.kts` so AAPT keeps
         * the asset stored-as-is (compressed assets fall back to
         * `ByteArray` and break the mmap path).
         *
         * @param decodeMode [VoxrtAsrNative.DECODE_RNNT] (default,
         *   recommended) or [VoxrtAsrNative.DECODE_CTC]. TDT is
         *   not supported on `streaming-medium-pc`.
         * @throws IllegalArgumentException if the native session
         *   could not be created (asset compressed, `.vxrt` parse
         *   failure, unsupported decode mode).
         */
        @JvmStatic @JvmOverloads
        fun fromAssetFd(
            assetFd: AssetFileDescriptor,
            decodeMode: Int = VoxrtAsrNative.DECODE_RNNT,
        ): VoxrtAsrStreamingEngine {
            val h = VoxrtAsrNative.streamingCreateFromFd(
                assetFd.parcelFileDescriptor.fd,
                assetFd.startOffset,
                assetFd.length,
                decodeMode,
            )
            require(h != 0L) {
                "VoxrtAsrStreamingEngine.fromAssetFd: native create returned 0 — " +
                    "asset uncompressed? `.vxrt` valid? decode mode supported?"
            }
            return VoxrtAsrStreamingEngine(h)
        }

        /**
         * Build an engine from a `.vxrt` byte buffer in memory.
         * Use this when the model was **downloaded** at first run
         * (i.e. lives in `context.filesDir` rather than `assets/`)
         * and you've already read it into a `ByteArray`.
         *
         * **Memory cost:** the byte array goes through the JVM
         * heap then is copied across the JNI boundary into Rust
         * heap during deserialization. Peak memory at create
         * time is roughly 2 × model size. For bundled assets,
         * prefer [fromAssetFd] which mmaps zero-copy instead.
         *
         * @throws IllegalArgumentException on the same conditions
         *   as [fromAssetFd].
         */
        @JvmStatic @JvmOverloads
        fun fromBytes(
            modelBytes: ByteArray,
            decodeMode: Int = VoxrtAsrNative.DECODE_RNNT,
        ): VoxrtAsrStreamingEngine {
            val h = VoxrtAsrNative.streamingCreate(modelBytes, decodeMode)
            require(h != 0L) {
                "VoxrtAsrStreamingEngine.fromBytes: native create returned 0 — " +
                    "`.vxrt` valid? decode mode supported?"
            }
            return VoxrtAsrStreamingEngine(h)
        }
    }
}
