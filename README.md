# VoxrtAsr for Android

Streaming speech recognition on the **VoxRT** custom on-device
inference runtime. NeMo FastConformer (32M parameters), 16 kHz
mono in, P&C-aware text out, cache-aware streaming with ~1.1 s
chunks.

- Current version: `v0.1.1`
- Minimum Android: API 26 (Android 8.0)
- ABIs shipped: `arm64-v8a` (NEON-accelerated), `x86_64` (scalar, emulator only)
- License: Apache-2.0 (Kotlin wrapper) · proprietary (compiled runtime, redistribution allowed via this artifact)
- Upstream model license: CC-BY-4.0 (NVIDIA NeMo)

---

## What is VoxRT?

VoxRT is a from-scratch inference runtime for on-device speech
models. No ONNX Runtime, no PyTorch Mobile, no LiteRT — a custom
Rust core sized and tuned for streaming voice workloads on
phone-class hardware.

`VoxrtAsr` is the streaming-ASR product on that runtime, alongside
the free [`VoxrtSilero`](https://github.com/VoxRT/voxrt-silero-android)
VAD demo. Both share the same runtime crate and the same NEON
kernel set. The runtime is the product; the models are what it
runs.

Commercial wake-word / KWS / domain-specific ASR models built on
the same runtime live at [voxrt.com](https://voxrt.com).

## Performance

Measured at ship time, `arm64-v8a` release builds, post-warmup, RTF =
wall-time-per-chunk ÷ chunk audio duration (lower is better):

| Device                                       | SoC class      | Decoder | Mode        | RTF       |
| -------------------------------------------- | -------------- | ------- | ----------- | --------- |
| Xiaomi Redmi 9C (SD 662, Cortex-A73)         | midrange-2020  | RNN-T   | file replay | **0.302** |
| Xiaomi Redmi 9C (SD 662, Cortex-A73)         | midrange-2020  | RNN-T   | live mic    | **0.353** |
| Samsung Galaxy S9+ (SD 845 / Exynos 9810)    | flagship-2018  | RNN-T   | file replay | **0.240** |
| Samsung Galaxy S9+ (SD 845 / Exynos 9810)    | flagship-2018  | RNN-T   | live mic    | **0.267** |

The live-mic RTF is the production-realistic number — the file
replay is a kernel-only upper bound that excludes `AudioRecord`
capture overhead and scheduler jitter. At RTF 0.35 on a midrange
Android, one core is ~35 % utilised during transcription, leaving
ample headroom for VAD, audio capture, UI, and network. A
flagship-tier SoC (even one from 2018) lands around RTF 0.25 with
the same headroom argument. CTC mode is ~15 % cheaper per chunk
than RNN-T at the cost of marginally lower accuracy (CTC: 4.895 %
WER on LibriSpeech test-clean vs 3.267 % for RNN-T).

## Binary footprint

- Kotlin wrapper, compiled `classes.jar`: ~20 KB
- `libvoxrt_asr.so` (arm64-v8a, stripped): ~600 KB
- `libvoxrt_asr.so` (x86_64, stripped, emulator-only): ~700 KB
- Streaming model `streaming_medium_pc.vxrt`: ~61 MB fp16 on disk
  (downloaded separately, not bundled — see below)
- Native heap at runtime: ~150 MB steady-state (weights expand to
  f32 for inference; mmap'd zero-copy at load time so no
  Java-heap pressure)

Net APK-size impact:
- **arm64-v8a only** (recommended for prod via `splits.abi`):
  ~700 KB binary
- **arm64-v8a + x86_64** (default): ~1.4 MB binary

Filter ABIs in your app's `build.gradle.kts`:

```kotlin
android {
    splits {
        abi {
            isEnable = true
            reset()
            include("arm64-v8a")
            isUniversalApk = false
        }
    }
}
```

## Install

Add JitPack to your `settings.gradle.kts`:

```kotlin
dependencyResolutionManagement {
    repositories {
        // ...
        maven { url = uri("https://jitpack.io") }
    }
}
```

Then in your app's `build.gradle.kts`:

```kotlin
dependencies {
    implementation("com.github.VoxRT:voxrt-asr-android:v0.1.1")
}
```

Gradle resolves it to a pre-built AAR served by JitPack from the
tagged commit of this repo.

## Get the streaming model

The model weights are NOT bundled — you fetch them once from
[`voxrt-asr-models`](https://github.com/VoxRT/voxrt-asr-models/releases/tag/v0.1.1):

```
https://github.com/VoxRT/voxrt-asr-models/releases/download/v0.1.1/streaming_medium_pc.vxrt
```

SHA-256: `0d723e429157a8a8cb58739a1f090574f2f23db311ca7916b43411f5f727c79c`

You decide where it lives. Three common patterns:

- **Bundle in `assets/`** — drop the file into `src/main/assets/`
  and tell AAPT to leave the asset uncompressed so the engine can
  `mmap` it zero-copy through `AssetFileDescriptor` (see
  [Required: `noCompress`](#required-nocompress-on-vxrt-assets) below).
  Works offline from first launch. Adds ~61 MB to your APK.
- **Download on first run** — `OkHttp` / `HttpURLConnection` into
  `context.filesDir`. Smaller APK; needs network at first launch.
  For this path you pass a `ParcelFileDescriptor` opened against the
  downloaded file (e.g. `ParcelFileDescriptor.open(file, MODE_READ_ONLY)`
  → `.toAssetFileDescriptor(...)`).
- **Download on demand** — Play Asset Delivery if you want Play
  Store to host the file.

### Required: `noCompress` on `.vxrt` assets

When you bundle `streaming_medium_pc.vxrt` under `assets/`, add the
following to your **app**'s `build.gradle.kts` so AAPT leaves the
asset stored-as-is and `openFd()` returns a real FD slice:

```kotlin
android {
    androidResources {
        noCompress.add("vxrt")
    }
}
```

Without this the asset is gzip'd inside the APK, `openFd()` falls
back to a decompressed in-RAM `ByteArray`, and you lose the
mmap zero-copy load (peak memory roughly doubles at session start).

## Quick start

```kotlin
import com.voxrt.asr.VoxrtAsrNative
import com.voxrt.asr.VoxrtAsrStreamingEngine

// 1. Open the model as a file-descriptor — no managed-heap copy.
val modelFd = context.assets.openFd("streaming_medium_pc.vxrt")

// 2. Build an engine. RNN-T decoder is the recommended default
//    (higher accuracy); pass VoxrtAsrNative.DECODE_CTC as a second
//    arg if you want the ~15 % cheaper CTC head instead.
val engine = VoxrtAsrStreamingEngine.fromAssetFd(modelFd)
modelFd.close()  // native side has copied weights; FD can go.

// 3. Feed PCM (Float32, 16 kHz, mono, [-1, 1]) blocks of any size.
//    processPcm returns the text emitted during this call — often
//    "" until ~1.12 s of audio has accumulated, then non-empty
//    every chunk boundary.
val delta = engine.processPcm(pcmFloatArray)
if (delta.isNotEmpty()) {
    // delta = new text emitted on this push
}

// For a one-shot file-replay, the engine has a convenience helper:
// val transcript = engine.transcribeAll(pcmFloatArray)

// 4. When the utterance ends, drain the tail and close.
val tail = engine.stop()         // remaining text (may be "")
engine.close()                   // releases native session
```

`engine.processPcm` / `stop` / `reset` / `close` are **synchronous,
stateful, single-thread-at-a-time** — same shape as
`VoxrtSileroVadEngine.processPcm` in the companion VAD library.
The engine does not own a worker thread. You drive it from your
own capture / file-IO thread.

If you downloaded the model on first run instead of bundling it,
use `VoxrtAsrStreamingEngine.fromBytes(downloadedBytes)` —
identical API, only the load path differs.

## Live microphone example

The canonical streaming pattern — Activity owns the worker thread,
`AudioRecord` is the capture source, engine is just a stateful
function.

```kotlin
import android.Manifest
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import com.voxrt.asr.VoxrtAsrStreamingEngine
import kotlin.concurrent.thread

// Caller is responsible for requesting RECORD_AUDIO permission
// before this point — see "Permissions" below.

val sampleRate = 16_000
val minBuf = AudioRecord.getMinBufferSize(
    sampleRate,
    AudioFormat.CHANNEL_IN_MONO,
    AudioFormat.ENCODING_PCM_16BIT,
)
val rec = AudioRecord(
    MediaRecorder.AudioSource.VOICE_RECOGNITION,
    sampleRate, AudioFormat.CHANNEL_IN_MONO,
    AudioFormat.ENCODING_PCM_16BIT,
    maxOf(minBuf, sampleRate / 5 * 2 * 4),  // ≥ 4× 200 ms headroom
)

val modelFd = context.assets.openFd("streaming_medium_pc.vxrt")
val engine = VoxrtAsrStreamingEngine.fromAssetFd(modelFd)
modelFd.close()

thread {
    rec.startRecording()
    val s16 = ShortArray(3200)  // 200 ms block @ 16 kHz
    val f32 = FloatArray(3200)
    val transcript = StringBuilder()
    try {
        while (!stopped) {
            val n = rec.read(s16, 0, s16.size, AudioRecord.READ_BLOCKING)
            if (n <= 0) continue
            for (i in 0 until n) f32[i] = s16[i] / 32768f
            val block = if (n < f32.size) f32.copyOf(n) else f32
            val delta = engine.processPcm(block)
            if (delta.isNotEmpty()) {
                transcript.append(delta)
                runOnUiThread { transcriptView.text = transcript }
            }
        }
        val tail = engine.stop()
        if (tail.isNotEmpty()) {
            transcript.append(tail)
            runOnUiThread { transcriptView.text = transcript }
        }
    } finally {
        rec.stop(); rec.release()
        engine.close()
    }
}
```

## Audio contract

- **Sample rate:** 16 000 Hz. **No automatic resampling.** If your
  source is 44.1 kHz / 48 kHz (typical), resample first (e.g.
  Android's `oboe::FlowGraph` or a simple `OboeResampler`). Feeding
  the wrong rate is the #1 source of "transcript is gibberish"
  bugs.
- **Sample format:** `FloatArray` PCM in `[-1, 1]`, mono, native
  endian. If you have `ShortArray` from `AudioRecord`, divide by
  `32768f`.
- **Buffer size:** any. The engine internally accumulates to its
  steady-state chunk size (17 920 samples ≈ 1.12 s) and emits
  text every chunk.
- **Latency:** one chunk (~1.12 s) of inherent buffering. Output
  text becomes available chunk-by-chunk in the listener's
  `onTextDelta`.

## Threading

- The engine is a **synchronous, stateful function**. It does NOT
  own a worker thread. Each `processPcm` call blocks on the calling
  thread for the duration of the inference work — for live mic,
  put the engine + capture loop on your own background thread (see
  the example above). Marshal text deltas back to UI via
  `runOnUiThread` / `Handler` / a `MutableStateFlow`.
- One engine instance is **single-thread-at-a-time**. Serialise
  `processPcm` / `stop` / `reset` / `close` against each other on
  a given instance. The engine is annotated `@Synchronized` for
  basic safety, but concurrent calls don't make transcription
  correct — only serial use does.
- One engine instance handles a stream of utterances. Between
  utterances, call `engine.reset()` to zero the K/V cache + LSTM
  state without paying weight-load cost again. Call `engine.close()`
  (or use `.use { }`) when done with the instance.

## Permissions

The library declares **no permissions** in its manifest. Your app
declares them as needed by your input pipeline:

- Live mic capture → `RECORD_AUDIO` (runtime-requested on Android 6+).
- Reading audio files from external storage → `READ_MEDIA_AUDIO`
  (Android 13+) or `READ_EXTERNAL_STORAGE` (lower).
- Network for downloading the `.vxrt` on first launch → `INTERNET`.

Add the line to your app's `AndroidManifest.xml`:

```xml
<uses-permission android:name="android.permission.RECORD_AUDIO" />
```

And request it at runtime before starting the engine.

## Decoder selection

**Recommended: RNN-T** — higher accuracy, modest extra cost. This
is the SDK default; you only need to pass an explicit decoder
constant if you specifically want CTC.

| Decoder | Constant | WER on LibriSpeech-500 | Per-chunk cost | When to use |
| ------- | -------- | ----------------------: | --------------: | ----------- |
| **RNN-T** ★ | `VoxrtAsrNative.DECODE_RNNT` | **3.267 %** | ~50 ms | **Recommended default.** Higher accuracy. LSTM state survives chunk boundaries. |
| CTC | `VoxrtAsrNative.DECODE_CTC` | 4.895 % | ~5 ms | Battery-constrained long sessions, or background transcription where the ~1.6 % WER hit is acceptable. Cross-chunk dedup handled internally. |

Both decoders run the same Conformer encoder; the head is
selected at session-create time. **TDT is not supported** on
streaming-medium-pc (no duration head) — passing
`VoxrtAsrNative.DECODE_TDT` fails the session creation.

## Architectures roadmap

`v0.1.1` ships only `arm64-v8a` for production. The `x86_64`
slice is included so the library works on Android emulators
(scalar code path, not NEON-optimized).

| ABI                       | Status     | Notes |
| ------------------------- | ---------- | ----- |
| arm64-v8a (NEON)          | ✅ Shipped | Full NEON-optimized inner loops. ~98 % of in-market Android devices. |
| x86_64                    | ✅ Shipped | Scalar fallback, emulator-only. No SSE/AVX kernels yet. |
| armeabi-v7a               | ⏸️ Deferred | All aarch64 NEON kernels would fall back to scalar (~4-8× slower); RTF projection ~2-4 on cheap ARMv7-only chips. Will be re-evaluated once we have a test device + customer demand. |
| x86                       | ☁️ On request | Tiny share, unlikely to ship. |

If you target a device whose ABI is not in this list, Gradle will
fail at install time with a `findLibrary` error. Filter
`splits.abi` accordingly.

## Project layout

```
voxrt-asr-android/
├── settings.gradle.kts                           # SPM equivalent
├── build.gradle.kts                              # plugin versions
├── voxrt-asr/                                    # the library module
│   ├── build.gradle.kts                          # publish config
│   ├── consumer-rules.pro                        # R8 keep rules for JNI symbols
│   └── src/main/
│       ├── AndroidManifest.xml
│       ├── java/com/voxrt/asr/                   # Kotlin wrapper (open, Apache-2.0)
│       │   ├── VoxrtAsrNative.kt
│       │   └── VoxrtAsrStreamingEngine.kt
│       └── jniLibs/
│           ├── arm64-v8a/libvoxrt_asr.so
│           └── x86_64/libvoxrt_asr.so
├── jitpack.yml                                   # JitPack build instructions
└── README.md                                     # this file
```

The compiled `libvoxrt_asr.so` per ABI is checked in as the
binary half of the distribution — JitPack does NOT rebuild Rust.

## License

- The Kotlin wrapper (`voxrt-asr/src/main/java/com/voxrt/asr/`) is
  licensed under **Apache-2.0**. See [`LICENSE`](LICENSE).
- The compiled `libvoxrt_asr.so` files are proprietary VoxRT
  runtime code owned by Elephant Enterprises LLC, redistributable
  as part of this unmodified Kotlin library. See
  [`LICENSE-BINARY`](LICENSE-BINARY) for the full terms.
- The streaming-medium-pc model weights are derived from
  `nvidia/stt_en_fastconformer_hybrid_medium_streaming_80ms_pc`,
  released under **CC-BY-4.0**. Attribution and upstream notice
  travel with the model artifact in the
  [models repository](https://github.com/VoxRT/voxrt-asr-models).
- Commercial integration / custom-model packaging questions:
  help@voxrt.com.

## Links

- VoxRT runtime + commercial models: [voxrt.com](https://voxrt.com)
- iOS counterpart: [voxrt-asr-ios](https://github.com/VoxRT/voxrt-asr-ios) (coming soon)
- ASR model weights & versions: [voxrt-asr-models](https://github.com/VoxRT/voxrt-asr-models)
- VAD companion: [voxrt-silero-android](https://github.com/VoxRT/voxrt-silero-android)
- Bugs / questions: open an issue on this repo
