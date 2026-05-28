# ProGuard / R8 rules consumed by apps that depend on voxrt-asr.
#
# Two failure modes we're defending against in any consumer build
# with `isMinifyEnabled = true`:
#
#   1. R8 renames `com.voxrt.asr.VoxrtAsrNative` to something like
#      `com.voxrt.asr.a`. JNI lookups from `libvoxrt_asr.so` use
#      the fully-qualified symbol name (`Java_com_voxrt_asr_VoxrtAsrNative_streamingCreate`)
#      and `UnsatisfiedLinkError`s at the first native call.
#
#   2. R8 strips or renames an `external fun` because the Kotlin
#      bytecode marks it as instance-method (the singleton receiver
#      is the implicit `this`), and R8 can't see through native-side
#      callers. Same outcome — silent failure at runtime.
#
# `VoxrtAsrNative` is a Kotlin `object`, so in JVM bytecode it is:
#
#   public final class com.voxrt.asr.VoxrtAsrNative {
#       public static final com.voxrt.asr.VoxrtAsrNative INSTANCE;
#       public final native long streamingCreate(byte[], int);
#       ...etc
#   }
#
# We must keep the class name AND its members AND the INSTANCE field.

-keep class com.voxrt.asr.VoxrtAsrNative {
    public static ** INSTANCE;
    public static <fields>;
    public <methods>;
    native <methods>;
}

# Defence in depth: any class that has native methods should keep
# them under their original names so JNI resolution still works.
# Default Android proguard rules already include this, but pinning
# it here makes the contract explicit and survives a consumer who
# overrides the default rule set.
-keepclasseswithmembernames class * {
    native <methods>;
}

# `VoxrtAsrStreamingEngine` itself is the entry point the consumer
# touches. Keep the class name + public methods (pushAudio / stop /
# reset / close / transcribeAll / stageTimings / ...) so the
# consumer's call sites still resolve after R8. The post-2026-05-28
# refactor moved away from a Listener-callback shape — the engine is
# now a synchronous stateful function with no inner interfaces to
# keep.
-keep class com.voxrt.asr.VoxrtAsrStreamingEngine {
    public *;
}
