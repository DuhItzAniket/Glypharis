import java.security.MessageDigest
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

val ortVersion = providers.gradleProperty("glypharis.ortVersion").get()
val ortAarSha256 = providers.gradleProperty("glypharis.ortAarSha256").get()
val jdkMajor = providers.gradleProperty("glypharis.jdkMajor").get()

android {
    namespace = "org.glypharis.bench"
    compileSdk = 35

    defaultConfig {
        applicationId = "org.glypharis.bench"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"
        ndk {
            abiFilters += "arm64-v8a"
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

kotlin {
    compilerOptions {
        jvmTarget.set(JvmTarget.JVM_17)
    }
}

val ortPackage: Configuration by configurations.creating {
    isCanBeConsumed = false
    isTransitive = false
}

dependencies {
    implementation("com.microsoft.onnxruntime:onnxruntime-android:$ortVersion")
    ortPackage("com.microsoft.onnxruntime:onnxruntime-android:$ortVersion@aar")
}

// The registered build: the JDK major version and the exact onnxruntime-android package.
val checkRegisteredBuild by tasks.registering {
    val aar: FileCollection = ortPackage
    inputs.files(aar)
    doLast {
        val jdk = JavaVersion.current().majorVersion
        check(jdk == jdkMajor) { "this build is registered for JDK $jdkMajor, running on $jdk" }
        val digest = MessageDigest.getInstance("SHA-256").digest(aar.singleFile.readBytes())
        val got = digest.joinToString("") { "%02x".format(it) }
        check(got == ortAarSha256) {
            "onnxruntime-android $ortVersion is $got, registered as $ortAarSha256"
        }
    }
}

tasks.named("preBuild") {
    dependsOn(checkRegisteredBuild)
}
