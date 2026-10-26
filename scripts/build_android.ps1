param(
    [string]$SdkPath = $env:ANDROID_HOME,
    [string]$JdkPath = $env:JAVA_HOME,
    [string]$GradlePath = $env:GRADLE_HOME,
    [string[]]$Tasks = @('testDebugUnitTest','lintDebug','assembleDebug'),
    [switch]$InitializeVerification,
    [switch]$Offline,
    [string]$SessionFixture,
    [string]$WavFixture
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path $PSScriptRoot -Parent
$oldTools = Join-Path $env:USERPROFILE '.zcode/workspace/default/_toolchain'
if (-not $SdkPath) { $SdkPath = Join-Path $oldTools 'android-sdk' }
if (-not $JdkPath) { $JdkPath = Join-Path $oldTools 'jdk/jdk-17.0.20.1+1' }
if (-not $GradlePath) { $GradlePath = Join-Path $oldTools 'gradle/gradle-8.5' }
$SdkPath = [IO.Path]::GetFullPath($SdkPath)
$JdkPath = [IO.Path]::GetFullPath($JdkPath)
$GradlePath = [IO.Path]::GetFullPath($GradlePath)
$gradleCommand = Join-Path $GradlePath 'bin/gradle.bat'
foreach ($required in @((Join-Path $SdkPath 'platforms/android-34/android.jar'),
    (Join-Path $SdkPath 'build-tools/34.0.0/aapt2.exe'), (Join-Path $JdkPath 'bin/java.exe'), $gradleCommand)) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Required local Android tool missing: $required. No SDK is downloaded automatically." }
}
$androidRoot = Join-Path $repoRoot 'android'
$verification = Join-Path $androidRoot 'gradle/verification-metadata.xml'
if (-not $InitializeVerification -and -not (Test-Path -LiteralPath $verification)) {
    throw 'Dependency verification metadata missing; initial audited bootstrap requires -InitializeVerification.'
}
$signingRoot = Join-Path $repoRoot 'private/android-signing'
$signingKey = Join-Path $signingRoot 'debug.keystore'
if (($Tasks -contains 'assembleDebug') -and -not (Test-Path -LiteralPath $signingKey)) {
    New-Item -ItemType Directory -Force -Path $signingRoot | Out-Null
    & (Join-Path $JdkPath 'bin/keytool.exe') -genkeypair -keystore $signingKey -storepass android -keypass android -alias androiddebugkey -keyalg RSA -keysize 2048 -validity 10000 -dname 'CN=Android Debug,O=Android,C=US' -noprompt
    if ($LASTEXITCODE -ne 0) { throw 'Project-local debug signing key creation failed.' }
}
$privateRoot = Join-Path $repoRoot 'private/android-build'
New-Item -ItemType Directory -Force -Path $privateRoot | Out-Null
$names = @('JAVA_HOME','ANDROID_HOME','ANDROID_SDK_ROOT','ANDROID_USER_HOME','GRADLE_USER_HOME','PATH')
$previous = @{}
foreach ($name in $names) { $previous[$name] = [Environment]::GetEnvironmentVariable($name,'Process') }
Push-Location $androidRoot
try {
    $env:JAVA_HOME = $JdkPath
    $env:ANDROID_HOME = $SdkPath
    $env:ANDROID_SDK_ROOT = $SdkPath
    $env:ANDROID_USER_HOME = Join-Path $privateRoot 'android-user'
    $env:GRADLE_USER_HOME = Join-Path $privateRoot 'gradle-user'
    $env:PATH = (Join-Path $JdkPath 'bin') + [IO.Path]::PathSeparator + $previous['PATH']
    New-Item -ItemType Directory -Force -Path $env:ANDROID_USER_HOME | Out-Null
    Set-Content -LiteralPath (Join-Path $androidRoot 'local.properties') -Value ('sdk.dir=' + $SdkPath.Replace('\','/')) -Encoding utf8
    $arguments = @('--no-daemon','--max-workers=2','--console=plain')
    if ($InitializeVerification) { $arguments += @('--write-verification-metadata','sha256','--write-locks') }
    else { $arguments += @('--dependency-verification','strict') }
    if ($WavFixture) { $arguments += ('-Dnightlab.wav.fixture=' + [IO.Path]::GetFullPath($WavFixture)) }
    if ($SessionFixture) { $arguments += ('-Dnightlab.session.fixture=' + [IO.Path]::GetFullPath($SessionFixture)) }
    if ($Offline) { $arguments += '--offline' }
    $arguments += $Tasks
    & $gradleCommand @arguments
    if ($LASTEXITCODE -ne 0) { throw "Android Gradle build failed (exit $LASTEXITCODE)." }
    if ($Tasks -contains 'assembleDebug') {
        $apk = Join-Path $androidRoot 'app/build/outputs/apk/debug/app-debug.apk'
        & (Join-Path $SdkPath 'build-tools/34.0.0/apksigner.bat') verify $apk
        if ($LASTEXITCODE -ne 0) { throw 'APK signature verification failed.' }
        $delivery = Join-Path $repoRoot 'private/android-delivery'
        New-Item -ItemType Directory -Force -Path $delivery | Out-Null
        $target = Join-Path $delivery 'NightLabRecorder-0.1.0-debug.apk'
        Copy-Item -LiteralPath $apk -Destination $target -Force
        $sourceHash = (Get-FileHash -LiteralPath $apk -Algorithm SHA256).Hash.ToLowerInvariant()
        if ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant() -ne $sourceHash) { throw 'APK copy hash mismatch.' }
        [ordered]@{ version='0.1.0'; type='debug'; sha256=$sourceHash; bytes=(Get-Item -LiteralPath $target).Length; signature_verified=$true; phone_recording_verified=$false } |
            ConvertTo-Json | Set-Content -LiteralPath (Join-Path $delivery 'build-receipt.json') -Encoding utf8
        Write-Output "APK: $target"
        Write-Output "SHA256: $sourceHash"
    }
}
finally {
    Pop-Location
    foreach ($name in $names) { [Environment]::SetEnvironmentVariable($name,$previous[$name],'Process') }
}
