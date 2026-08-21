#!/usr/bin/env bash

set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage: bash sign-release.sh <artifact.apk|artifact.aab> [options]

Options:
  --output PATH       Signed artifact path (defaults to *-signed.apk/aab)
  --keystore PATH     Signing keystore path
  --alias NAME        Key alias (default: volleycut-release)
  --force             Replace an existing output file
  -h, --help          Show this help

The signer prompts for the keystore password directly. Do not pass a password
on the command line or store one in this script.
EOF
}

die() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

to_posix_path() {
    local path=$1
    if [[ $path =~ ^[A-Za-z]:[\\/] ]]; then
        if command -v cygpath >/dev/null 2>&1; then
            cygpath -u -- "$path"
            return
        fi
        if command -v wslpath >/dev/null 2>&1; then
            wslpath -u -- "$path"
            return
        fi
    fi
    printf '%s\n' "$path"
}

absolute_existing_path() {
    local path=$1
    [[ -e $path ]] || die "Path does not exist: $path"
    local directory
    directory=$(cd -- "$(dirname -- "$path")" && pwd -P)
    printf '%s/%s\n' "$directory" "$(basename -- "$path")"
}

absolute_output_path() {
    local path=$1
    local directory
    directory=$(dirname -- "$path")
    [[ -d $directory ]] || die "Output directory does not exist: $directory"
    directory=$(cd -- "$directory" && pwd -P)
    printf '%s/%s\n' "$directory" "$(basename -- "$path")"
}

windows_user_directory() {
    local value=${USERPROFILE:-}
    if [[ -z $value && -n ${LOCALAPPDATA:-} ]]; then
        value=${LOCALAPPDATA%[\\/]AppData[\\/]Local}
    fi
    if [[ -z $value && $(uname -r 2>/dev/null || true) == *[Mm]icrosoft* ]] && \
        command -v cmd.exe >/dev/null 2>&1; then
        value=$(cmd.exe /d /c 'echo %USERPROFILE%' 2>/dev/null | tr -d '\r' | tail -n 1)
    fi
    if [[ -n $value ]]; then
        to_posix_path "$value"
    else
        printf '%s\n' "$HOME"
    fi
}

default_android_sdk() {
    local value=${ANDROID_HOME:-${ANDROID_SDK_ROOT:-}}
    if [[ -n $value ]]; then
        to_posix_path "$value"
        return
    fi

    value=${LOCALAPPDATA:-}
    if [[ -z $value && $(uname -r 2>/dev/null || true) == *[Mm]icrosoft* ]] && \
        command -v cmd.exe >/dev/null 2>&1; then
        value=$(cmd.exe /d /c 'echo %LOCALAPPDATA%' 2>/dev/null | tr -d '\r' | tail -n 1)
    fi
    if [[ -n $value ]]; then
        value=$(to_posix_path "$value")
        printf '%s/Android/Sdk\n' "${value%/}"
    elif [[ $(uname -s) == Darwin ]]; then
        printf '%s/Library/Android/sdk\n' "$HOME"
    else
        printf '%s/Android/Sdk\n' "$HOME"
    fi
}

find_java_tool() {
    local name=$1
    local candidate
    if [[ -n ${JAVA_HOME:-} ]]; then
        for candidate in "$JAVA_HOME/bin/$name" "$JAVA_HOME/bin/$name.exe"; do
            if [[ -x $candidate ]]; then
                printf '%s\n' "$candidate"
                return
            fi
        done
    fi
    if command -v "$name" >/dev/null 2>&1; then
        command -v "$name"
        return
    fi
    for candidate in \
        '/c/Program Files/Android/Android Studio/jbr/bin/'"$name"'.exe' \
        '/mnt/c/Program Files/Android/Android Studio/jbr/bin/'"$name"'.exe'; do
        if [[ -x $candidate ]]; then
            printf '%s\n' "$candidate"
            return
        fi
    done
    return 1
}

path_for_tool() {
    local tool=$1
    local path=$2
    if [[ $(uname -r 2>/dev/null || true) == *[Mm]icrosoft* && $tool == *.exe ]] && \
        command -v wslpath >/dev/null 2>&1; then
        wslpath -w -- "$path"
    else
        printf '%s\n' "$path"
    fi
}

artifact=''
output=''
keystore=''
key_alias='volleycut-release'
force=false

while (($#)); do
    case $1 in
        --output)
            (($# >= 2)) || die '--output requires a path'
            output=$2
            shift 2
            ;;
        --keystore)
            (($# >= 2)) || die '--keystore requires a path'
            keystore=$2
            shift 2
            ;;
        --alias)
            (($# >= 2)) || die '--alias requires a name'
            key_alias=$2
            shift 2
            ;;
        --force)
            force=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            (($# == 1)) || die "'--' must be followed by exactly one artifact path"
            [[ -z $artifact ]] || die 'Only one APK or AAB may be signed at a time'
            artifact=$1
            shift
            ;;
        -*)
            die "Unknown option: $1"
            ;;
        *)
            [[ -z $artifact ]] || die 'Only one APK or AAB may be signed at a time'
            artifact=$1
            shift
            ;;
    esac
done

[[ -n $artifact ]] || {
    usage >&2
    exit 2
}
(($# == 0)) || die 'Unexpected positional arguments'

artifact=$(to_posix_path "$artifact")
artifact=$(absolute_existing_path "$artifact")

if [[ -z $keystore ]]; then
    keystore="$(windows_user_directory)/.android-keystores/volleycut-release.jks"
fi
keystore=$(to_posix_path "$keystore")
keystore=$(absolute_existing_path "$keystore")

case ${artifact,,} in
    *.apk) extension=apk ;;
    *.aab) extension=aab ;;
    *) die 'The input file must be an APK or AAB' ;;
esac

if [[ -z $output ]]; then
    output=${artifact%.*}
    output=${output%-unsigned}
    output="$output-signed.$extension"
else
    output=$(to_posix_path "$output")
    output=$(absolute_output_path "$output")
fi

[[ ${output,,} == *.$extension ]] || die "Output must have the .$extension extension"
[[ $output != "$artifact" ]] || die 'Output path must differ from the unsigned artifact'
if [[ -e $output && $force != true ]]; then
    die "Output already exists (use --force to replace it): $output"
fi

temporary_output=$(mktemp "${output}.tmp.XXXXXX.$extension")
rm -f -- "$temporary_output"
cleanup() {
    rm -f -- "$temporary_output"
}
trap cleanup EXIT

if [[ $extension == apk ]]; then
    java=$(find_java_tool java) || die 'Java was not found; set JAVA_HOME or install Android Studio'
    android_sdk=$(default_android_sdk)
    [[ -d $android_sdk/build-tools ]] || die "Android SDK build-tools were not found under: $android_sdk"
    build_tools=$(
        for directory in "$android_sdk"/build-tools/*; do
            [[ -d $directory ]] && printf '%s\n' "$directory"
        done | sort -V | tail -n 1
    )
    [[ -n $build_tools ]] || die "No Android SDK build-tools installation was found under: $android_sdk"

    apksigner_jar=$build_tools/lib/apksigner.jar
    [[ -f $apksigner_jar ]] || die "apksigner.jar was not found under: $build_tools"
    zipalign=$build_tools/zipalign
    [[ -x $zipalign ]] || zipalign=$build_tools/zipalign.exe
    [[ -x $zipalign ]] || die "zipalign was not found under: $build_tools"

    zipalign_artifact=$(path_for_tool "$zipalign" "$artifact")
    "$zipalign" -c -P 16 4 "$zipalign_artifact"

    java_jar=$(path_for_tool "$java" "$apksigner_jar")
    java_keystore=$(path_for_tool "$java" "$keystore")
    java_output=$(path_for_tool "$java" "$temporary_output")
    java_artifact=$(path_for_tool "$java" "$artifact")
    "$java" -jar "$java_jar" sign \
        --ks "$java_keystore" \
        --ks-key-alias "$key_alias" \
        --min-sdk-version 34 \
        --v1-signing-enabled false \
        --v2-signing-enabled true \
        --v3-signing-enabled true \
        --v4-signing-enabled false \
        --out "$java_output" \
        "$java_artifact"
    "$java" -jar "$java_jar" verify \
        --min-sdk-version 34 \
        --verbose \
        --print-certs \
        "$java_output"
else
    jarsigner=$(find_java_tool jarsigner) || die 'jarsigner was not found; set JAVA_HOME to a JDK'
    signer_keystore=$(path_for_tool "$jarsigner" "$keystore")
    signer_output=$(path_for_tool "$jarsigner" "$temporary_output")
    signer_artifact=$(path_for_tool "$jarsigner" "$artifact")
    "$jarsigner" \
        -verbose \
        -sigalg SHA256withRSA \
        -digestalg SHA-256 \
        -keystore "$signer_keystore" \
        -signedjar "$signer_output" \
        "$signer_artifact" \
        "$key_alias"
    "$jarsigner" -verify -verbose -certs "$signer_output"
fi

if [[ -e $output ]]; then
    rm -f -- "$output"
fi
mv -- "$temporary_output" "$output"
trap - EXIT
printf 'Signed %s: %s\n' "${extension^^}" "$output"
