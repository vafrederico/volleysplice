"""GitHub-hosted macOS release helper; credentials are read only from environment."""
import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import plistlib
import re
import secrets
import shutil
import subprocess
import sys

BUNDLE_ID = "com.volleysplice.VolleySplice"


def required(name):
    value = os.environ.get(name, "")
    if not value:
        raise ValueError(f"Configure {name} in the ios-release GitHub environment")
    return value


def validate():
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", required("RELEASE_VERSION")):
        raise ValueError("Release version must be three numeric components, e.g. 1.0.0")
    if not re.fullmatch(r"[1-9][0-9]{0,3}", required("RELEASE_BUILD_NUMBER")):
        raise ValueError("Build number must be an integer from 1 to 9999")


def run(*args, **kwargs):
    # Do not print command arguments: security import receives the P12 password.
    return subprocess.run([str(arg) for arg in args], check=True, **kwargs)


def decode_secret(name):
    # Clipboard/PowerShell input can add line breaks to a base64 secret.
    return base64.b64decode("".join(required(name).split()), validate=True)


def validate_profile(profile, team):
    entitlements = profile.get("Entitlements", {})
    if team not in profile.get("TeamIdentifier", []):
        raise ValueError("Provisioning profile does not belong to IOS_TEAM_ID")
    prefixes = profile.get("ApplicationIdentifierPrefix", [])
    if entitlements.get("application-identifier") not in [p + "." + BUNDLE_ID for p in prefixes]:
        raise ValueError("Provisioning profile must explicitly match " + BUNDLE_ID)
    if (entitlements.get("get-task-allow", False) or profile.get("ProvisionedDevices")
            or profile.get("ProvisionsAllDevices", False)):
        raise ValueError("Use an App Store Connect distribution profile, not development/ad hoc/enterprise")
    if profile["ExpirationDate"].replace(tzinfo=datetime.timezone.utc) <= datetime.datetime.now(datetime.timezone.utc):
        raise ValueError("Provisioning profile has expired")
    if not re.fullmatch(r"[A-Fa-f0-9-]{36}", profile["UUID"]):
        raise ValueError("Invalid provisioning profile UUID")


def build(temp):
    validate()
    team = required("IOS_TEAM_ID")
    if not re.fullmatch(r"[A-Z0-9]{10}", team):
        raise ValueError("IOS_TEAM_ID must be the 10-character Apple Developer Team ID")
    signing = temp / "ios-signing"
    signing.mkdir(mode=0o700, exist_ok=True)
    p12 = signing / "distribution.p12"
    profile_file = signing / "app.mobileprovision"
    p12.write_bytes(decode_secret("IOS_DISTRIBUTION_P12_BASE64"))
    profile_file.write_bytes(decode_secret("IOS_APP_STORE_PROFILE_BASE64"))
    profile = plistlib.loads(run("security", "cms", "-D", "-i", profile_file, capture_output=True).stdout)
    validate_profile(profile, team)
    keychain = temp / "ios-release.keychain-db"
    password = secrets.token_urlsafe(32)
    run("security", "create-keychain", "-p", password, keychain)
    run("security", "set-keychain-settings", "-lut", "21600", keychain)
    run("security", "unlock-keychain", "-p", password, keychain)
    run("security", "import", p12, "-P", required("IOS_DISTRIBUTION_P12_PASSWORD"),
        "-A", "-t", "cert", "-f", "pkcs12", "-k", keychain)
    run("security", "set-key-partition-list", "-S", "apple-tool:,apple:,codesign:",
        "-k", password, keychain, stdout=subprocess.DEVNULL)
    run("security", "list-keychains", "-d", "user", "-s", keychain)
    identities = run("security", "find-identity", "-v", "-p", "codesigning", keychain, capture_output=True).stdout.decode()
    fingerprints = [hashlib.sha1(cert).hexdigest().upper() for cert in profile["DeveloperCertificates"]]
    matching = [fp for fp in fingerprints if fp in identities]
    if len(matching) != 1:
        raise ValueError("P12 must contain the valid signing identity selected in the profile")
    identity = matching[0]
    # Xcode 16+ uses this location for manually installed profiles.
    profiles = Path.home() / "Library/Developer/Xcode/UserData/Provisioning Profiles"
    profiles.mkdir(parents=True, exist_ok=True)
    installed = profiles / (profile["UUID"] + ".mobileprovision")
    (signing / "installed-profile.txt").write_text(str(installed))
    shutil.copyfile(profile_file, installed)
    source = temp / "ios-source"
    archive = temp / "VolleySplice.xcarchive"
    run("xcodebuild", "-project", source / "VolleySplice.xcodeproj", "-scheme", "VolleySplice",
        "-configuration", "Release", "-destination", "generic/platform=iOS",
        "-archivePath", archive, "-derivedDataPath", temp / "ios-derived-data",
        "archive", "CODE_SIGN_STYLE=Manual", f"CODE_SIGN_IDENTITY={identity}",
        f"DEVELOPMENT_TEAM={team}", f"PROVISIONING_PROFILE_SPECIFIER={profile['UUID']}",
        f"MARKETING_VERSION={required('RELEASE_VERSION')}",
        f"CURRENT_PROJECT_VERSION={required('RELEASE_BUILD_NUMBER')}",
        f"VOLLEYCUT_IOS_OPENCV_FRAMEWORK_ROOT={temp / 'opencv'}", "DEBUG_INFORMATION_FORMAT=dwarf-with-dsym")
    run("codesign", "--verify", "--deep", "--strict", archive / "Products/Applications/VolleySplice.app")
    options = signing / "ExportOptions.plist"
    options.write_bytes(plistlib.dumps({
        "method": "app-store-connect", "destination": "export", "teamID": team,
        "signingStyle": "manual", "signingCertificate": identity,
        "provisioningProfiles": {BUNDLE_ID: profile["UUID"]},
        "manageAppVersionAndBuildNumber": False, "uploadSymbols": True,
    }))
    exported = temp / "ios-export"
    run("xcodebuild", "-exportArchive", "-archivePath", archive,
        "-exportPath", exported, "-exportOptionsPlist", options)
    ipas = list(exported.glob("*.ipa"))
    if len(ipas) != 1:
        raise ValueError("Expected exactly one exported IPA")
    output = temp / "ios-release-output"
    output.mkdir(exist_ok=True)
    shutil.copyfile(ipas[0], output / "VolleySplice.ipa")
    run("ditto", "-c", "-k", "--keepParent", archive / "dSYMs", output / "dSYMs.zip")
    metadata = {"version": required("RELEASE_VERSION"), "build": required("RELEASE_BUILD_NUMBER"),
                "bundle_id": BUNDLE_ID, "commit": os.environ.get("GITHUB_SHA"),
                "sha256": hashlib.sha256(ipas[0].read_bytes()).hexdigest(),
                "xcode": run("xcodebuild", "-version", capture_output=True).stdout.decode().strip()}
    (output / "release.json").write_text(json.dumps(metadata, indent=2) + "\n")


def upload(temp):
    key_id, issuer = required("ASC_KEY_ID"), required("ASC_ISSUER_ID")
    if not re.fullmatch(r"[A-Z0-9]{10}", key_id):
        raise ValueError("Invalid App Store Connect key ID")
    keys = temp / "ios-signing" / "private_keys"
    keys.mkdir(parents=True, mode=0o700, exist_ok=True)
    (keys / f"AuthKey_{key_id}.p8").write_text(required("ASC_PRIVATE_KEY_P8"))
    # altool searches ./private_keys for the key. Upload does not submit App Review.
    run("xcrun", "altool", "--upload-app", "-f", temp / "ios-release-output/VolleySplice.ipa",
        "--type", "ios", "--apiKey", key_id, "--apiIssuer", issuer, cwd=keys.parent)


def cleanup(temp):
    signing = temp / "ios-signing"
    marker = signing / "installed-profile.txt"
    if marker.exists():
        installed = Path(marker.read_text())
        expected = Path.home() / "Library/Developer/Xcode/UserData/Provisioning Profiles"
        if installed.parent == expected and installed.suffix == ".mobileprovision":
            installed.unlink(missing_ok=True)
    keychain = temp / "ios-release.keychain-db"
    if keychain.exists():
        run("security", "delete-keychain", keychain)
    shutil.rmtree(signing, ignore_errors=True)


if __name__ == "__main__":
    os.umask(0o077)
    try:
        command = sys.argv[1]
        if command == "validate":
            validate()
        else:
            {"build": build, "upload": upload, "cleanup": cleanup}[command](Path(required("RUNNER_TEMP")))
    except subprocess.CalledProcessError as error:
        # CalledProcessError's default text includes secret command arguments.
        print(f"Release command failed (exit {error.returncode}); see preceding tool output.", file=sys.stderr)
        sys.exit(1)
    except (ValueError, KeyError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
