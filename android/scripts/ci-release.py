"""Build metadata, signing, and Google Play publishing helpers for GitHub Actions.

Secrets are accepted only through environment variables. The helper never prints
credential values or includes them directly in subprocess command arguments.
"""

from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any
from urllib.parse import quote


DEFAULT_PACKAGE_NAME = "com.volleycut.nativeanalysis"
ANDROID_PUBLISHER_SCOPE = "https://www.googleapis.com/auth/androidpublisher"


class PublisherError(RuntimeError):
    pass


def required_env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise ValueError(f"Configure {name} in the protected GitHub environment")
    return value


def run(*args: object, **kwargs: Any) -> subprocess.CompletedProcess:
    # Command arguments contain only secret *environment variable names*, never
    # their values. Avoid echoing commands here so that remains true.
    return subprocess.run([str(arg) for arg in args], check=True, **kwargs)


def read_gradle_metadata(path: Path) -> dict[str, Any]:
    source = path.read_text(encoding="utf-8")

    def one(pattern: str, label: str) -> str:
        matches = re.findall(pattern, source, flags=re.MULTILINE)
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one literal {label} in {path}")
        return matches[0]

    package_name = one(r'^\s*applicationId\s*=\s*"([^"]+)"\s*$', "applicationId")
    version_code_text = one(r"^\s*versionCode\s*=\s*([0-9]+)\s*$", "versionCode")
    version_name = one(r'^\s*versionName\s*=\s*"([^"]+)"\s*$', "versionName")
    version_code = int(version_code_text)
    if package_name != DEFAULT_PACKAGE_NAME:
        raise ValueError(f"Release applicationId must remain {DEFAULT_PACKAGE_NAME}")
    if version_code < 1:
        raise ValueError("versionCode must be positive")
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version_name):
        raise ValueError("versionName must have three numeric components, for example 1.2.3")
    return {
        "package_name": package_name,
        "version_code": version_code,
        "version_name": version_name,
    }


def write_github_outputs(path: Path, values: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as output:
        for key, value in values.items():
            text = str(value)
            if "\n" in text or "\r" in text:
                raise ValueError(f"GitHub output {key} may not contain a newline")
            output.write(f"{key}={text}\n")


def normalize_sha256(value: str) -> str:
    normalized = re.sub(r"[^0-9A-Fa-f]", "", value).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise ValueError("Upload certificate SHA-256 must contain exactly 64 hexadecimal digits")
    return normalized


def sign_bundle(input_path: Path, output_path: Path, keystore: Path, alias: str,
                expected_sha256: str) -> dict[str, Any]:
    if input_path.suffix.lower() != ".aab" or not input_path.is_file():
        raise ValueError(f"Unsigned Android App Bundle was not found: {input_path}")
    if not keystore.is_file():
        raise ValueError(f"Play upload keystore was not found: {keystore}")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", alias):
        raise ValueError("ANDROID_UPLOAD_KEY_ALIAS contains unsupported characters")
    required_env("ANDROID_UPLOAD_STORE_PASSWORD")
    required_env("ANDROID_UPLOAD_KEY_PASSWORD")

    certificate = run(
        "keytool", "-list", "-v", "-keystore", keystore,
        "-storepass:env", "ANDROID_UPLOAD_STORE_PASSWORD", "-alias", alias,
        capture_output=True, text=True,
    )
    match = re.search(r"SHA256:\s*([0-9A-Fa-f:]+)", certificate.stdout + certificate.stderr)
    if not match:
        raise ValueError("Could not read the upload certificate SHA-256 from the keystore")
    actual_sha256 = normalize_sha256(match.group(1))
    if actual_sha256 != normalize_sha256(expected_sha256):
        raise ValueError(
            "Play upload certificate SHA-256 does not match ANDROID_UPLOAD_CERT_SHA256"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    run(
        "jarsigner", "-keystore", keystore,
        "-storepass:env", "ANDROID_UPLOAD_STORE_PASSWORD",
        "-keypass:env", "ANDROID_UPLOAD_KEY_PASSWORD",
        "-sigalg", "SHA256withRSA", "-digestalg", "SHA-256",
        "-signedjar", output_path, input_path, alias,
    )
    run("jarsigner", "-verify", "-verbose", "-certs", output_path)
    return {
        "path": str(output_path),
        "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "certificate_sha256": actual_sha256,
    }


class PublisherClient:
    def __init__(self, access_token: str, timeout: int = 180):
        if not access_token:
            raise ValueError("ANDROID_PUBLISHER_ACCESS_TOKEN is empty")
        self.access_token = access_token
        self.timeout = timeout

    def request(self, method: str, host: str, path: str, body: object | None = None,
                file_path: Path | None = None) -> dict[str, Any]:
        connection = http.client.HTTPSConnection(host, timeout=self.timeout)
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Accept": "application/json",
        }
        try:
            if file_path is not None:
                size = file_path.stat().st_size
                connection.putrequest(method, path)
                for name, value in headers.items():
                    connection.putheader(name, value)
                connection.putheader("Content-Type", "application/octet-stream")
                connection.putheader("Content-Length", str(size))
                connection.endheaders()
                with file_path.open("rb") as bundle:
                    while chunk := bundle.read(1024 * 1024):
                        connection.send(chunk)
            else:
                encoded = None
                if body is not None:
                    encoded = json.dumps(body, separators=(",", ":")).encode("utf-8")
                    headers["Content-Type"] = "application/json; charset=utf-8"
                connection.request(method, path, body=encoded, headers=headers)
            response = connection.getresponse()
            payload = response.read()
        finally:
            connection.close()
        decoded: dict[str, Any] = {}
        if payload:
            try:
                decoded = json.loads(payload)
            except json.JSONDecodeError:
                decoded = {"raw": payload.decode("utf-8", errors="replace")[:1000]}
        if response.status < 200 or response.status >= 300:
            message = decoded.get("error", decoded)
            raise PublisherError(f"Google Play API returned HTTP {response.status}: {message}")
        return decoded

    @staticmethod
    def edit_path(package_name: str, edit_id: str | None = None) -> str:
        path = f"/androidpublisher/v3/applications/{quote(package_name, safe='')}/edits"
        if edit_id is not None:
            path += "/" + quote(edit_id, safe="")
        return path

    def create_edit(self, package_name: str) -> str:
        response = self.request("POST", "androidpublisher.googleapis.com",
                                self.edit_path(package_name), body={})
        edit_id = response.get("id")
        if not isinstance(edit_id, str) or not edit_id:
            raise PublisherError("Google Play did not return an edit ID")
        return edit_id

    def delete_edit(self, package_name: str, edit_id: str) -> None:
        self.request("DELETE", "androidpublisher.googleapis.com",
                     self.edit_path(package_name, edit_id))

    def upload_bundle(self, package_name: str, edit_id: str, bundle: Path) -> int:
        path = self.edit_path(package_name, edit_id) + "/bundles?uploadType=media"
        response = self.request("POST", "androidpublisher.googleapis.com", path,
                                file_path=bundle)
        try:
            return int(response["versionCode"])
        except (KeyError, TypeError, ValueError) as error:
            raise PublisherError("Google Play did not return the uploaded versionCode") from error

    def upload_deobfuscation_file(self, package_name: str, edit_id: str,
                                  version_code: int, file_type: str,
                                  file_path: Path) -> dict[str, Any]:
        if file_type not in {"proguard", "nativeCode"}:
            raise ValueError("Unsupported Google Play deobfuscation file type")
        path = (
            self.edit_path(package_name, edit_id)
            + f"/apks/{version_code}/deobfuscationFiles/{file_type}?uploadType=media"
        )
        return self.request("POST", "androidpublisher.googleapis.com", path,
                            file_path=file_path)

    def update_track(self, package_name: str, edit_id: str, track: str,
                     release: dict[str, Any]) -> dict[str, Any]:
        path = self.edit_path(package_name, edit_id) + "/tracks/" + quote(track, safe="")
        return self.request("PUT", "androidpublisher.googleapis.com", path,
                            body={"track": track, "releases": [release]})

    def commit_edit(self, package_name: str, edit_id: str) -> dict[str, Any]:
        return self.request("POST", "androidpublisher.googleapis.com",
                            self.edit_path(package_name, edit_id) + ":commit", body={})


def validate_package_name(package_name: str) -> None:
    if package_name != DEFAULT_PACKAGE_NAME:
        raise ValueError(f"Publishing is restricted to {DEFAULT_PACKAGE_NAME}")


def release_payload(version_code: int, name: str, status: str,
                    release_notes: str = "", user_fraction: str | None = None) -> dict[str, Any]:
    if version_code < 1:
        raise ValueError("versionCode must be positive")
    if status not in {"completed", "inProgress"}:
        raise ValueError("Release status must be completed or inProgress")
    release: dict[str, Any] = {
        "name": name,
        "status": status,
        "versionCodes": [str(version_code)],
    }
    if status == "inProgress":
        try:
            fraction = Decimal(user_fraction or "")
        except InvalidOperation as error:
            raise ValueError("Production rollout fraction must be a decimal") from error
        if fraction <= 0 or fraction >= 1:
            raise ValueError("A staged production rollout fraction must be greater than 0 and less than 1")
        release["userFraction"] = float(fraction)
    elif user_fraction is not None:
        raise ValueError("userFraction is valid only for an inProgress release")
    if release_notes:
        if len(release_notes) > 500:
            raise ValueError("English (United States) release notes may not exceed 500 characters")
        release["releaseNotes"] = [{"language": "en-US", "text": release_notes}]
    return release


def run_edit(client: PublisherClient, package_name: str, operation: Any) -> dict[str, Any]:
    edit_id = client.create_edit(package_name)
    committed = False
    try:
        result = operation(edit_id)
        response = client.commit_edit(package_name, edit_id)
        committed = True
        return {"edit": response, **result}
    finally:
        if not committed:
            try:
                client.delete_edit(package_name, edit_id)
            except Exception:
                # The original exception is more useful, and edits expire on their own.
                pass


def publish_bundle(client: PublisherClient, package_name: str, bundle: Path, track: str,
                   expected_version_code: int, release_name: str,
                   release_notes: str = "", mapping_file: Path | None = None,
                   native_symbols: Path | None = None) -> dict[str, Any]:
    validate_package_name(package_name)
    if track != "internal":
        raise ValueError("New bundles may only be uploaded to the internal track")
    if not bundle.is_file():
        raise ValueError(f"Signed Android App Bundle was not found: {bundle}")
    if mapping_file is not None and not mapping_file.is_file():
        raise ValueError(f"R8 mapping file was not found: {mapping_file}")
    if native_symbols is not None and not native_symbols.is_file():
        raise ValueError(f"Native debug symbols were not found: {native_symbols}")

    def operation(edit_id: str) -> dict[str, Any]:
        uploaded_version_code = client.upload_bundle(package_name, edit_id, bundle)
        if uploaded_version_code != expected_version_code:
            raise PublisherError(
                f"Uploaded versionCode {uploaded_version_code} did not match expected "
                f"{expected_version_code}"
            )
        if mapping_file is not None:
            client.upload_deobfuscation_file(
                package_name, edit_id, uploaded_version_code, "proguard", mapping_file
            )
        if native_symbols is not None:
            client.upload_deobfuscation_file(
                package_name, edit_id, uploaded_version_code, "nativeCode", native_symbols
            )
        release = release_payload(uploaded_version_code, release_name, "completed", release_notes)
        client.update_track(package_name, edit_id, track, release)
        return {"version_code": uploaded_version_code, "track": track, "status": "completed"}

    return run_edit(client, package_name, operation)


def promote_version(client: PublisherClient, package_name: str, version_code: int,
                    release_name: str, user_fraction: str,
                    release_notes: str = "") -> dict[str, Any]:
    validate_package_name(package_name)

    def operation(edit_id: str) -> dict[str, Any]:
        release = release_payload(
            version_code, release_name, "inProgress", release_notes, user_fraction
        )
        client.update_track(package_name, edit_id, "production", release)
        return {
            "version_code": version_code,
            "track": "production",
            "status": "inProgress",
            "user_fraction": float(Decimal(user_fraction)),
        }

    return run_edit(client, package_name, operation)


def metadata_command(args: argparse.Namespace) -> None:
    metadata = read_gradle_metadata(args.gradle_file)
    if args.github_output:
        write_github_outputs(args.github_output, metadata)
    print(json.dumps(metadata, indent=2))


def sign_command(args: argparse.Namespace) -> None:
    result = sign_bundle(
        args.input, args.output, args.keystore,
        required_env("ANDROID_UPLOAD_KEY_ALIAS"),
        required_env("ANDROID_UPLOAD_CERT_SHA256"),
    )
    metadata = read_gradle_metadata(args.gradle_file)
    release = {
        **metadata,
        "commit": os.environ.get("GITHUB_SHA"),
        "bundle_sha256": result["sha256"],
        "upload_certificate_sha256": result["certificate_sha256"],
    }
    (args.output.parent / "release.json").write_text(
        json.dumps(release, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Signed and verified {args.output.name}")


def client_from_environment() -> PublisherClient:
    return PublisherClient(required_env("ANDROID_PUBLISHER_ACCESS_TOKEN"))


def publish_command(args: argparse.Namespace) -> None:
    result = publish_bundle(
        client_from_environment(), args.package_name, args.bundle, "internal",
        args.version_code, args.release_name, args.release_notes,
        args.mapping, args.native_symbols,
    )
    print(json.dumps(result, indent=2))


def promote_command(args: argparse.Namespace) -> None:
    result = promote_version(
        client_from_environment(), args.package_name, args.version_code,
        args.release_name, args.user_fraction, args.release_notes,
    )
    print(json.dumps(result, indent=2))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)

    metadata = commands.add_parser("metadata")
    metadata.add_argument("--gradle-file", type=Path, required=True)
    metadata.add_argument("--github-output", type=Path)
    metadata.set_defaults(handler=metadata_command)

    sign = commands.add_parser("sign")
    sign.add_argument("--input", type=Path, required=True)
    sign.add_argument("--output", type=Path, required=True)
    sign.add_argument("--keystore", type=Path, required=True)
    sign.add_argument("--gradle-file", type=Path, required=True)
    sign.set_defaults(handler=sign_command)

    publish = commands.add_parser("publish-internal")
    publish.add_argument("--bundle", type=Path, required=True)
    publish.add_argument("--package-name", default=DEFAULT_PACKAGE_NAME)
    publish.add_argument("--version-code", type=int, required=True)
    publish.add_argument("--release-name", required=True)
    publish.add_argument("--release-notes", default="")
    publish.add_argument("--mapping", type=Path)
    publish.add_argument("--native-symbols", type=Path)
    publish.set_defaults(handler=publish_command)

    promote = commands.add_parser("promote-production")
    promote.add_argument("--package-name", default=DEFAULT_PACKAGE_NAME)
    promote.add_argument("--version-code", type=int, required=True)
    promote.add_argument("--release-name", required=True)
    promote.add_argument("--release-notes", default="")
    promote.add_argument("--user-fraction", required=True)
    promote.set_defaults(handler=promote_command)
    return root


if __name__ == "__main__":
    os.umask(0o077)
    try:
        arguments = parser().parse_args()
        arguments.handler(arguments)
    except subprocess.CalledProcessError as error:
        print(f"Release command failed (exit {error.returncode}); see preceding output.", file=sys.stderr)
        sys.exit(1)
    except (OSError, PublisherError, ValueError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
