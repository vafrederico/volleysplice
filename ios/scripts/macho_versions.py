"""Read deployment versions from packaged Mach-O executables without Apple tools."""
import re
import struct


def version(value):
    if not isinstance(value, str) or not re.fullmatch(r'[1-9][0-9]*(?:\.[0-9]+){0,2}', value):
        raise ValueError('Missing or invalid MinimumOSVersion')
    parts = tuple(map(int, value.split('.')))
    return parts + (0,) * (3 - len(parts))


def version_text(value):
    return '.'.join(map(str, value if value[2] else value[:2]))


def deployment_versions(data):
    """Return every thin/fat slice's platform and minimum; malformed input fails closed."""
    def thin(blob):
        formats = {b'\xcf\xfa\xed\xfe': ('<', 32), b'\xfe\xed\xfa\xcf': ('>', 32),
                   b'\xce\xfa\xed\xfe': ('<', 28), b'\xfe\xed\xfa\xce': ('>', 28)}
        if blob[:4] not in formats:
            raise ValueError('Expected an embedded Mach-O executable')
        endian, header = formats[blob[:4]]
        if len(blob) < header:
            raise ValueError('Truncated Mach-O header')
        cpu, subtype, kind, count, length = struct.unpack_from(endian + 'IIIII', blob, 4)
        if kind not in (2, 6, 8) or count > 65536 or header + length > len(blob):
            raise ValueError('Invalid Mach-O executable/load commands')
        cursor, found = header, []
        for _ in range(count):
            if cursor + 8 > header + length:
                raise ValueError('Truncated Mach-O load command')
            command, size = struct.unpack_from(endian + 'II', blob, cursor)
            if size < 8 or cursor + size > header + length:
                raise ValueError('Invalid Mach-O load command size')
            if command == 0x32:  # LC_BUILD_VERSION
                if size < 24:
                    raise ValueError('Truncated LC_BUILD_VERSION')
                platform, minimum, _, tools = struct.unpack_from(endian + 'IIII', blob, cursor + 8)
                if 24 + tools * 8 != size:
                    raise ValueError('Invalid LC_BUILD_VERSION tools')
                found.append((platform, minimum))
            elif command == 0x25:  # LC_VERSION_MIN_IPHONEOS
                if size != 16:
                    raise ValueError('Invalid LC_VERSION_MIN_IPHONEOS')
                minimum = struct.unpack_from(endian + 'I', blob, cursor + 8)[0]
                found.append((7 if cpu in (7, 0x1000007) else 2, minimum))
            cursor += size
        if cursor != header + length or len(found) != 1:
            raise ValueError('Missing or ambiguous Mach-O deployment version')
        platform, packed = found[0]
        minimum = (packed >> 16, (packed >> 8) & 255, packed & 255)
        if not minimum[0]:
            raise ValueError('Invalid Mach-O minimum version')
        return {'cpu': cpu, 'subtype': subtype, 'platform': platform, 'minimum': minimum}

    fats = {b'\xca\xfe\xba\xbe': ('>', False), b'\xbe\xba\xfe\xca': ('<', False),
            b'\xca\xfe\xba\xbf': ('>', True), b'\xbf\xba\xfe\xca': ('<', True)}
    if data[:4] not in fats:
        return [thin(data)]
    endian, wide = fats[data[:4]]
    if len(data) < 8:
        raise ValueError('Truncated fat Mach-O header')
    count = struct.unpack_from(endian + 'I', data, 4)[0]
    width = 32 if wide else 20
    if not 0 < count <= 32 or 8 + count * width > len(data):
        raise ValueError('Invalid fat Mach-O slice count')
    slices, spans = [], []
    for i in range(count):
        fields = struct.unpack_from(endian + ('IIQQII' if wide else 'IIIII'), data, 8 + i * width)
        cpu, subtype, offset, size = fields[:4]
        if size < 28 or offset < 8 + count * width or offset + size > len(data):
            raise ValueError('Invalid fat Mach-O slice bounds')
        if any(offset < end and offset + size > start for start, end in spans):
            raise ValueError('Overlapping fat Mach-O slices')
        value = thin(data[offset:offset + size])
        if value['cpu'] != cpu or value['subtype'] != subtype:
            raise ValueError('Fat Mach-O architecture mismatch')
        slices.append(value); spans.append((offset, offset + size))
    return slices


def check_bundle_version(info, executable, app_minimum=None, platform=2):
    declared = version(info.get('MinimumOSVersion'))
    slices = deployment_versions(executable)
    if any(item['platform'] != platform for item in slices):
        raise ValueError('Embedded executable targets the wrong Apple platform')
    required = max(item['minimum'] for item in slices)
    if required > declared:
        raise ValueError('MinimumOSVersion ' + version_text(declared) +
                         ' is below Mach-O requirement ' + version_text(required))
    if app_minimum is not None and declared > app_minimum:
        raise ValueError('Framework MinimumOSVersion exceeds the app deployment target')
    return {'declared': version_text(declared), 'binary_minimum': version_text(required),
            'slices': [{'cpu': item['cpu'], 'platform': item['platform'],
                        'minimum': version_text(item['minimum'])} for item in slices]}
