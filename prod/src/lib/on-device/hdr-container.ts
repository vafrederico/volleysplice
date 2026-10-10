/** Validate Dolby Vision's base-layer compatibility before discarding dynamic metadata. */
export function validateDolbyVisionRecord(record: Uint8Array, transfer: string): void {
  if (record.length < 5) throw new Error("Truncated Dolby Vision configuration.");
  const profile = record[2] >> 1;
  const enhancementLayer = (record[3] >> 1) & 1;
  const baseLayer = record[3] & 1;
  const compatibility = record[4] >> 4;
  if (profile !== 8 || enhancementLayer || !baseLayer ||
      !((compatibility === 4 && transfer === "hlg") || (compatibility === 1 && transfer === "pq"))) {
    throw new Error("This Dolby Vision profile has no supported HLG/PQ base layer. HDR export supports compatible Profile 8 recordings.");
  }
}

const typeAt = (bytes: Uint8Array, offset: number) => String.fromCharCode(...bytes.subarray(offset, offset + 4));
export function validateHdrMovieMetadata(bytes: Uint8Array, transfer: string): void {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  function walk(start: number, end: number, depth: number): void {
    if (depth > 12) throw new Error("Invalid movie box nesting.");
    for (let offset = start; offset < end;) {
      if (offset + 8 > end) throw new Error("Truncated movie metadata.");
      let size = view.getUint32(offset), header = 8;
      const type = typeAt(bytes, offset + 4);
      if (size === 1) {
        if (offset + 16 > end) throw new Error("Truncated movie box.");
        size = Number(view.getBigUint64(offset + 8)); header = 16;
      } else if (size === 0) size = end - offset;
      if (!Number.isSafeInteger(size) || size < header || offset + size > end) throw new Error("Invalid movie box size.");
      const body = offset + header, next = offset + size;
      if (["moov", "trak", "mdia", "minf", "stbl"].includes(type)) walk(body, next, depth + 1);
      else if (type === "stsd") walk(body + 8, next, depth + 1);
      else if (["hvc1", "hev1", "dvh1", "dvhe"].includes(type)) walk(body + 78, next, depth + 1);
      else if (type === "dvcC" || type === "dvvC") validateDolbyVisionRecord(bytes.subarray(body, next), transfer);
      offset = next;
    }
  }
  walk(0, bytes.length, 0);
}

export async function validateHdrContainer(file: Blob, transfer: string): Promise<void> {
  // Read only top-level headers and bounded moov metadata, never the whole movie.
  for (let offset = 0, boxes = 0; offset < file.size && boxes < 10000; boxes++) {
    const header = new Uint8Array(await file.slice(offset, offset + 16).arrayBuffer());
    if (header.length < 8) break;
    const view = new DataView(header.buffer);
    let size = view.getUint32(0);
    const type = typeAt(header, 4);
    if (boxes === 0 && !["ftyp", "moov", "mdat", "free", "wide", "skip"].includes(type)) {
      throw new Error("HDR HEVC export currently requires an MP4 or QuickTime movie.");
    }
    if (size === 1 && header.length >= 16) size = Number(view.getBigUint64(8));
    else if (size === 0) size = file.size - offset;
    if (!Number.isSafeInteger(size) || size < 8 || offset + size > file.size) throw new Error("Invalid movie container.");
    if (type === "moov") {
      if (size > 32 * 1024 * 1024) throw new Error("HDR movie metadata exceeds the supported size.");
      validateHdrMovieMetadata(new Uint8Array(await file.slice(offset, offset + size).arrayBuffer()), transfer);
      return;
    }
    offset += size;
  }
  throw new Error("HDR movie metadata is missing.");
}
