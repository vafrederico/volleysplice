import type { HdrDecoderRequest, HdrDecoderResponse } from "./hdr-hevc-protocol";

type DecoderModule = {
  HEAPU8: Uint8Array;
  _malloc(size: number): number;
  _free(pointer: number): void;
  _vc_hdr_open(pointer: number, size: number): number;
  _vc_hdr_reset(): void;
  _vc_hdr_send(pointer: number, size: number, timestamp: number, duration: number): number;
  _vc_hdr_receive(): number;
  _vc_hdr_size(): number;
  _vc_hdr_copy(pointer: number, size: number): number;
  _vc_hdr_width(): number;
  _vc_hdr_height(): number;
  _vc_hdr_pts(): number;
  _vc_hdr_duration(): number;
  _vc_hdr_primaries(): number;
  _vc_hdr_transfer(): number;
  _vc_hdr_matrix(): number;
  _vc_hdr_range(): number;
};
let module: DecoderModule;
function checked(result: number): void {
  if (result < 0) throw new Error(`The HDR HEVC decoder failed (${result}).`);
}
function withBytes(bytes: Uint8Array, use: (pointer: number) => void): void {
  const pointer = module._malloc(bytes.byteLength);
  if (!pointer) throw new Error("Insufficient memory for HDR decoding.");
  try { module.HEAPU8.set(bytes, pointer); use(pointer); } finally { module._free(pointer); }
}
// The client sends one request at a time, including during module initialization.
self.onmessage = async (event: MessageEvent<HdrDecoderRequest>) => {
  const request = event.data;
  const response: HdrDecoderResponse = {};
  try {
    if (request.kind === "open") {
      const factory = (await import(/* @vite-ignore */ request.url)).default;
      module = await factory({ locateFile: (name: string) => new URL(name, request.url).href });
      withBytes(request.extra, pointer => checked(module._vc_hdr_open(pointer, request.extra.length)));
    } else if (request.kind === "reset") module._vc_hdr_reset();
    else if (request.kind === "send") {
      if (request.data) withBytes(request.data, pointer => checked(module._vc_hdr_send(pointer, request.data!.length, request.timestamp * 1e6, request.duration * 1e6)));
      else checked(module._vc_hdr_send(0, 0, 0, 0));
    } else {
      const result = module._vc_hdr_receive();
      checked(result);
      response.frame = null;
      if (result) {
        const size = module._vc_hdr_size();
        if (size <= 0) throw new Error("Invalid decoded HDR frame size.");
        const pointer = module._malloc(size);
        if (!pointer) throw new Error("Insufficient memory for HDR frame readback.");
        let pixels: ArrayBuffer;
        try {
          checked(module._vc_hdr_copy(pointer, size));
          pixels = module.HEAPU8.slice(pointer, pointer + size).buffer;
        } finally { module._free(pointer); }
        response.frame = {
          pixels, width: module._vc_hdr_width(), height: module._vc_hdr_height(),
          timestamp: module._vc_hdr_pts() / 1e6, duration: module._vc_hdr_duration() / 1e6,
          primaries: module._vc_hdr_primaries(), transfer: module._vc_hdr_transfer(),
          matrix: module._vc_hdr_matrix(), range: module._vc_hdr_range(),
        };
      }
    }
  } catch (error) { response.error = error instanceof Error ? error.message : String(error); }
  self.postMessage(response, { transfer: response.frame ? [response.frame.pixels] : [] });
};
