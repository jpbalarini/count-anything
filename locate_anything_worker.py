"""Resident locate-anything worker, spawned by locate_anything_detector.py.

Loads liblocate_anything (the C ABI of locate-anything.cpp, built with
-DLA_SHARED=ON) once, then answers one request per frame so the model is
not reloaded every time:

    python locate_anything_worker.py LIB MODEL THREADS MODE PROMPT

Protocol on stdin / stdout, both directions are length-prefixed messages
(4-byte little-endian size, then the payload):

    parent -> worker   an encoded image (JPEG / PNG bytes)
    worker -> parent   1 status byte (0 ok, 1 error) + body; for "ok" the
                       body is the detections JSON, or empty for the
                       "model loaded" message sent once at startup; for
                       "error" it is the error text

It is a separate process on purpose: if the GPU backend crashes, only
this process dies and the caller can skip the frame and start a new one.
stdin closing (or EOF) is the shutdown signal.
"""

from __future__ import annotations

import ctypes
import os
import signal
import struct
import sys

OK, ERROR = b"\x00", b"\x01"
MODES = {"hybrid": 0, "slow": 1, "fast": 2}


def main() -> None:
    lib_path, model_path, threads, mode, prompt = sys.argv[1:6]

    # Keep the protocol channel private: anything the C library prints to
    # stdout goes to stderr instead of corrupting the messages.
    out = os.fdopen(os.dup(1), "wb", buffering=0)
    os.dup2(2, 1)
    inp = sys.stdin.buffer
    # Ctrl-C reaches the whole process group; let the parent decide when
    # to stop us (it closes our stdin).
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    def send(status: bytes, body: bytes = b"") -> None:
        out.write(struct.pack("<I", 1 + len(body)) + status + body)

    lib = ctypes.CDLL(lib_path)
    lib.la_capi_load.restype = ctypes.c_void_p
    lib.la_capi_load.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.la_capi_free.argtypes = [ctypes.c_void_p]
    # c_void_p (not c_char_p) so the pointer can be handed back to free.
    lib.la_capi_locate_buffer.restype = ctypes.c_void_p
    lib.la_capi_locate_buffer.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    lib.la_capi_free_string.argtypes = [ctypes.c_void_p]
    lib.la_capi_last_error.restype = ctypes.c_char_p
    lib.la_capi_last_error.argtypes = [ctypes.c_void_p]

    ctx = lib.la_capi_load(model_path.encode(), int(threads))
    if not ctx:
        send(ERROR, f"could not load model {model_path}".encode())
        sys.exit(1)
    send(OK)

    prompt_b, mode_i = prompt.encode(), MODES[mode]
    try:
        while True:
            header = inp.read(4)
            if len(header) < 4:
                break
            (size,) = struct.unpack("<I", header)
            image = inp.read(size)
            if len(image) < size:
                break
            ptr = lib.la_capi_locate_buffer(
                ctx, image, len(image), prompt_b, mode_i
            )
            if ptr:
                send(OK, ctypes.string_at(ptr))
                lib.la_capi_free_string(ptr)
            else:
                send(ERROR, lib.la_capi_last_error(ctx) or b"unknown error")
    except BrokenPipeError:  # the parent went away
        pass
    finally:
        # Freeing the context matters: ggml's Metal backend aborts at exit
        # if its buffers are still alive.
        lib.la_capi_free(ctx)


if __name__ == "__main__":
    main()
