from __future__ import annotations

import contextlib
from dataclasses import dataclass
import io
import sys
from threading import Lock
from typing import Any, IO, Optional, Generator, cast

stdout_lock = Lock()

def write_b(b: bytes, stream: Optional[IO[bytes]] = None) -> None:
    if stream is None:
        with stdout_lock:
            stream = sys.stdout.buffer

    stream.write(b)
    stream.flush()

def write(s: str, stream: Optional[IO[bytes]] = None) -> None:
    write_b(s.encode(), stream)

def write_line_b(
        s: bytes | None = None,
        stream: Optional[IO[bytes]] = None,
        logfile_name: Optional[str] = None,
) -> None:
    if stream is None:
        with stdout_lock:
            stream = sys.stdout.buffer

    with contextlib.ExitStack() as exit_stack:
        output_streams = [stream]
        if logfile_name:
            stream = exit_stack.enter_context(open(logfile_name, 'ab'))
            output_streams.append(stream)

        for output_stream in output_streams:
            if s is not None:
                output_stream.write(s)
            output_stream.write(b'\n')
            output_stream.flush()


def write_line(s: str | None = None, **kwargs: Any) -> None:
    write_line_b(s.encode() if s is not None else s, **kwargs)


@contextlib.contextmanager
def paused_stdout() -> Generator[None, None, None]:
    redirected_output = io.TextIOWrapper(io.BytesIO())
    with contextlib.redirect_stdout(redirected_output):
        yield
        # We need to hold this lock through resetting stdout _and_ writing the saved contents,
        # because otherwise another thread might write to stdout before we can write the buffered
        # stdout, resulting in out-of-order output.
        stdout_lock.acquire()
    write_b(cast(io.BytesIO, redirected_output.buffer).getvalue(), sys.stdout.buffer)  # Supply buffer here so we don't deadlock.
    stdout_lock.release()
