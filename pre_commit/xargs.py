from __future__ import annotations

import concurrent.futures
import contextlib
import math
import multiprocessing
import os
import re
import select
import shutil
import subprocess
import sys
from collections.abc import Generator
from collections.abc import Iterable
from collections.abc import MutableMapping
from collections.abc import Sequence
from typing import Any
from typing import Callable
from typing import TypeVar

from pre_commit import parse_shebang
from pre_commit.util import cmd_output_b
from pre_commit.util import cmd_output_p
from pre_commit.output import stdout_lock

TArg = TypeVar('TArg')
TRet = TypeVar('TRet')


def cpu_count() -> int:
    try:
        # On systems that support it, this will return a more accurate count of
        # usable CPUs for the current process, which will take into account
        # cgroup limits
        return len(os.sched_getaffinity(0))
    except AttributeError:
        pass

    try:
        return multiprocessing.cpu_count()
    except NotImplementedError:
        return 1


def _environ_size(_env: MutableMapping[str, str] | None = None) -> int:
    environ = _env if _env is not None else getattr(os, 'environb', os.environ)
    size = 8 * len(environ)  # number of pointers in `envp`
    for k, v in environ.items():
        size += len(k) + len(v) + 2  # c strings in `envp`
    return size


def _get_platform_max_length() -> int:  # pragma: no cover (platform specific)
    if os.name == 'posix':
        maximum = os.sysconf('SC_ARG_MAX') - 2048 - _environ_size()
        maximum = max(min(maximum, 2 ** 17), 2 ** 12)
        return maximum
    elif os.name == 'nt':
        return 2 ** 15 - 2048  # UNICODE_STRING max - headroom
    else:
        # posix minimum
        return 2 ** 12


def _command_length(*cmd: str) -> int:
    full_cmd = ' '.join(cmd)

    # win32 uses the amount of characters, more details at:
    # https://github.com/pre-commit/pre-commit/pull/839
    if sys.platform == 'win32':
        return len(full_cmd.encode('utf-16le')) // 2
    else:
        return len(full_cmd.encode(sys.getfilesystemencoding()))


class ArgumentTooLongError(RuntimeError):
    pass


def partition(
        cmd: Sequence[str],
        varargs: Sequence[str],
        target_concurrency: int,
        _max_length: int | None = None,
) -> tuple[tuple[str, ...], ...]:
    _max_length = _max_length or _get_platform_max_length()

    # Generally, we try to partition evenly into at least `target_concurrency`
    # partitions, but we don't want a bunch of tiny partitions.
    max_args = max(4, math.ceil(len(varargs) / target_concurrency))

    cmd = tuple(cmd)
    ret = []

    ret_cmd: list[str] = []
    # Reversed so arguments are in order
    varargs = list(reversed(varargs))

    total_length = _command_length(*cmd) + 1
    while varargs:
        arg = varargs.pop()

        arg_length = _command_length(arg) + 1
        if (
                total_length + arg_length <= _max_length and
                len(ret_cmd) < max_args
        ):
            ret_cmd.append(arg)
            total_length += arg_length
        elif not ret_cmd:
            raise ArgumentTooLongError(arg)
        else:
            # We've exceeded the length, yield a command
            ret.append(cmd + tuple(ret_cmd))
            ret_cmd = []
            total_length = _command_length(*cmd) + 1
            varargs.append(arg)

    ret.append(cmd + tuple(ret_cmd))

    return tuple(ret)


@contextlib.contextmanager
def _thread_mapper(maxsize: int) -> Generator[
    Callable[[Callable[[TArg], TRet], Iterable[TArg]], Iterable[TRet]],
]:
    if maxsize == 1:
        yield map
    else:
        with concurrent.futures.ThreadPoolExecutor(maxsize) as ex:
            yield ex.map


def stream_subprocess_output(cmd, **kwargs) -> Generator[tuple[bytes, int | None], None, None]:
    """
    Run `cmd` as a subprocess and yield (chunk, returncode) tuples as output becomes available.
    Merged stdout + stderr (because of stderr=STDOUT).
    returncode is None until the process completes.
    """
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        **kwargs,
    )

    try:
        while True:
            process_done = (proc.poll() is not None)

            if not process_done:
                ready, _, _ = select.select([proc.stdout], [], [], 0.1)
                if not ready:
                    continue

            chunk = proc.stdout.read1(1024)
            if chunk:
                yield chunk, None
            else:
                if process_done:
                    break
    finally:
        proc.stdout.close()
        proc.wait()
        # Yield one final time with the returncode
        yield b'', proc.returncode


def xargs(
        cmd: tuple[str, ...],
        varargs: Sequence[str],
        *,
        color: bool = False,
        target_concurrency: int = 1,
        _max_length: int = _get_platform_max_length(),
        stream_output: bool | None = None,
        **kwargs: Any,
) -> tuple[int, bytes]:
    """A simplified implementation of xargs.

    color: Make a pty if on a platform that supports it
    target_concurrency: Target number of partitions to run concurrently
    """
    cmd_fn = cmd_output_p if color else cmd_output_b
    retcode = 0
    stdout = b''

    try:
        cmd = parse_shebang.normalize_cmd(cmd)
    except parse_shebang.ExecutableNotFoundError as e:
        return e.to_output()[:2]

    # on windows, batch files have a separate length limit than windows itself
    if (
            sys.platform == 'win32' and
            cmd[0].lower().endswith(('.bat', '.cmd'))
    ):  # pragma: win32 cover
        # this is implementation details but the command gets translated into
        # full/path/to/cmd.exe /c *cmd
        cmd_exe = parse_shebang.find_executable('cmd.exe')
        # 1024 is additionally subtracted to give headroom for further
        # expansion inside the batch file
        _max_length = 8192 - len(cmd_exe) - len(' /c ') - 1024

    partitions = partition(cmd, varargs, target_concurrency, _max_length)

    def run_cmd_partition(
            run_cmd: tuple[str, ...],
    ) -> tuple[int, bytes, bytes | None]:
        if not stream_output:
            return cmd_fn(
                *run_cmd, check=False, stderr=subprocess.STDOUT, **kwargs,
            )

        output = b''
        returncode = 0

        with stdout_lock:
            sys.stdout.buffer.write(b'\n')
            sys.stdout.buffer.flush()

        for chunk, maybe_returncode in stream_subprocess_output(cmd):
            output += chunk
            with stdout_lock:
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
            if maybe_returncode is not None:
                returncode = maybe_returncode

        terminal_width = shutil.get_terminal_size((80, 20)).columns
        strip_ansi = re.compile(rb'\x1B\[[0-?]*[ -/]*[@-~]')
        plain_output = strip_ansi.sub(b'', output)
        standard_lines = plain_output.split(b'\n')
        line_count = 0

        for line in standard_lines:
            displayed_width = max(1, len(line))
            line_count += math.ceil(displayed_width / terminal_width)

        with stdout_lock:
            sys.stdout.buffer.write(b'\0337')  # Save cursor position
            # Move cursor back to original line and move 73 columns to the right, where the status result begins
            sys.stdout.buffer.write(f'\033[{line_count}A\033[73C'.encode())
            sys.stdout.buffer.flush()
        
        return returncode, output, None

    threads = min(len(partitions), target_concurrency)
    with _thread_mapper(threads) as thread_map:
        results = thread_map(run_cmd_partition, partitions)

        for proc_retcode, proc_out, _ in results:
            if abs(proc_retcode) > abs(retcode):
                retcode = proc_retcode
            stdout += proc_out

    return retcode, stdout
