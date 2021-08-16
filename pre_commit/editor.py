import argparse
import concurrent.futures
import contextlib
import platform
import shlex
import subprocess
from pathlib import Path
from typing import Any
from typing import Callable
from typing import List
from typing import TYPE_CHECKING
from typing import TypeVar
if TYPE_CHECKING:
    from typing import NoReturn

import psutil

from pre_commit import git
from pre_commit import output
from pre_commit.metrics import monitor

T = TypeVar('T')

# Keeping the file name the same as git's makes it more likely that editors will set the file type
# correctly when opening it.
COMMIT_MESSAGE_DRAFT_PATH = Path('.git/pre-commit/COMMIT_EDITMSG')
COMMIT_MESSAGE_EXPIRED_DRAFT_PATH = Path('.git/pre-commit/COMMIT_EDITMSG_OLD')
COMMIT_MESSAGE_HEADER = """
# Please enter the commit message for your changes. Lines starting
# with '#' will be ignored, and an empty message aborts the commit.
#
"""


def should_run_concurrently(hook_stage: str) -> bool:
    return (
        hook_stage == 'commit' and
        _is_editor_script_configured() and
        _should_open_editor() and
        platform.system != 'Windows'
    )


def run_concurrently(fun: Callable[..., T], *args: Any) -> T:
    # Allow user to enter commit message concurrently with running pre-commit hooks to lower
    # wait times.
    output.write_line('Waiting for your editor (pre-commit hooks running in background)...')

    # Run git commands before starting hooks to avoid race conditions and git lock errors.
    commit_message_template = _get_commit_message_template()

    with contextlib.ExitStack() as paused_stdout_stack:
        paused_stdout_stack.enter_context(output.paused_stdout())

        def launch_editor() -> None:
            _edit_commit_message(commit_message_template)
            paused_stdout_stack.close()  # Resume terminal output as soon as the editor closes
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            ex.submit(launch_editor)
            retval_future = ex.submit(fun, *args)
            return retval_future.result()
    raise RuntimeError('unreachable')


def should_clean_draft(hook_stage: str) -> bool:
    # We need to clean up the draft if one exists but we're not editing it, otherwise it's at risk
    # of being committed without further user interaction.
    return (
        hook_stage == 'commit' and
        _is_editor_script_configured() and
        not _should_open_editor() and
        COMMIT_MESSAGE_DRAFT_PATH.exists() and
        platform.system != 'Windows'
    )


def clean_draft() -> None:
    COMMIT_MESSAGE_DRAFT_PATH.rename(COMMIT_MESSAGE_EXPIRED_DRAFT_PATH)


class ParseFailed(BaseException):
    pass


class NoExitParser(argparse.ArgumentParser):
    # exit_on_error option only exists in 3.9+, so we have to do this ourselves.
    def error(self, _: str) -> 'NoReturn':
        raise ParseFailed()


def _should_open_editor() -> bool:
    git_invocation = psutil.Process().parent().cmdline()
    if git_invocation[:2] != ['git', 'commit']:
        # Some other command is being run; let's be conservative.
        return False
    try:
        # Teach the parser about all the allowable arguments to git commit and have it fail if any
        # others are present.
        parser = NoExitParser(add_help=False)

        def allowed_flag(*args: Any) -> None:
            parser.add_argument(*args, action='store_true')

        def allowed_option(*args: Any) -> None:
            parser.add_argument(*args)

        allowed_flag('-a', '--all')
        allowed_flag('-p', '--patch')
        allowed_flag('--reset-author')
        allowed_flag('--branch')
        allowed_flag('--allow-empty')
        allowed_flag('--allow-empty-message')
        allowed_flag('--no-post-rewrite')
        allowed_flag('--status')  # We always include status.
        allowed_flag('-i', '--include')
        allowed_flag('-o', '--only')
        allowed_flag('-q', '--quiet')
        allowed_option('--author')
        allowed_option('--date')
        allowed_option('--cleanup')
        allowed_option('--pathspec-from-file')
        allowed_flag('--pathspec-file-nul')

        # == Disallowed arguments ==
        # -m <msg>, --message=<msg>  # Alternate means of supplying a commit message.
        # -F <file>, --file=<file>  # Alternate means of supplying a commit message.
        # -C <commit>, --reuse-message=<commit>  # Alternate means of supplying a commit message.
        # -c <commit>, --reedit-message=<commit>  # Alternate means of supplying a commit message.

        # --amend  # We don't support showing the correct set of changes or previous commit message.
        # --fixup=<commit>  # We don't support automatically constructing a fixup commit message.
        # --squash=<commit>  # We don't support automatically constructing a squash commit message.
        # --dry-run  # No commit actually made.
        # --short  # Implies --dry-run.
        # --porcelain  # Implies --dry-run.
        # --long  # Implies --dry-run.
        # -z, --null  # Intended to be used with --short or --porcelain.
        # -t <file>, --template=<file>  # We don't support template files.
        # -s, --signoff  # We don't support adding a signoff line.
        # -n, --no-verify  # We shouldn't be called if this is passed.
        # -e, --edit  # We don't support editing a commit message supplied from other sources.
        # --no-edit  # Explicitly requests not launching an editor.
        # -u[<mode>], --untracked-files[=<mode>]  # We don't support showing untracked files differently.
        # -v, --verbose  # We don't support verbose git status.
        # --no-status  # We don't support not including the status.
        # -S, --gpg-sign[=<keyid>], --no-gpg-sign  # Optional arg is annoying to support; rarely used.

        parser.add_argument('pathspec', nargs='+')
        parser.parse_args(git_invocation)

        # Parse succeeded -- no unknown args.
        return True
    except ParseFailed:
        return False


def _is_editor_script_configured() -> bool:
    editor_script_path = git.get_editor_script_path()
    local_git_editor = _get_local_git_editor()
    return (
        bool(local_git_editor) and
        local_git_editor[0] == editor_script_path and
        Path(editor_script_path).exists()
    )


def _edit_commit_message(template: str) -> None:
    if not COMMIT_MESSAGE_DRAFT_PATH.exists():
        COMMIT_MESSAGE_DRAFT_PATH.parent.mkdir(parents=True, exist_ok=True)
        COMMIT_MESSAGE_DRAFT_PATH.write_text(template)
    else:
        # Update commit draft with new status
        commit_draft = COMMIT_MESSAGE_DRAFT_PATH.read_text()
        commit_draft = '\n'.join(line for line in commit_draft.splitlines() if not line.startswith('#'))
        COMMIT_MESSAGE_DRAFT_PATH.write_text(commit_draft + template)
    git_editor = _get_global_git_editor()  # Doesn't run in this repo, so the concurrency won't cause git lock errors.
    with monitor.trace('precommit.editor'):
        # For some editors (e.g. vim), it's important that stdin be an interactive terminal.
        # /dev/tty is a synonym for our process's controlling terminal.
        with open('/dev/tty') as stdin:
            subprocess.call(git_editor + [str(COMMIT_MESSAGE_DRAFT_PATH)], stdin=stdin)


def _get_local_git_editor() -> List[str]:
    editor_str = subprocess.run(['git', 'var', 'GIT_EDITOR'], check=True, capture_output=True).stdout.decode('utf-8')
    return shlex.split(editor_str)


def _get_global_git_editor() -> List[str]:
    # The repo-local editor has been set to a special script. This gets the globally configured
    # editor.
    editor_str = subprocess.run(
        ['git', 'var', 'GIT_EDITOR'], cwd='/',
        check=True, capture_output=True,
    ).stdout.decode('utf-8')
    return shlex.split(editor_str)


def _get_commit_message_template() -> str:
    status = subprocess.run(['git', 'status'], check=True, capture_output=True).stdout.decode('utf-8')
    commented_status = '\n'.join('# ' + line for line in status.splitlines())
    return COMMIT_MESSAGE_HEADER + commented_status
