import concurrent.futures
import io
import shlex
import argparse
import contextlib
import functools
import logging
import os
import re
import subprocess
import time
import unicodedata
from pathlib import Path
from typing import Any
from typing import Collection
from typing import Dict
from typing import List
from typing import MutableMapping
from typing import Sequence
from typing import Set
from typing import Tuple

from identify.identify import tags_from_path

from pre_commit import color
from pre_commit import git
from pre_commit import output
from pre_commit.clientlib import load_config
from pre_commit.hook import Hook
from pre_commit.languages.all import languages
from pre_commit.repository import all_hooks
from pre_commit.repository import install_hook_envs
from pre_commit.staged_files_only import staged_files_only
from pre_commit.store import Store
from pre_commit.util import cmd_output_b
from pre_commit.metrics import monitor


# Keeping the file name the same as git's makes it more likely that editors will set the file type
# correctly when opening it.
COMMIT_MESSAGE_DRAFT_PATH = Path('.git/pre-commit/COMMIT_EDITMSG')

logger = logging.getLogger('pre_commit')


def _len_cjk(msg: str) -> int:
    widths = {'A': 1, 'F': 2, 'H': 1, 'N': 1, 'Na': 1, 'W': 2}
    return sum(widths[unicodedata.east_asian_width(c)] for c in msg)


def _start_msg(*, start: str, cols: int, end_len: int) -> str:
    dots = '.' * (cols - _len_cjk(start) - end_len - 1)
    return f'{start}{dots}'


def _full_msg(
        *,
        start: str,
        cols: int,
        end_msg: str,
        end_color: str,
        use_color: bool,
        postfix: str = '',
) -> str:
    dots = '.' * (cols - _len_cjk(start) - len(postfix) - len(end_msg) - 1)
    end = color.format_color(end_msg, end_color, use_color)
    return f'{start}{dots}{postfix}{end}\n'


def filter_by_include_exclude(
        names: Collection[str],
        include: str,
        exclude: str,
) -> List[str]:
    include_re, exclude_re = re.compile(include), re.compile(exclude)
    return [
        filename for filename in names
        if include_re.search(filename)
        if not exclude_re.search(filename)
    ]


class Classifier:
    def __init__(self, filenames: Collection[str]) -> None:
        self.filenames = [f for f in filenames if os.path.lexists(f)]

    @functools.lru_cache(maxsize=None)
    def _types_for_file(self, filename: str) -> Set[str]:
        return tags_from_path(filename)

    def by_types(
            self,
            names: Sequence[str],
            types: Collection[str],
            types_or: Collection[str],
            exclude_types: Collection[str],
    ) -> List[str]:
        types = frozenset(types)
        types_or = frozenset(types_or)
        exclude_types = frozenset(exclude_types)
        ret = []
        for filename in names:
            tags = self._types_for_file(filename)
            if (
                    tags >= types and
                    (not types_or or tags & types_or) and
                    not tags & exclude_types
            ):
                ret.append(filename)
        return ret

    def filenames_for_hook(self, hook: Hook) -> Tuple[str, ...]:
        names = self.filenames
        names = filter_by_include_exclude(names, hook.files, hook.exclude)
        names = self.by_types(
            names,
            hook.types,
            hook.types_or,
            hook.exclude_types,
        )
        return tuple(names)

    @classmethod
    def from_config(
            cls,
            filenames: Collection[str],
            include: str,
            exclude: str,
    ) -> 'Classifier':
        # on windows we normalize all filenames to use forward slashes
        # this makes it easier to filter using the `files:` regex
        # this also makes improperly quoted shell-based hooks work better
        # see #1173
        if os.altsep == '/' and os.sep == '\\':
            filenames = [f.replace(os.sep, os.altsep) for f in filenames]
        filenames = filter_by_include_exclude(filenames, include, exclude)
        return Classifier(filenames)


def _get_skips(environ: MutableMapping[str, str]) -> Set[str]:
    skips = environ.get('SKIP', '')
    return {skip.strip() for skip in skips.split(',') if skip.strip()}


NO_FILES = '(no files to check)'


def _subtle_line(s: str, use_color: bool) -> None:
    output.write_line(color.format_color(s, color.SUBTLE, use_color))


def _run_single_hook(
        classifier: Classifier,
        hook: Hook,
        skips: Set[str],
        cols: int,
        diff_before: bytes,
        verbose: bool,
        use_color: bool,
        stage: str,
) -> Tuple[bool, bytes]:
    filenames = classifier.filenames_for_hook(hook)

    if hook.id in skips or hook.alias in skips:
        duration = None
        retcode = 0
        diff_after = diff_before
        files_modified = False
        hook_failed = False
        out = b''
    elif not filenames and not hook.always_run:
        duration = None
        retcode = 0
        diff_after = diff_before
        files_modified = False
        hook_failed = False
        out = b''
    else:
        # print hook and dots first in case the hook takes a while to run
        output.write(_start_msg(start=hook.name, end_len=6, cols=cols))

        if not hook.pass_filenames:
            filenames = ()
        time_before = time.time()
        language = languages[hook.language]
        with monitor.trace(f'precommit.{stage}.hook.{hook.name}') as trace:
            retcode, out = language.run_hook(hook, filenames, use_color)

            duration = round(time.time() - time_before, 2) or 0
            diff_after = _get_diff()

            # if the hook makes changes, fail the commit or add the changes automatically
            files_modified = diff_before != diff_after
            # We can't easily commit changes if another hook has made uncomitted modifications.
            # In that case, just fail the hook instead.
            can_commit_changes = hook.commit_changes and not diff_before
            hook_failed = bool(retcode) or (files_modified and not can_commit_changes)
            trace.set_success(not hook_failed)

        if files_modified and can_commit_changes:
            git.update_changes_concurrent()
            # All changes should now be added -- there should no longer be any diff.
            diff_after = _get_diff()
            assert not diff_after

        if hook_failed:
            print_color = color.RED
            status = 'Failed'
        else:
            print_color = color.GREEN
            status = 'Passed'

        output.write_line(color.format_color(status, print_color, use_color))

    if verbose or hook.verbose or retcode or files_modified:
        _subtle_line(f'- hook id: {hook.id}', use_color)

        if (verbose or hook.verbose) and duration is not None:
            _subtle_line(f'- duration: {duration}s', use_color)

        if retcode:
            _subtle_line(f'- exit code: {retcode}', use_color)

        # Print a message if failing due to file modifications
        if files_modified:
            _subtle_line('- files were modified by this hook', use_color)

        if out.strip():
            output.write_line()
            output.write_line_b(out.strip(), logfile_name=hook.log_file)
            output.write_line()

    return hook_failed, diff_after


def _compute_cols(hooks: Sequence[Hook]) -> int:
    """Compute the number of columns to display hook messages.  The widest
    that will be displayed is in the no files case:

        Hook name...(no files to check) Passed
    """
    if hooks:
        name_len = max(_len_cjk(hook.name) for hook in hooks)
    else:
        name_len = 0

    max_status_size = 6  # Passed or Failed
    cols = name_len + 3 + len(NO_FILES) + 1 + max_status_size
    return max(cols, 80)


def _all_filenames(args: argparse.Namespace) -> Collection[str]:
    # these hooks do not operate on files
    if args.hook_stage in {'post-checkout', 'post-commit', 'post-merge'}:
        return ()
    elif args.hook_stage in {'prepare-commit-msg', 'commit-msg'}:
        return (args.commit_msg_filename,)
    elif args.from_ref and args.to_ref:
        return git.get_changed_files(args.from_ref, args.to_ref)
    elif args.files:
        return args.files
    elif args.all_files:
        return git.get_all_files()
    elif git.is_in_merge_conflict():
        return git.get_conflicted_files()
    else:
        return git.get_staged_files()


def _get_diff() -> bytes:
    _, out, _ = cmd_output_b(
        'git', 'diff', '--no-ext-diff', '--ignore-submodules', retcode=None,
    )
    return out


def _run_hooks(
        config: Dict[str, Any],
        hooks: Sequence[Hook],
        skips: Set[str],
        args: argparse.Namespace,
        environ: MutableMapping[str, str],
) -> int:
    """Actually run the hooks."""
    cols = _compute_cols(hooks)
    classifier = Classifier.from_config(
        _all_filenames(args), config['files'], config['exclude'],
    )
    retval = 0
    prior_diff = _get_diff()
    for hook in hooks:
        current_retval, prior_diff = _run_single_hook(
            classifier, hook, skips, cols, prior_diff,
            verbose=args.verbose, use_color=args.color,
            stage=args.hook_stage,
        )
        retval |= current_retval
        if retval and config['fail_fast']:
            break
    if retval and args.show_diff_on_failure and prior_diff:
        if args.all_files:
            output.write_line(
                'pre-commit hook(s) made changes.\n'
                'If you are seeing this message in CI, '
                'reproduce locally with: `pre-commit run --all-files`.\n'
                'To run `pre-commit` as part of git workflow, use '
                '`pre-commit install`.',
            )
        output.write_line('All changes made by hooks:')
        # args.color is a boolean.
        # See user_color function in color.py
        git_color_opt = 'always' if args.color else 'never'
        subprocess.call((
            'git', '--no-pager', 'diff', '--no-ext-diff',
            f'--color={git_color_opt}',
        ))

    return retval


def _has_unmerged_paths() -> bool:
    _, stdout, _ = cmd_output_b('git', 'ls-files', '--unmerged')
    return bool(stdout.strip())


def _has_unstaged_config(config_file: str) -> bool:
    retcode, _, _ = cmd_output_b(
        'git', 'diff', '--no-ext-diff', '--exit-code', config_file,
        retcode=None,
    )
    # be explicit, other git errors don't mean it has an unstaged config.
    return retcode == 1


def run(
        config_file: str,
        store: Store,
        args: argparse.Namespace,
        environ: MutableMapping[str, str] = os.environ,
) -> int:
    stash = not args.all_files and not args.files

    # Check if we have unresolved merge conflict files and fail fast.
    if _has_unmerged_paths():
        logger.error('Unmerged files.  Resolve before committing.')
        return 1
    if bool(args.from_ref) != bool(args.to_ref):
        logger.error('Specify both --from-ref and --to-ref.')
        return 1
    if stash and _has_unstaged_config(config_file):
        logger.error(
            f'Your pre-commit configuration is unstaged.\n'
            f'`git add {config_file}` to fix this.',
        )
        return 1
    if (
            args.hook_stage in {'prepare-commit-msg', 'commit-msg'} and
            not args.commit_msg_filename
    ):
        logger.error(
            f'`--commit-msg-filename` is required for '
            f'`--hook-stage {args.hook_stage}`',
        )
        return 1
    # prevent recursive post-checkout hooks (#1418)
    if (
            args.hook_stage == 'post-checkout' and
            environ.get('_PRE_COMMIT_SKIP_POST_CHECKOUT')
    ):
        return 0

    # Expose from-ref / to-ref as environment variables for hooks to consume
    if args.from_ref and args.to_ref:
        # legacy names
        environ['PRE_COMMIT_ORIGIN'] = args.from_ref
        environ['PRE_COMMIT_SOURCE'] = args.to_ref
        # new names
        environ['PRE_COMMIT_FROM_REF'] = args.from_ref
        environ['PRE_COMMIT_TO_REF'] = args.to_ref

    if (
        args.remote_name and args.remote_url and
        args.remote_branch and args.local_branch
    ):
        environ['PRE_COMMIT_LOCAL_BRANCH'] = args.local_branch
        environ['PRE_COMMIT_REMOTE_BRANCH'] = args.remote_branch
        environ['PRE_COMMIT_REMOTE_NAME'] = args.remote_name
        environ['PRE_COMMIT_REMOTE_URL'] = args.remote_url

    if args.checkout_type:
        environ['PRE_COMMIT_CHECKOUT_TYPE'] = args.checkout_type

    if args.is_squash_merge:
        environ['PRE_COMMIT_IS_SQUASH_MERGE'] = args.is_squash_merge

    # Set pre_commit flag
    environ['PRE_COMMIT'] = '1'

    with contextlib.ExitStack() as exit_stack:
        # Metrics should get reported as the last thing that happens.
        exit_stack.callback(monitor.report_metrics)

        # Start the timing trace.
        trace = exit_stack.enter_context(monitor.trace(f'precommit.{args.hook_stage}'))
        if stash:
            exit_stack.enter_context(staged_files_only(store.directory))

        config = load_config(config_file)
        monitor.set_report_command(config['metrics_command'])
        hooks = [
            hook
            for hook in all_hooks(config, store)
            if not args.hook or hook.id == args.hook or hook.alias == args.hook
            if args.hook_stage in hook.stages
        ]

        if args.hook and not hooks:
            output.write_line(
                f'No hook with id `{args.hook}` in stage `{args.hook_stage}`',
            )
            trace.set_success(False)
            return 1

        skips = _get_skips(environ)
        to_install = [hook for hook in hooks if hook.id not in skips]
        install_hook_envs(to_install, store)

        if args.hook_stage == 'commit' and not _is_git_message_supplied() and _is_editor_script_configured():
            # Allow user to enter commit message concurrently with running pre-commit hooks to lower
            # wait times.

            # Run git commands before starting hooks to avoid race conditions and git lock errors.
            commit_message_template = _get_commit_message_template()

            with contextlib.ExitStack() as paused_stdout_stack:
                paused_stdout_stack.enter_context(output.paused_stdout())
                def launch_editor():
                    _edit_commit_message(commit_message_template)
                    paused_stdout_stack.close()  # Resume terminal output as soon as the editor closes
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
                    ex.submit(launch_editor)
                    retval_future = ex.submit(_run_hooks, config, hooks, skips, args, environ)
            retval = retval_future.result()
        else:
            retval = _run_hooks(config, hooks, skips, args, environ)

        trace.set_success(retval == 0)
        return retval


    # https://github.com/python/mypy/issues/7726
    raise AssertionError('unreachable')

def _is_editor_script_configured() -> bool:
    editor_script_path = git.get_editor_script_path()
    local_git_editor = _get_local_git_editor()
    return (bool(local_git_editor)
            and local_git_editor[0] == editor_script_path
            and os.path.exists(editor_script_path)
           )


def _get_local_git_editor() -> List[str]:
    editor_str = subprocess.run(['git', 'var', 'GIT_EDITOR'], check=True, capture_output=True).stdout.decode('utf-8')
    return shlex.split(editor_str)

def _get_global_git_editor() -> List[str]:
    # The repo-local editor has been set to a special script. This gets the globally configured
    # editor.
    editor_str = subprocess.run(['git', 'var', 'GIT_EDITOR'], cwd='/', check=True, capture_output=True).stdout.decode('utf-8')
    return shlex.split(editor_str)

def _get_commit_message_template() -> str:
    initial_text = """\
# Please enter the commit message for your changes. Lines starting
# with '#' will be ignored, and an empty message aborts the commit.
#
"""
    status = subprocess.run(['git', 'status'], check=True, capture_output=True).stdout.decode('utf-8')
    commented_status = "\n".join('# ' + line for line in status.splitlines())
    return initial_text + commented_status


def _edit_commit_message(template: str) -> None:
    if not COMMIT_MESSAGE_DRAFT_PATH.exists():
        COMMIT_MESSAGE_DRAFT_PATH.parent.mkdir(parents=True, exist_ok=True)
        COMMIT_MESSAGE_DRAFT_PATH.write_text(template)

    git_editor = _get_global_git_editor()  # Doesn't run in this repo, so the concurrency won't cause git lock errors.
    with monitor.trace('precommit.editor'):
        subprocess.call(git_editor + [str(COMMIT_MESSAGE_DRAFT_PATH)])


def _is_git_message_supplied() -> bool:
    # TODO: this
    import psutil
    git_invocation = psutil.Process().parent().cmdline()
    return '-m' in git_invocation


def _run_auto_editor(commit_msg_filename: str) -> int:
    if COMMIT_MESSAGE_DRAFT_PATH.exists():
        commit_msg_path = Path(commit_msg_filename)
        commit_msg_path.write_text(COMMIT_MESSAGE_DRAFT_PATH.read_text())
        COMMIT_MESSAGE_DRAFT_PATH.unlink()
        return 0
    else:
        git_editor = _get_global_git_editor()  # Doesn't run in this repo, so the concurrency won't cause git lock errors.
        return subprocess.call(git_editor + [commit_msg_filename])
        
