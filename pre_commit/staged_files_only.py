from __future__ import annotations

import contextlib
import logging
import os.path
import time
from collections.abc import Generator

from pre_commit import git
from pre_commit.errors import FatalError
from pre_commit.util import CalledProcessError
from pre_commit.util import cmd_output
from pre_commit.util import cmd_output_b
from pre_commit.xargs import xargs


logger = logging.getLogger('pre_commit')

# without forcing submodule.recurse=0, changes in nested submodules will be
# discarded if `submodule.recurse=1` is configured
# we choose this instead of `--no-recurse-submodules` because it works on
# versions of git before that option was added to `git checkout`
_CHECKOUT_CMD = ('git', '-c', 'submodule.recurse=0', 'checkout', '--', '.')


def _git_apply(patch: str) -> None:
    args = ('apply', '--whitespace=nowarn', patch)
    try:
        cmd_output_b('git', *args)
    except CalledProcessError:
        # Retry with autocrlf=false -- see #570
        cmd_output_b('git', '-c', 'core.autocrlf=false', *args)


@contextlib.contextmanager
def _intent_to_add_cleared() -> Generator[None]:
    intent_to_add = git.intent_to_add_files()
    if intent_to_add:
        logger.warning('Unstaged intent-to-add files detected.')

        xargs(('git', 'rm', '--cached', '--'), intent_to_add)
        try:
            yield
        finally:
            xargs(('git', 'add', '--intent-to-add', '--'), intent_to_add)
    else:
        yield


@contextlib.contextmanager
def _unstaged_changes_cleared(patch_dir: str) -> Generator[None]:
    tree = cmd_output('git', 'write-tree')[1].strip()
    diff_cmd = (
        'git', 'diff-index', '--ignore-submodules', '--binary',
        '--exit-code', '--no-color', '--no-ext-diff', tree, '--',
    )
    retcode, diff_stdout, diff_stderr = cmd_output_b(*diff_cmd, check=False)
    if retcode == 0:
        # There weren't any staged files so we don't need to do anything
        # special
        yield
    elif retcode == 1 and not diff_stdout.strip():
        # due to behaviour (probably a bug?) in git with crlf endings and
        # autocrlf set to either `true` or `input` sometimes git will refuse
        # to show a crlf-only diff to us :(
        yield
    elif retcode == 1 and diff_stdout.strip():
        patch_filename = f'patch{int(time.time())}-{os.getpid()}'
        patch_filename = os.path.join(patch_dir, patch_filename)
        logger.warning('Unstaged files detected.')
        logger.info(f'Stashing unstaged files to {patch_filename}.')
        # Save the current unstaged changes as a patch
        os.makedirs(patch_dir, exist_ok=True)
        with open(patch_filename, 'wb') as patch_file:
            patch_file.write(diff_stdout)

        # prevent recursive post-checkout hooks (#1418)
        no_checkout_env = dict(os.environ, _PRE_COMMIT_SKIP_POST_CHECKOUT='1')

        try:
            cmd_output_b(*_CHECKOUT_CMD, env=no_checkout_env)
            yield
        finally:
            # Try to apply the patch we saved
            try:
                _git_apply(patch_filename)
            except CalledProcessError:
                logger.warning(
                    'Stashed changes conflicted with hook auto-fixes... '
                    'Rolling back fixes...',
                )
                # We failed to apply the patch, presumably due to fixes made
                # by hooks.
                # Roll back the changes made by hooks.
                # Save the current state of the index, as that could include partially staged files,
                # a whole successful new commit, etc.
                new_tree = cmd_output('git', 'write-tree')[1].strip()
                # Restore worktree to the patch base (which unavoidably updates the index as a side effect).
                cmd_output_b(*_CHECKOUT_CMD, env=no_checkout_env)
                # Apply patch.
                _git_apply(patch_filename)
                # Restore new saved index.
                cmd_output_b('git', 'read-tree', new_tree, env=no_checkout_env)

            logger.info(f'Restored changes from {patch_filename}.')
    else:  # pragma: win32 no cover
        # some error occurred while requesting the diff
        e = CalledProcessError(retcode, diff_cmd, b'', diff_stderr)
        raise FatalError(
            f'pre-commit failed to diff -- perhaps due to permissions?\n\n{e}',
        )


@contextlib.contextmanager
def staged_files_only(patch_dir: str) -> Generator[None]:
    """Clear any unstaged changes from the git working directory inside this
    context.
    """
    with _intent_to_add_cleared(), _unstaged_changes_cleared(patch_dir):
        yield
