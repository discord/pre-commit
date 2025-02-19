import json
import shlex
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from pathlib import Path
from pre_commit import git, parse_shebang
from typing import Dict
from typing import Generator
from typing import List
from typing import Optional


@dataclass
class MetricRecord:
    name: str
    successful: bool
    start_time: datetime
    end_time: datetime

    def for_json(self) -> Dict[str, object]:
        return {
            'name': self.name,
            'success': self.successful,
            'startTimeSecs': self.start_time.timestamp(),
            'endTimeSecs': self.end_time.timestamp(),
        }


class Trace:
    def __init__(self) -> None:
        self.successful = True

    def set_success(self, successful: bool) -> None:
        self.successful = successful


class Monitor:
    """Metrics monitoring class indended to be used as a singleton."""

    def __init__(self) -> None:
        self.records: List[MetricRecord] = []
        self.report_command: Optional[List[str]] = None

    @contextmanager
    def trace(self, name: str) -> Generator[Trace, None, None]:
        """Record and trace the execution time and result of the wrapped sequence
        """
        start_time = datetime.now(timezone.utc)
        trace = Trace()

        try:
            yield trace
        except BaseException:
            trace.set_success(False)
            raise
        finally:
            end_time = datetime.now(timezone.utc)
            self.records.append(
                MetricRecord(
                    name=name, successful=trace.successful,
                    start_time=start_time, end_time=end_time,
                ),
            )

    def set_report_command(self, command: Optional[str]) -> None:
        if command is not None:
            self.report_command = shlex.split(command)
        else:
            self.report_command = None

    def report_metrics(self) -> None:
        if self.report_command:
            root = git.get_root()
            metrics_file = Path(root) / 'discord_clyde' / '.metrics.json'
            with open(metrics_file, 'w') as f:
                json.dump([record.for_json() for record in self.records], f)
            normalized_command = list(parse_shebang.normalize_cmd(tuple(self.report_command)))
            subprocess.run(normalized_command)


monitor = Monitor()
