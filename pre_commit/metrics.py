from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import shlex
import subprocess
from typing import List, Generator, Dict, Optional
import json

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
        except:
            trace.set_success(False)
            raise
        finally:
            end_time = datetime.now(timezone.utc)
            self.records.append(MetricRecord(name=name, successful=trace.successful, start_time=start_time, end_time=end_time))

    def set_report_command(self, command: str) -> None:
        self.report_command = shlex.split(command)

    def report_metrics(self) -> None:
        if self.report_command:
            record_json = json.dumps([record.for_json() for record in self.records])
            subprocess.run(self.report_command + [record_json])

monitor = Monitor()
