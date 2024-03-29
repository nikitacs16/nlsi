import contextlib
import io
import pathlib
import sys
import traceback
import warnings
from dataclasses import dataclass
from io import SEEK_SET, TextIOBase
from typing import Iterator, List, Optional, TextIO


@dataclass
class Tee(TextIOBase):
    """ "A write-only file-like object that forwards writes to `sinks`."""

    sinks: List[TextIO]
    closed: bool = False

    def close(self) -> None:
        self.flush()
        self.closed = True

    def fileno(self) -> int:
        raise OSError

    def flush(self) -> None:
        for sink in self.sinks:
            sink.flush()

    def isatty(self) -> bool:
        return False

    def readable(self) -> bool:
        return False

    def readline(self, size=-1) -> str:
        raise io.UnsupportedOperation

    def readlines(self, hint=-1) -> List[str]:
        raise io.UnsupportedOperation

    def seek(self, offset, whence=SEEK_SET) -> int:
        raise io.UnsupportedOperation

    def seekable(self) -> bool:
        return False

    def tell(self) -> int:
        raise io.UnsupportedOperation

    def truncate(self, size=None):
        raise io.UnsupportedOperation

    def writable(self) -> bool:
        return True

    def writelines(self, lines: List[str]) -> None:
        for sink in self.sinks:
            sink.writelines(lines)

    @property
    def encoding(self) -> str:
        return self.sinks[0].encoding

    @property
    def errors(self) -> Optional[str]:
        return self.sinks[0].errors

    def detach(self) -> None:
        raise io.UnsupportedOperation

    def read(self, size=-1) -> str:
        raise io.UnsupportedOperation

    def write(self, s: str) -> int:
        results: List[int] = []
        for sink in self.sinks:
            results.append(sink.write(s))
        if not all(r == results[0] for r in results[1:]):
            warnings.warn("Sinks wrote different number of characters", ResourceWarning)
        return results[0]


@contextlib.contextmanager
def intercept_output(
    stdout_path: pathlib.Path, stderr_path: pathlib.Path
) -> Iterator[None]:
    """Write all stdout and stderr to both the screen and these files."""

    with open(stdout_path, "a") as stdout_file, open(stderr_path, "a") as stderr_file:
        true_stdout = sys.stdout
        true_stderr = sys.stderr
        sys.stdout = Tee([true_stdout, stdout_file])
        sys.stderr = Tee([true_stdout, stderr_file])
        try:
            yield
        except:
            traceback.print_exc(file=stderr_file)
            raise
        finally:
            sys.stdout = true_stdout
            sys.stderr = true_stderr
