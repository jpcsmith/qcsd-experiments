"""Helper functions and classes for using docopt and schema"""
# pylint: disable=too-few-public-methods
import sys
import csv
import typing
from typing import IO

from docopt import docopt
from schema import (
    SchemaError, And, Schema, Regex, Optional, Use, Forbidden, Const,
    Literal, Or
)

__all__ = [
    'Schema', 'And', 'Or', 'Regex', 'Optional', 'Use', 'Forbidden',
    'Const', 'Literal', 'PositiveInt', 'positive_int', 'AtLeast', 'doceasy',
    'CsvFile',
]


PositiveInt = And(Use(int), lambda n: n >= 1,
                  error="Value should be an integer and at least 1")


def positive_int(value: str):
    """Extracts a positive integer from a string.

    Raises ValueError if the string does not contain a positive integer.
    """
    integer = int(value)
    if integer < 1:
        raise ValueError(f"invalid literal for a positive integer: '{value}'")
    return integer


class AtLeast:
    """Validator to ensure that the argument is at least the value
    specified in the constructor.
    """
    def __init__(self, min_value):
        self.min_value = min_value

    def validate(self, value):
        """Attempt to validate the provided value."""
        if value < self.min_value:
            raise SchemaError(f"The value should be at least {self.min_value}")
        return value

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, str(self.min_value))


class Mapping:
    """Validator that creates mappings.

    The parameters kt_callable and vt_callable if provided should be
    callables, such as "int", returning the desired type.  If only
    kt_callable is provided it is used to conver the value in the
    mapping.  If both are provided kt_callable converts the key and
    vt_callable converts the value.
    """
    def __init__(self, kt_callable=None, vt_callable=None):
        if vt_callable is None:
            self._kt_callable = str
            self._vt_callable = kt_callable or str
        else:
            assert kt_callable is not None
            self._kt_callable = kt_callable
            self._vt_callable = vt_callable

    def validate(self, map_string: str) -> typing.Dict[str, typing.Any]:
        """Validate and extract the mapping."""
        try:
            items = [key_val.split("=", maxsplit=1)
                     for key_val in map_string.split(",")]
            return {self._kt_callable(key): self._vt_callable(value)
                    for key, value in items}
        except ValueError as err:
            raise SchemaError(
                f"Invalid mapping string for callables {map_string}") from err

    @staticmethod
    def to_string(mapping) -> str:
        """Convert the mapping to a string parsable by a Mapping
        validator.
        """
        return ",".join(f"{key}={value}" for key, value in mapping.items())


class File:
    """Validator that creates file objects for command line files or '-'.
    """
    def __init__(self, mode: str = 'r', default: typing.Optional[str] = None):
        self.mode = mode
        self.default = default

    def validate(self, filename: typing.Optional[str]) -> IO:
        """Validate the filename and return the associated file object."""
        filename = filename or self.default
        stdout = sys.stdout.buffer if 'b' in self.mode else sys.stdout
        stdin = sys.stdin.buffer if 'b' in self.mode else sys.stdin

        if filename == '-':
            if any(m in self.mode for m in ['w', 'a', 'x']):
                return stdout  # type: ignore
            return stdin  # type: ignore

        if filename is None:
            raise SchemaError("Invalid object to create a file: '{filename}'")

        try:
            return open(filename, mode=self.mode)
        except Exception as err:
            raise SchemaError(str(err)) from err


class CsvFile(File):
    """Validate and create a csv input/output file.

    If dict_args is not None, a DictReader/-Writer will be created.
    """
    def __init__(self, *args, dict_args: dict = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.dict_args = dict_args

    def validate(self, filename: typing.Optional[str]):
        stream = super().validate(filename)

        if any(m in self.mode for m in ['w', 'a', 'x']):
            if self.dict_args is not None:
                return csv.DictWriter(stream, **self.dict_args)
            return csv.writer(stream)

        if self.dict_args is not None:
            return csv.DictReader(stream, **self.dict_args)
        return csv.reader(stream)


def _validate(arguments: dict, schema: Schema) -> dict:
    try:
        return schema.validate(arguments)
    except SchemaError as err:
        sys.exit(f"Invalid argument: {err}")


def _rename_arguments(arguments: dict):
    return {
        key.lower().strip('-<>').replace('-', '_'): value
        for key, value in arguments.items()
    }


def doceasy(
    docstring: str,
    schema: typing.Union[Schema, typing.Dict, None] = None,
    rename: bool = True, **kwargs
) -> dict:
    """Parse the command line arguments."""
    arguments = docopt(docstring, **kwargs)

    if isinstance(schema, dict):
        schema = Schema(schema)
    if schema is not None:
        arguments = _validate(arguments, schema)

    if rename:
        arguments = _rename_arguments(arguments)

    return arguments
