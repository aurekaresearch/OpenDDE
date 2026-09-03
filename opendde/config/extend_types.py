# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
class DefaultNoneWithType(object):
    def __init__(self, dtype):
        self.dtype = dtype


class ValueMaybeNone(object):
    def __init__(self, value):
        assert value is not None
        self.dtype = type(value)
        self.value = value


class GlobalConfigValue(object):
    def __init__(self, global_key):
        self.global_key = global_key


class RequiredValue(object):
    def __init__(self, dtype):
        self.dtype = dtype


class ListValue(object):
    def __init__(self, value, dtype=None):
        if value:
            self.value = value
            self.dtype = type(value[0])
        elif value is not None:
            # Empty list (e.g. an "unset" default); element type can't be
            # inferred, so fall back to the explicit dtype.
            self.value = value
            self.dtype = dtype
        else:
            self.value = None
            self.dtype = dtype


def get_bool_value(bool_str):
    if isinstance(bool_str, bool):
        return bool_str
    bool_str_lower = bool_str.lower()
    if bool_str_lower in ("false", "f", "no", "n", "0"):
        return False
    elif bool_str_lower in ("true", "t", "yes", "y", "1"):
        return True
    else:
        raise ValueError(f"Cannot interpret {bool_str} as bool")
