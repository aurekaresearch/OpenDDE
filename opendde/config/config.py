# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import argparse
import copy
import shlex
import sys
from typing import Any, Optional

from ml_collections.config_dict import ConfigDict

from opendde.config.extend_types import (
    DefaultNoneWithType,
    GlobalConfigValue,
    ListValue,
    RequiredValue,
    ValueMaybeNone,
    get_bool_value,
)


class ArgumentNotSet(object):
    pass


class ConfigManager(object):
    """
    Initialize the ConfigManager instance.

    Args:
        global_configs (dict): A dictionary containing global configuration settings.
        fill_required_with_null (bool, optional):
            A boolean flag indicating whether required values should be filled with `None` if not provided. Defaults to False.
    """

    def __init__(self, global_configs: dict, fill_required_with_null: bool = False):
        self.global_configs = global_configs
        self.fill_required_with_null = fill_required_with_null
        self.config_infos, self.default_configs = self.get_config_infos()

    def get_value_info(
        self, value
    ) -> tuple[Any, Optional[Any], Optional[bool], Optional[bool]]:
        """
        Return the type, default value, whether it allows None, and whether it is required for a given value.

        Args:
            value: The value to determine the information for.

        Returns:
            tuple: A tuple containing the following elements:
                - dtype: The type of the value.
                - default_value: The default value for the value.
                - allow_none: A boolean indicating whether the value can be None.
                - required: A boolean indicating whether the value is required.
        """
        if isinstance(value, DefaultNoneWithType):
            return value.dtype, None, True, False
        elif isinstance(value, ValueMaybeNone):
            return value.dtype, value.value, True, False
        elif isinstance(value, RequiredValue):
            if self.fill_required_with_null:
                return value.dtype, None, True, False
            else:
                return value.dtype, None, False, True
        elif isinstance(value, GlobalConfigValue):
            return self.get_value_info(self.global_configs[value.global_key])
        elif isinstance(value, ListValue):
            return (value.dtype, value.value, False, False)
        elif isinstance(value, list):
            return (type(value[0]), value, False, False)
        else:
            return type(value), value, False, False

    def _get_config_infos(self, config_dict: dict) -> tuple[dict, dict]:
        """
        Recursively extracts configuration information from a given dictionary.

        Args:
            config_dict (dict): The dictionary containing configuration settings.

        Returns:
            tuple: A tuple containing two dictionaries:
                - all_keys: A dictionary mapping keys to their corresponding configuration information.
                - default_configs: A dictionary mapping keys to their default configuration values.

        Raises:
            AssertionError: If a key contains a period (.), which is not allowed.
        """
        all_keys = {}
        default_configs = {}
        for key, value in config_dict.items():
            assert "." not in key
            if isinstance(value, (dict)):
                children_keys, children_configs = self._get_config_infos(value)
                all_keys.update(
                    {
                        f"{key}.{child_key}": child_value_type
                        for child_key, child_value_type in children_keys.items()
                    }
                )
                default_configs[key] = children_configs
            else:
                value_info = self.get_value_info(value)
                all_keys[key] = value_info
                default_configs[key] = value_info[1]
        return all_keys, default_configs

    def get_config_infos(self) -> tuple[dict, dict]:
        return self._get_config_infos(self.global_configs)

    def _merge_configs(
        self,
        new_configs: dict,
        global_configs: dict,
        local_configs: dict,
        prefix="",
    ) -> None:
        """Overwrite default configs with new configs recursively.
        Args:
            new_configs: global flattern config dict with all hierarchical config keys joined by '.', i.e.
                {
                    'c_z': 32,
                    'model.evoformer.c_z': 16,
                    ...
                }
            global_configs: global hierarchical merging configs, i.e.
                {
                    'c_z' 32,
                    'c_m': 128,
                    'model': {
                        'evoformer': {
                            ...
                        }
                    }
                }
            local_configs: hierarchical merging config dict in current level, i.e. for 'model' level, this maybe
                {
                    'evoformer': {
                        'c_z': GlobalConfigValue("c_z"),
                    },
                    'embedder': {
                        ...
                    }
                }
            prefix (str, optional): A prefix string to prepend to keys during recursion. Defaults to an empty string.

        Returns:
            None: Mutates ``local_configs`` in place.

        Raises:
            Exception: If a required config value is not allowed to be None.
        """
        # Merge configs in current level first, since these configs maybe referenced by lower level
        for key, value in local_configs.items():
            if isinstance(value, dict):
                continue
            full_key = f"{prefix}.{key}" if prefix else key
            dtype, default_value, allow_none, required = self.config_infos[full_key]
            if full_key in new_configs and not isinstance(
                new_configs[full_key], ArgumentNotSet
            ):
                if allow_none and new_configs[full_key] in [
                    "None",
                    "none",
                    "null",
                ]:
                    local_configs[key] = None
                elif dtype == bool:
                    local_configs[key] = get_bool_value(new_configs[full_key])
                elif isinstance(value, (ListValue, list)):
                    local_configs[key] = (
                        [dtype(s) for s in new_configs[full_key].strip().split(",")]
                        if new_configs[full_key].strip()
                        else []
                    )
                else:
                    local_configs[key] = dtype(new_configs[full_key])
            elif isinstance(value, GlobalConfigValue):
                local_configs[key] = global_configs[value.global_key]
            else:
                if not allow_none and default_value is None:
                    raise Exception(f"config {full_key} not allowed to be none")
                local_configs[key] = default_value
        for key, value in local_configs.items():
            if not isinstance(value, dict):
                continue
            self._merge_configs(
                new_configs, global_configs, value, f"{prefix}.{key}" if prefix else key
            )

    def merge_configs(self, new_configs: dict) -> ConfigDict:
        configs = copy.deepcopy(self.global_configs)
        self._merge_configs(new_configs, configs, configs)
        return ConfigDict(configs)


def parse_configs(
    configs: dict, arg_str: Optional[str] = None, fill_required_with_null: bool = False
) -> ConfigDict:
    """
    Parses and merges configuration settings from a dictionary and command-line arguments.

    Args:
        configs (dict): A dictionary containing initial configuration settings.
        arg_str (str, optional): A string representing command-line arguments. Defaults to None.
        fill_required_with_null (bool, optional):
            A boolean flag indicating whether required values should be filled with `None` if not provided. Defaults to False.

    Returns:
        ConfigDict: The merged configuration dictionary.
    """
    manager = ConfigManager(configs, fill_required_with_null=fill_required_with_null)
    parser = argparse.ArgumentParser()
    # Register arguments
    for key, (
        dtype,
        default_value,
        allow_none,
        required,
    ) in manager.config_infos.items():
        # All config use str type, strings will be converted to real dtype later
        parser.add_argument(
            "--" + key, type=str, default=ArgumentNotSet(), required=required
        )
    # Merge user commandline pargs with default ones
    merged_configs = manager.merge_configs(
        vars(parser.parse_args(shlex.split(arg_str))) if arg_str else {}
    )
    return merged_configs


def parse_sys_args() -> str:
    """
    Check whether command-line arguments are valid.
    Each argument is expected to be in the format `--key value`.

    Returns:
        str: A string formatted as command-line arguments.

    Raises:
        ValueError: If a config key is malformed or missing its value.
    """
    args = sys.argv[1:]
    parsed_args = []
    idx = 0
    while idx < len(args):
        key = args[idx]
        if key in ("-h", "--help"):
            parsed_args.append(key)
            idx += 1
            continue
        if not key.startswith("--"):
            raise ValueError(f"Expected a config key starting with '--', got {key!r}")
        if "=" in key:
            parsed_args.append(key)
            idx += 1
            continue
        if idx + 1 >= len(args) or args[idx + 1].startswith("--"):
            raise ValueError(f"Missing value for config key {key!r}")
        parsed_args.extend([key, args[idx + 1]])
        idx += 2
    return " ".join(shlex.quote(arg) for arg in parsed_args)
