import yaml


class YAMLConfig:
    def __init__(self, data):
        if isinstance(data, dict):
            for key, value in data.items():
                setattr(self, key, self._convert(value))
        else:
            raise ValueError("Input must be a dictionary")

    def _convert(self, obj):
        if isinstance(obj, dict):
            return YAMLConfig(obj)
        elif isinstance(obj, list):
            return [self._convert(item) for item in obj]
        else:
            return obj

    def __repr__(self):
        return f"YAMLConfig({self.__dict__})"


def load_config(config_path):
    with open(config_path, "r") as file:
        yaml_data = yaml.safe_load(file)
        config = YAMLConfig(yaml_data)
    return config
