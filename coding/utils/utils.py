import re


def get_base_model(identifier: str) -> str:
    """
    Extracts the “base model class” (e.g. 'llama3.2', 'qwen3') from strings like
    'meta-llama/Llama-3.2-1B' or 'Qwen/Qwen3-8B'.
    """
    # 1. Skip any "{namespace}/" prefix
    # 2. Capture letters ([A-Za-z]+) as group 1
    # 3. Optionally match a hyphen/underscore
    # 4. Capture the version number (\d+(\.\d+)?) as group 2
    m = re.match(r"^(?:.*/)?([A-Za-z]+)(?:[-_]?(\d+(?:\.\d+)*))?(?=[^/]*$)", identifier)
    if not m:
        raise ValueError(f"couldn't parse model from {identifier!r}")
    model, version = m.groups()
    if version is not None:
        return f"{model.lower()}{version}"
    else:
        return model.lower()
