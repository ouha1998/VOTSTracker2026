from importlib import resources


def resource_filename(package_or_requirement: str, resource_name: str) -> str:
    return str(resources.files(package_or_requirement).joinpath(resource_name))
