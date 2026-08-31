import importlib
import pkgutil


def discover_metrics(*packages):
    """Automatically discover metric modules exposing an evaluate function and METADATA.

    Scans the given packages (e.g. metrics.shared, metrics.monitor,
    metrics.diagnostic). Any module with an ``evaluate`` function and ``METADATA``
    is registered under its module name.
    """
    metrics = {}
    for package in packages:
        prefix = package.__name__ + "."
        for _, modname, ispkg in pkgutil.iter_modules(package.__path__, prefix):
            if ispkg:
                continue
            mod = importlib.import_module(modname)
            if hasattr(mod, "evaluate") and hasattr(mod, "METADATA"):
                metrics[modname.split(".")[-1]] = mod
    return metrics
