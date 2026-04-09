from pathlib import Path
import sys


def prepare_imports() -> None:
    project_root = Path(__file__).resolve().parents[1]

    try:
        import gsrs.model  # type: ignore # noqa: F401
    except ModuleNotFoundError:
        sibling_model_root = project_root.parent / 'gsrs.model'
        if sibling_model_root.exists():
            sys.path.insert(0, str(sibling_model_root))

    try:
        import gsrs
    except ModuleNotFoundError:
        return

    local_gsrs_root = project_root / 'gsrs'
    if local_gsrs_root.exists():
        local_path = str(local_gsrs_root)
        current_paths = list(gsrs.__path__)
        if local_path in current_paths:
            current_paths.remove(local_path)
        gsrs.__path__[:] = [local_path, *current_paths]


prepare_imports()
