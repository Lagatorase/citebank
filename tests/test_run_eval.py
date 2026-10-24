from citebank.config import ROOT
from eval.run_eval import _settings_for


def test_experiment_folders_resolve_against_root(settings):
    # folders in an experiment yaml are text, and should end up as paths under the project root
    st = _settings_for(settings, {"name": "demo", "raw_dir": "data/demo", "index_dir": "data/index/demo", "auto_filter": False})
    assert st.raw_dir == ROOT / "data" / "demo"
    assert st.index_path().parent == ROOT / "data" / "index" / "demo"


def test_experiment_without_folders_keeps_base_folders(settings):
    st = _settings_for(settings, {"name": "plain", "top_k": 2})
    assert st.raw_dir == settings.raw_dir and st.top_k == 2
