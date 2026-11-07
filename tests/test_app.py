from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from orderbook.data import synthetic_book
from orderbook.research import run, save_artifact

APP = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture(scope="module")
def artifact_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("app")
    book, provenance = synthetic_book(1200)
    save_artifact(run(book, provenance), directory / "artifacts/research.json.gz")
    return directory


@pytest.mark.parametrize(
    "page", ["Research desk", "Book replay", "Experiments", "Execution", "Method & data"]
)
def test_pages_render(page, artifact_dir, monkeypatch):
    monkeypatch.chdir(artifact_dir)
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    app.sidebar.radio[0].set_value(page).run()
    assert not app.exception
    assert app.title
    if page == "Execution":
        app.selectbox[2].set_value(0.0).run()
        assert not app.exception
    if page == "Book replay":
        app.slider[0].set_value(20).run()
        assert not app.exception
