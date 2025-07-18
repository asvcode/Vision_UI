import pathlib


def test_readme_contains_project_name():
    readme_path = pathlib.Path(__file__).resolve().parents[1] / 'README.md'
    content = readme_path.read_text(encoding='utf-8')
    assert 'Vision_UI' in content
