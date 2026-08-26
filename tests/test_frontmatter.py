from fastkernel.frontmatter import parse_yaml_subset, render_frontmatter, split_frontmatter

DOC = """---
model: mimi
objective: "Make it fast: really"
min_improvement: 0.01
continuous: true
max_iterations: null
gates:
  precision: strict
  stages: [smoke, shapes, numerical]
bench:
  repeats: 50
  ramp_seconds: 1.0
model_args:
  seconds: 1.0
  sweep: [0.25, 5.0]
protected:
  - GOAL.md
  - spec.py
---

# Body
text
"""


def test_split_and_parse():
    data, body = split_frontmatter(DOC)
    assert data["model"] == "mimi"
    assert data["objective"] == "Make it fast: really"
    assert data["min_improvement"] == 0.01
    assert data["continuous"] is True
    assert data["max_iterations"] is None
    assert data["gates"]["precision"] == "strict"
    assert data["gates"]["stages"] == ["smoke", "shapes", "numerical"]
    assert data["bench"]["repeats"] == 50
    assert data["model_args"]["sweep"] == [0.25, 5.0]
    assert data["protected"] == ["GOAL.md", "spec.py"]
    assert body.strip().startswith("# Body")


def test_render_roundtrip():
    data, body = split_frontmatter(DOC)
    text = render_frontmatter(data, body)
    data2, body2 = split_frontmatter(text)
    assert data2 == data
    assert body2.strip() == body.strip()


def test_no_frontmatter():
    data, body = split_frontmatter("plain text")
    assert data == {} and body == "plain text"


def test_inline_map_and_comments():
    data = parse_yaml_subset("a: {x: 1, y: two}  # comment\n# full comment\nb: [1, 'q, r']\n")
    assert data["a"] == {"x": 1, "y": "two"}
    assert data["b"] == [1, "q, r"]
