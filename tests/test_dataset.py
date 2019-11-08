import tempfile
from src.datasets import TokenSpoilerDataset
CONLL = """# global.columns = ID FORM REDDIT_SPOILERS:SPOILER REDDIT_SPOILERS:SUPERSCRIPT REDDIT_SPOILERS:STRONG REDDIT_SPOILERS:EMPHASIS REDDIT_SPOILERS:LIST_ITEM REDDIT_SPOILERS:CODE REDDIT_SPOILERS:PRE
1    True           0  0  0  0  0  0  0
2    Remembrance    0  0  0  0  0  0  0
3    [newline]      0  0  0  0  0  0  0
4    I              0  0  0  0  0  0  0
5    honestly       0  0  0  0  0  0  0
6    did            0  0  0  0  0  0  0
7    not            0  0  0  0  0  0  0
8    expect         0  0  0  0  0  0  0
9    much           0  0  0  0  0  0  0
10   from           0  0  0  0  0  0  0
11   this           0  0  0  0  0  0  0
12   VN.            0  0  0  0  0  0  0
13   I              0  0  0  0  0  0  0
14   read           0  0  0  0  0  0  0
15   it             0  0  0  0  0  0  0
16   only           1  0  0  0  0  0  0
17   because        1  0  0  0  0  0  0
18   I              1  0  0  0  0  0  0
19   was            1  0  0  0  0  0  0
20   out            1  0  0  0  0  0  0
21   of             1  0  0  0  0  0  0
22   VN             1  0  0  0  0  0  0
23   money          1  0  0  0  0  0  0
24   and            1  0  0  0  0  0  0
25   it             1  0  0  0  0  0  0
26   got            1  0  0  0  0  0  0
27   recommended    1  0  0  0  0  0  0
""".split("\n")


def test_conll():
    class FakeTokenizer():
        def tokenize(self, w):
            return w.split("'")
        def convert_tokens_to_ids(self, l):
            return [137 for _ in l]
    dataset_file = tempfile.NamedTemporaryFile("wt")
    for line in CONLL:
        dataset_file.write(line + "\n")
    dataset_file.flush()
    ds = TokenSpoilerDataset([dataset_file.name], FakeTokenizer())
    print(ds.saved_data)
    assert ds.saved_data["0"].labels[-2] == True
    assert ds.saved_data["0"].labels[1] == False
