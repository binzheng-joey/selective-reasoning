"""Causal refinement of stage-1 punctuation triggers; preserve frozen v1 source."""
import re
from prefix_compare import ends_seen as original_ends


def rejection(text, end):
    before = text[:end].rstrip('\"\u201d\u2019\x27')
    if not before.endswith('.'):
        return None
    if before.endswith('..'):
        return 'ellipsis'
    token = re.search(r'(?<!\w)(\d+|[A-D]|[ivxIVX]+)\.$', before)
    if token:
        context = before[:token.start()]
        line = context.rsplit('\n', 1)[-1]
        if re.fullmatch(r'\s*(?:[-*+>]\s*)*(?:\*\*)?', line):
            return 'decorated_line_enumeration'
        if re.search(r'\b(?:candidates?|options?|choices?)\s*:', line, re.I):
            return 'inline_candidate_enumeration'
        if re.search(r'\(\s*$', context):
            return 'parenthesized_enumeration'
        if re.search(r'\(\s*\d+\.\s+[^)]*[,;]\s*$', line):
            return 'parenthesized_enumeration'
    if re.search(r'\b(?:U\.S|U\.K|cf|viz|approx|Fig|Eq|et al)\.$', before, re.I):
        return 'abbreviation'
    return None


def ends_seen(text):
    return [end for end in original_ends(text) if rejection(text, end) is None]


class StreamDetector:
    """Same causal v2 decisions, scanning only newly observable punctuation."""
    pattern=re.compile(r'[。！？]|[.!?][\"\u201d\u2019\x27]*(?=\s)')

    def __init__(self):
        self.cursor=0
        self.previous=''

    def update(self,text):
        assert text.startswith(self.previous),'Nonmonotonic decoded text'
        self.previous=text
        result=[]
        for m in self.pattern.finditer(text,self.cursor):
            self.cursor=m.end()
            if text[m.start()]=='.':
                before=text[:m.start()+1]
                if re.search(r'(?:\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc)|\be\.g|\bi\.e)\.$',before,re.I):
                    continue
                if re.search(r'(?:^|\n)\s*(?:\d+|[A-D])\.$',before):
                    continue
            if rejection(text,m.end()) is None:
                result.append(m.end())
        return result


def check():
    examples = [
        ('Candidates: 1. True, 2. False, 3. Unknown. ', 1),
        ('Keep order? Yes (1. True, 2. False, 3. Unknown). ', 2),
        ('* 1. Heading\n* 2. Next heading\n', 0),
        ('Options: A. First B. Second\n', 0),
        ('The answer is 4. Next. ', 2),
        ('In 1993. Then in 1994. ', 2),
        ('Only A or B. Not A. Therefore B. ', 3),
        ('* Fact 1: The dog is red.\n* Fact 2: It is kind.\n', 2),
        ('Value 3.14 and Dr. Wu. ', 1),
        ('See Fig. 1 for details. ', 1),
        ('完成。继续！', 2),
    ]
    for text, count in examples:
        assert len(ends_seen(text)) == count, (text, ends_seen(text))
        seen = set()
        stream=StreamDetector();stream_seen=set()
        for n in range(len(text) + 1):
            current = set(ends_seen(text[:n]))
            assert seen.issubset(current), (text, n, seen, current)
            seen = current
            stream_seen.update(stream.update(text[:n]))
            assert stream_seen==current,(text,n,stream_seen,current)
    return {'sentence_tests': len(examples), 'causal_prefix_stability': 'passed'}


if __name__ == '__main__':
    print(check())
