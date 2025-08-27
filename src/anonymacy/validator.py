from spacy import registry
from spacy.tokens import Span

def elf_proef(span: Span) -> bool:
    only_digits = "".join(c for c in span.text if c.isdigit())
    if all(only_digits[0] == c for c in only_digits):
        return False

    if len(only_digits) == 8:
        only_digits = "0" + only_digits
        
    if len(only_digits) != 9:
        return False

    total = 0
    for char, factor in zip(only_digits, [9, 8, 7, 6, 5, 4, 3, 2, -1]):
        total += int(char) * factor

    return total % 11 == 0

@registry.misc("anonymacy.elf_proef.v1")
def make_elf_proef():
    return elf_proef