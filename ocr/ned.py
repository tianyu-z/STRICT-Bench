import Levenshtein


def ned(reference: str, generation: str) -> float:
    """
    Calculates the Normalized Edit Distance (NED) between a reference string
    and a generation string, assuming uniform operation costs.

    As per the paper "The Normalized Edit Distance with Uniform Operation Costs is a Metric"
    (e.g., arXiv:2201.06115v3 [cs.FL]), NED is defined as wgt(p) / len(p),
    where wgt(p) is the sum of costs of edit operations (Levenshtein distance
    with cost 1 for ins/del/sub) and len(p) is the total number of
    operations in the optimal edit path (each op counts as 1 in length).

    Args:
        reference: The reference string.
        generation: The generation string.

    Returns:
        The Normalized Edit Distance as a float. Returns 0.0 if both
        strings are empty, as len(p) would be 0.
    """
    if not isinstance(reference, str) or not isinstance(generation, str):
        raise TypeError("Both reference and generation must be strings.")

    wgt_p = Levenshtein.distance(reference, generation)
    opcodes = Levenshtein.opcodes(reference, generation)

    len_p = 0
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'insert':
            len_p += (j2 - j1)  # Number of characters inserted
        elif tag == 'delete':
            len_p += (i2 - i1)  # Number of characters deleted
        elif tag == 'replace':
            # Number of characters replaced (each is one 'c' op)
            len_p += (i2 - i1)
        elif tag == 'equal':
            # Number of characters matched (each is one 'n' op)
            len_p += (i2 - i1)

    if len_p == 0:
        if wgt_p == 0:  # Both strings are empty
            return 0.0
        else:
            return 0.0

    return wgt_p / len_p


if __name__ == "__main__":
    print(ned("hello world", "hello worl"))  # Example usage
    print(ned("hello world", "hello world"))  # Example usage
    print(ned('sdf', 'aafefjoijqannznxkcjnvkjnjkiahdsiwufhoj'))
    print(ned('aafefjoijqannznxkcjnvkjnjkiahdsiwufhoj', 'sdf'))
    print(ned('a', 'bbbb'))