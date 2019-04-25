def bitstring(n):
    if n < 0:
        raise ValueError('\'n\' can not be negative')

    def generate(bits, i=0):
        if i == n:
            yield bits

        else:
            bits[i] = 0
            yield from generate(bits, i + 1)

            bits[i] = 1
            yield from generate(bits, i + 1)

    return generate(n * [None])
