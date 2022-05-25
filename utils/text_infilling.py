import math
import numpy as np


class TextInfilling:
    def __init__(self, text, mask_token_id=-1):
        self.text = text
        self.n = len(text)
        self.probabilities = self.get_probabilities()
        self.span = self.get_span()
        self.mask_token_id = mask_token_id
        self.masked_text = self.mask()

    @staticmethod
    def poisson_distribution(n, lamb=3):
        p_n = (lamb**n)*math.exp(-lamb)/math.factorial(n)
        if p_n < 0:
            return 0
        else:
            return p_n

    def get_probabilities(self):
        tmp = np.arange(self.n)
        probabilities = []
        for idx in tmp:
            probabilities.append(self.poisson_distribution(idx))
        return probabilities

    def get_span(self):
        span = np.random.choice(self.n, p=self.probabilities)
        return span

    def mask(self):
        mask_start_idx = np.random.choice(self.n-self.span)
        mask_end_idx = mask_start_idx + self.span

        start_cut = self.text[:mask_start_idx]
        end_cut = self.text[mask_end_idx:]
        result = np.concatenate([start_cut, [self.mask_token_id], end_cut])
        return result


if __name__ == '__main__':
    sample = list(np.arange(30))
    text_infilling = TextInfilling(sample)