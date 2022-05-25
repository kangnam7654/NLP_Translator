import numpy as np


class TextInfilling:
    def __init__(self, mask_token_id=4, return_list=True):
        self._return_list = return_list
        self.mask_token_id = [mask_token_id]

    def mask(self, text):
        len_text = len(text)
        span = np.random.poisson(3)
        while span >= len_text:
            span = np.random.poisson(3)
        mask_start_idx = np.random.choice(len_text-span)
        mask_end_idx = mask_start_idx + span

        start_cut = text[:mask_start_idx]
        end_cut = text[mask_end_idx:]
        result = np.concatenate([start_cut, self.mask_token_id, end_cut])
        if self._return_list:
            return list(result)
        else:
            return result


if __name__ == '__main__':
    sample = list(np.arange(30))
    text_infilling = TextInfilling()
    masked = text_infilling.mask(sample)