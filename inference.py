from transformers import BartForConditionalGeneration
from custom_tokenizer.custom_tokenizer import get_tokenizer
import torch


def main():
    model = BartForConditionalGeneration.from_pretrained('./model_binary')
    tokenizer = get_tokenizer()

    # kor = input('원문 입력: ')
    kor = 'I know that, I am the one.'

    input_ids = tokenizer.encode(kor)
    input_ids = torch.tensor(input_ids)
    input_ids = input_ids.unsqueeze(0)

    out = model.generate(input_ids, eos_token_id=2, max_length=256, num_beams=5)
    translated = tokenizer.decode(out[0], skip_special_tokens=True)

    print(translated)

if __name__ == '__main__':
    main()
