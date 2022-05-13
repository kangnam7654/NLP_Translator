from transformers import BartForConditionalGeneration
from custom_tokenizer.custom_tokenizer import get_tokenizer
import torch


def main():
    model = BartForConditionalGeneration.from_pretrained('./binary/kang_bart')
    tokenizer = get_tokenizer()

    # kor = input('원문 입력: ')
    kor = '어느정도 괜찮은 것 같네'

    input_ids = tokenizer.encode(kor)
    input_ids = torch.tensor(input_ids)
    input_ids = input_ids.unsqueeze(0)

    out = list(torch.argmax(model(input_ids).logits, dim=2))
    # out = model.generate(input_ids, eos_token_id=2, max_length=256, num_beams=5)
    translated = tokenizer.decode(out[0], skip_special_tokens=True)

    print(translated)

if __name__ == '__main__':
    main()
