from torch import cuda, bfloat16
import transformers
from transformers import StoppingCriteria, StoppingCriteriaList
import torch

print("loaded all packages")
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

print("Printing Device...")
print(device)

print("loading model....")
model_id ="meta-llama/Llama-2-7b-chat-hf"
hf_auth = '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'
model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)

bnb_config = transformers.BitsAndBytesConfig(load_in_4bit = True,
bnb_4bit_quant_tyoe = 'nf4',
bnb_4bit_use_double_quant=True,
bnb_4bit_compute_dtype=bfloat16
)


model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto',
    use_auth_token=hf_auth,
    offload_folder="save_folder"
)


# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_auth)


stop_list = ['\nHuman:', '\n```\n']

stop_token_ids = [tokenizer(x)['input_ids'] for x in stop_list]

import torch
stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]



# define custom stopping criteria object
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                return True
        return False

stopping_criteria = StoppingCriteriaList([StopOnTokens()])

generate_text = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    return_full_text=False, 
    task='text-generation',
    temperature=0.1,  
    max_new_tokens=512,  
    repetition_penalty=1.1 
)

print("===============================================")


res = generate_text("Explain to me the difference between nuclear fission and fusion.")
print(res[0]["generated_text"])
