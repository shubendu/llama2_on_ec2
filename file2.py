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


text = """Transcript
Mayank Aggarwal: I hope everybody has done the lunch.
Lovish Verma: Yeah.
Mayank Mehta: just,
Mayank Aggarwal: …
Mayank Mehta: Finished.
Mayank Aggarwal: I hope everybody has taken the nap. Please be awake for 15, 20 minutes.
I would take your time. So let me give you a quick recap how we need to update this resource capacity
report. I have pulled in from mine for right now. I have to select the option And selective months over here
as June to August and then generated.
Mayank Aggarwal: Okay.
Mayank Aggarwal: Over here. We can see internal and external. And the right hand side over here, there's
a default by default, it is selected as task. So every projections you need to update is at the task level, not
at the project or client level. Majorly. You have all done either at the client or that create the project level.
So that is wrong. So kindly update that I will showcase some of the team members projections as well,
who of whom I have seen it right now. Okay. So this is like I have updated for myself.
Mayank Aggarwal: this is a updated for myself for the month of June, but as I added to us, I don't know to
us nints, I've got four hours. Okay. And this annotus is at the client level. These are the projects over here.
And these are the task levels. So you need to expand it to the maximum and then you can need to add the
hours over here. But right now I have identity to and it automatically gets populated at the project level
over here. and at the client level, Majorly all have done is for and it has directed towards over here not
under the project or plant level. So kindly update over there as well.
Mayank Aggarwal: Okay, there could be a times that you might face a challenge over here. I'll be
happening for right now. You have updated two hours over here and if that populated about, now later on,
if you're ready three hours over here, But it's not getting updated over here. So you need to do it manually.
Mayank Aggarwal: Or some It's showing for some of the resources not showing so kindly make sure your
client level total is The total at the project level as well and at the task level, as well. once you do it then at
the external level, it gets populated automatically And you can calculate it as well from manual.
"""


res = generate_text(f"Please give me summary of the following transcript: {text}")
print(res[0]["generated_text"])
