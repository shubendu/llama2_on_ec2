from huggingface_hub import hf_hub_download
from llama_cpp import Llama

model_name_or_path = "TheBloke/Llama-2-13B-chat-GGML"
model_basename = "llama-2-13b-chat.ggmlv3.q5_1.bin" # the model is in bin format

model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)

text = """Transcript 1 
Transcript
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

# GPU
lcpp_llm = None
lcpp_llm = Llama(
    model_path=model_path,
    n_threads=2, # CPU cores
    n_batch=512, # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    n_gpu_layers=32 # Change this value based on your model and your GPU VRAM pool.
    )

print(lcpp_llm.params.n_gpu_layers)

prompt = "Write a linear regression in python"
prompt_template=f'''SYSTEM: You are a helpful, respectful and honest assistant. Always answer as helpfully.

USER: {prompt}

ASSISTANT:
'''

response=lcpp_llm(prompt=prompt_template, max_tokens=256, temperature=0.5, top_p=0.95,
                  repeat_penalty=1.2, top_k=150,
                  echo=True)

print(response)

print("================================================")

print(response["choices"][0]["text"])

