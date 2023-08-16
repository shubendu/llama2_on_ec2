from torch import cuda, bfloat16
import transformers
from transformers import StoppingCriteria, StoppingCriteriaList
import torch

print("loaded all packages")
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

print("Printing Device...")
print(device)

print("loading model....")
model_id ="meta-llama/Llama-2-13b-chat-hf"
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
    max_new_tokens=4096,  
    repetition_penalty=1.1 
)

print("===============================================")

long_text = """Transcript start :
Mayank Aggarwal: I hope everybody has done the lunch.
Lovish Verma: Yeah.
Mayank Mehta: just,
Mayank Aggarwal: …
Mayank Mehta: Finished.
Mayank Aggarwal: I hope everybody has taken the map abhay time. Please be awake for 15, 20 minutes. I would take your time. So let me give you a quick recap how we need to update this resource capacity report. I have pulled in from mine for right now. I have to select the option And selective months over here as June to August and then generated.
Mayank Aggarwal: Okay.
Mayank Aggarwal: Over here. We can see internal and external. And the right hand side over here, there's a default by default, it is selected as task. So every projections you need to update is at the task level, not at the project or client level. Majorly. You have all done either at the client or that create the project level. So that is wrong. So kindly update that I will showcase some of the team members projections as well, who of whom I have seen it right now. Okay. So this is like I have updated for myself.
Mayank Aggarwal: this is a updated for myself for the month of June, but as I added to us, I don't know to us nints, I've got four hours. Okay. And this annotus is at the client level. These are the projects over here. And these are the task levels. So you need to expand it to the maximum and then you can need to add the hours over here. But right now I have identity to and it automatically gets populated at the project level over here. and at the client level, Majorly all have done is for and it has directed towards over here not under the project or plant level. So kindly update over there as well.
Mayank Aggarwal: Okay, there could be a times that you might face a challenge over here. I'll be happening for right now. You have updated two hours over here and if that populated about, now later on, if you're ready three hours over here, But it's not getting updated over here. So you need to do it manually.
Mayank Aggarwal: Or some It's showing for some of the resources not showing so kindly make sure your client level total is The total at the project level as well and at the task level, as well. once you do it then at the external level, it gets populated automatically And you can calculate it as well from manual.
Abhay Saini: But mind, I heard about that. So if I let's say try to put a decimal in there,…
Mayank Aggarwal: Please.
Abhay Saini: that's a 5.4 or something. So then it's me that can only accept 1530 45. So that means it's minutes not hours, right?
Mayank Aggarwal: Yes, exactly.
Abhay Saini: So then if you wanted two minutes,
Mayank Aggarwal: 18 minutes. I didn't got you on this
Abhay Saini: No, no. You three hours there and then it does. So, three hours 16 to 3 18 minutes so 180 we have to write.
Mayank Aggarwal: Haha.
Mayank Aggarwal: No, no. You have to mention the hours, only hours, and three hours and 30 minutes suppose. Like this.
Abhay Saini: okay, okay.
Mayank Aggarwal: Okay, either three hours, 15 minutes, three hours, 30 minutes, three, or fourteen, five minutes, and four hours. Okay. And
Abhay Saini: I'll Can you take care? Try to write 5.4 there? It gives them.
Mayank Aggarwal: Yes, because timesheet…
Abhay Saini: It gives a prompt.
Mayank Aggarwal: because time sheet is only accepting, the intervals of 45 minutes. So that is the reason.
Mayank Aggarwal: So, you can update it, you ask 45 minutes over here. And let's get automatically populated. So sometimes you might face an issue on this total. Sometimes you might not So, kindly make sure that a total is correct.
Mayank Aggarwal: And never is Functionality is disabled for right now. So please do not clone right now. Forecasted leave is released in the next release. It's not working. As if you have applied in leave for the month of June July August in advance, then you need to sync it. And It will get I'm not a private. So it's showing over zero over here. These are the four days leave which I have applied in the month of June. So, that is why it's showing 32.
00:05:00
Mayank Aggarwal: Any questions over here in this part and yes, one more thing. I had a connector abhay in the morning so he was facing an issue there are not accessible projects are not showing up and below. So there are two conditions to update the projections over here And this dashboard. Firstly, the task should be assigned to you. client call and documentation should be assigned to me, otherwise if it's not 	assigned, then I cannot update the projections over here. and suppose if the project is marked as completed from backend. So you can't at the projections from here. for example as reality.
Mayank Aggarwal: For Lovish and Shubendu it might be showing but for others it might not be showing because the project is smartest completed from packing. So every take that as an exception case for his relative but for others if there's anything like this then he's To talk about the reality violated sources and they can exclude that reality of projections for right now. I will check that with ramadama and we'll update you in the group.
Mayank Aggarwal: Any other questions you have it over here? Till now?
Mayank Mehta: Yeah, Mehta had one question. So in the current project that we are working on. So the tasks are assigned as per our call with the client. So we are working based on the tax tasks So how can we update that for three months in advance that what tasks will hear sign first? Yeah.
Mayank Aggarwal: Okay, so if I take the case of, Big Bang BH example,…
Mayank Mehta: Yeah. Yes.
Mayank Aggarwal: reference thing. So, for right now, you can update the projections over the task assigned to you. Right now, this is about a quick line. Quality signed up with us, you can add the projections over here as then, the 15 hours for the next months. And they start off next month,…
Mayank Mehta: Okay.
Mayank Aggarwal: you can like astronom for assigning task. And she will upgrade you the task and accordingly based on the data that itself. You can add the projections over here, but at the client level we should have a knowledge key for that American Mehta has a resource capacity.
Mayank Mehta: Yeah.
Mayank Aggarwal: So at least 40 years dull, then a bucket asked about segregation of pathania basis on the basis of task assigned.
Mayank Mehta: yeah, level weeks, maybe he would ask a sign of just so we should enter it on the project level and then update when those stars are assigned to us in that particular week,
Mayank Aggarwal: Just a small addition over here, not at the project.
Mayank Mehta: Yep.
Mayank Aggarwal: Level job could ask be assigned and pvh camera. You can add the projections over there in that task and…
Mayank Mehta: Okay.
Mayank Aggarwal: later on when the tasks are assigned, then you can update A projections over that respective task licking our projections of mehta task level peak or angry.
Mayank Mehta: So my lobby mere pass. ABC tasks assigned so currently put and make 10 hours go there go he stay up and ask again you smetana so abhi bhani. But you next week? Merino or mere pass 10 hours of be ABC task behind and though weeks comment over to my it's time period. Karma projection you see me doll who came here?
Mayank Mehta: This is my point key.
Mayank Aggarwal: For.
Mayank Mehta: Task level pay both ambiguity already.
Mayank Aggarwal: Or task level you must be having six to seven task assigned right now in the bbh. An. You can divide the task on roughly basis. You may put July or August 50th, so you can divide your 50 years on roughly basis in those six to seven tasks. Addington, ten hours, five hours, four hours, 20 hours, 15 hours, whatever it is, okay? Or right now,…
00:10:00
Mayank Mehta: Okay.
Mayank Aggarwal: you can update it on roughly basis. Yeah, and
Mayank Aggarwal: Yes, somebody was saying something, You can roughly add the projections over there, in those six to seven tasks, making it to 50. Take a Jitender people approximately kamalai, a projections. And finally projection.
Mayank Mehta: We have to have,…
Mayank Aggarwal: Is that from
Mayank Mehta: we have work in piping, but we don't have the tasks yet. Kiyama pathania can make it a cookie. It's just a combined tasks. Yeah, BIA guitars. So gay. So, our d, also, we have documentation, also related to some particular tasks.
Mayank Aggarwal: The.
Mayank Mehta: So, to go Yeah, it knows me hours. Please see me confusion.
Mayank Aggarwal: man, just say I
Mayank Mehta: Okay.
Mayank Aggarwal: Whatever is your projection? You can divide your task. Projections and those six to seven tasks. Again, 50 years, come here, roughly basis, bbh camera. Take a task of the assignment that I understood. But up was no projections, that documentation client. Call me PUC, may go assigned a bbh camera roughly but so we will have an idea. apical deviation, approximately 50 years,…
Mayank Mehta: Chicken. Done.
Mayank Aggarwal: 40 years to go home. I was sure 45 minutes, Prema show, 40 years.
Mayank Mehta: Okay.
Mayank Aggarwal: Must be later on in the month of July, then you can update your task. Projections based on the task assigned, it will be a weekly task for all it won't be fixed cabinet again. Those waves are fixed top or Monday Tuesday. Whenever I based on your availability you need to update these projections. Jack was Kitna Kamal.
Mayank Aggarwal: Yes.
Sumanshu Sharma: A young man and actually Joe man Pathania raised again and son sikki, I don't know. For example, this month, I have one task, for example, in Pacific clinics the tasks, you are created this, all the tickets is very frank. So those tickets are like baseball. You get results in next week. So let's say I have, populated some projections. This I have in this month and these are you us in timesheet up, on completed by end of So then again, I have to do it for now for next month because I've already completed on the tasks which are right now, Mark completed in timesheet. So, the traditional will automatically get created, right?
Mayank Aggarwal: Right. Smart.
Sumanshu Sharma: If we can do and for such projects, if we can add some tasks like development or something like that, requirement analysis, wherever you can. Take help, just mentioned reaction and then you can specify Sunday when we have the specific that's created but PMS just a suggestion. If you can have that route Just for the sake of predictions.
Mayank Aggarwal: Yes.
Sumanshu Sharma: More generalized, task names, so that is easier for us to and it will save time, Your particularly will have to update the projections first. No. Or such stars.
Mayank Aggarwal: There are few suggestions which are already shared with a PM team as well. But This Dev team and this month would be like a pilot thing and…
Sumanshu Sharma: Okay.
Mayank Aggarwal: for one or two months, it will be a challenge for all updating this resource capacity.
00:15:00
Mayank Aggarwal: That if we can implement this functionality for future references and it will be easier for all generalized tasks everything.
Sumanshu Sharma: Sure. And
Mayank Aggarwal: Okay. ESI money.
Himani Vasudeva: I guess I went, I wanted to add to my mehta's point like…
Mayank Aggarwal: I,
Himani Vasudeva: since we are asked to fill the projections for, y ugust till July August right. So we will not be aware as to in which project will be aligned or what trainings, we are. We'll be working under So, how are we going to our projections for them?
Mayank Aggarwal: Yes, write that is the challenge being faced by all. Have to just add the roughly projection,…
Himani Vasudeva: Yes.
Mayank Aggarwal: that all I can say. This is the challenging case by us as well. PM team as a yeah,…
Himani Vasudeva: but your projections and…
Mayank Aggarwal: many people.
Himani Vasudeva: the kiss calendar daily,
Mayank Aggarwal: Working here we just have indeed to come and clearly training kalyan internal discussions. A Buckley with me predictions, does it mean a key? I will check with
Himani Vasudeva: By chance, Aggarwal create project. How am I going to add?
Mayank Aggarwal: Then you can update your projections.
Himani Vasudeva: Hours for that.
Mayank Aggarwal: That, as I said key, it will be a weekly task for all. I will get you the path assigned in the TS as well for this. Okay, so up,…
Himani Vasudeva: Okay.
Mayank Aggarwal: but it will projections update. These are key, projections means it's rough amount.
Himani Vasudeva: To these are not permanent,…
Himani Vasudeva: right? We?
Mayank Aggarwal: No, no,…
Mayank Aggarwal: no, no. These are your projecting key. What I will be doing in the next future coming months. And how much bandwidth I have right now. So, after us bandits, I will show or yogi to team can assign you the tasks or something or they can loop you into their projects or something like this. So, this is just a projection, if you have the bandwidth, then if we get the project so you will get DPUC or you get the training accordingly, your Orleans can take up the decision, Money was about 30 as the bandwidth there. Let's Give Assign her POC. Or if you have eight years bandwidth, then let's looper into the project or something like this. So this is the projections.
Himani Vasudeva: Okay.
Mayank Aggarwal: Not the final amount.
Sumanshu Sharma: Donald meant one more suggestion. I mean if this is I know. I mean just already that you mentioned that for three months if we can do right for just current month we need to update it at task level for that. for further two months, we can upgrade your project level and starting off the next month and…
Mayank Mehta: Yeah.
Sumanshu Sharma: can you specify the tasks whenever celebrated I mean just a suggestion or…
Mayank Aggarwal: Time.
Sumanshu Sharma: saying that we should start doing it in just by it is but just this is also a workable, approach. But current one should be at, task level. And next, two months can be a projected.
Mayank Aggarwal: It was advised,…
Mayank Mehta: I2 had the same solution.
Mayank Aggarwal: it was advised to us that position. Should always be a detox level.
Mayank Aggarwal: That is why we have been saying that you have to add the Predictionary task level. again, I will Take this into consideration for right now. I will ask the problem as well, if we can do this right now. But yes, for right now, please update it on task level, but I will be updating you on the final crux at the month and itself that what we can do on that part.
Mayank Aggarwal: Let me talk that point down.
Mayank Aggarwal: Okay, anybody else is having questions to be suppose if is abhishikha Candy called.
Abhishikha Sharma: Yes, ma'am.
Mayank Aggarwal: So abhishikha as I've said you need to do it on some tasks So, here you have done it over here on task level.
Mayank Aggarwal: Cornerstone. But on the internal task you have the 181 hours which is impossible. you won't be feeling time sheet of 301 us so, there is some calculations mistake in As of 19 hours.
00:20:00
Mayank Aggarwal: Of 17 US. That's cool. There's a interactive 120 hours so I believe it should be 62 hours, not 120. You need to fix it. Over here because in Analytics, team, it's 60 hours and ideal distance to us.
Mayank Aggarwal: so, please Check it once cross check as well once that you have added 60 hours and below over here.
Abhishikha Sharma: Okay.
Mayank Aggarwal: Please make sure and update it accordingly. So this should not be 181, obviously this. And is the time from the cold.
Mayank Aggarwal: Paradise, Pathania,…
Ritansh Thakur: Hi. Then.
Mayank Aggarwal: I believe that you have added the task on the project level. gratitude training, 70 hours but not on the task level. It's added on the project level over here, not on the task level.
Ritansh Thakur: Okay.
Mayank Aggarwal: Please add the projections over here.
Ritansh Thakur: Sure.
Mayank Aggarwal: Shivam got them as well. I believe there is some issue on this same. External is 175 and internal is 89.
Mayank Aggarwal: Is working on the call.
Mayank Aggarwal: Okay, have a update.
Mayank Aggarwal: Okay Any other question and you can also check the logs over here. who has updated? The hours accordingly. If you're going to check the laws too.
Mayank Aggarwal: Any other questions do you have on this and always for help? You can go over here? On the sap.
Mayank Aggarwal: Anybody taking nap? Or is it good for all?
Akshat Agrawal: Communist.
Sumanshu Sharma: Not here, but thanks man for explaining.
Mayank Aggarwal: Presentation.
Sumanshu Sharma: Let us know if any of these suggestions, God put by the decision makers and we can have them in private phase. So let us know. Yeah.
Mayank Mehta: Yeah. Thanks man.
Sumanshu Sharma: Thanks.
Mayank Aggarwal: Thanks guys, have a nice day and Take care.
Himani Vasudeva: Thank you.
Mayank Aggarwal: He?
Meeting ended after 00:29:28 👋
"""


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

print(len(long_text[:7000]))
print(long_text[:7000])
print("======================================")
print("Response")
print(len(text))
#pre_prompt = "Your are a helpful assistant. You do not repsond as 'User' or pretend to be 'User'. You only respond as 'Assistant'."
prompt_input = f"what is the summary of given transcript: {text} ?"
pre_prompt = "You are helpful assistant who gives summary of the transcripts.You do not respond as 'User' or pretend to be 'User'. Do not complete transcripts, just give summary."
pre_promt = "You are helpful assistant."

ll = long_text[3000:3000]
print(ll)
print("============================")
res = generate_text(f"{pre_prompt} Summary of the given text: {ll} Assistant:" )

#res = generate_text(f"Please give me summary of the following transcript: {long_text[:2264]} Transcript end")
print(res[0]["generated_text"])
