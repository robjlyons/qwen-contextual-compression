import random
def prompt_split(prompt_ids,seed=42,fractions=(.7,.15,.15)):
 prompts=sorted(set(map(int,prompt_ids)));random.Random(seed).shuffle(prompts);a=round(len(prompts)*fractions[0]);b=a+round(len(prompts)*fractions[1]);groups={"train":set(prompts[:a]),"validation":set(prompts[a:b]),"test":set(prompts[b:])};return {name:[i for i,p in enumerate(prompt_ids) if int(p) in values] for name,values in groups.items()}
