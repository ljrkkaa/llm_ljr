from vllm import LLM, SamplingParams

prompts = ["hell0, my name is..", "fuck you!", "不错不错"]

 
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=50)

llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct")

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

