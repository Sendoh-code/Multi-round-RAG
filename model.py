# modules/model.py
from transformers import pipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

class LocalLLM:
    def __init__(
        self,
        model_name: str,
        device_map="auto",
        torch_dtype="bfloat16",
        max_new_tokens=256,
        temperature=0.2,
        top_p=0.9,
        max_input_tokens=3500,
    ):
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        self.pipe = pipeline("text-generation", model=model_name, tokenizer=self.tokenizer)

        # Add special token BEFORE loading model
        special_tokens = {"additional_special_tokens": ["[ENDOFPROMPT]"]}
        num_new_tokens = self.tokenizer.add_special_tokens(special_tokens)

        # Load model
        dtype = torch.bfloat16 if torch_dtype == "bfloat16" else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=dtype,
            trust_remote_code=True
        )

        # Resize embedding if special tokens were added
        if num_new_tokens > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.max_input_tokens = max_input_tokens
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
    '''
    @torch.no_grad()
    def generate(self, prompt: str):
        # 1. 编码输入
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_ids = inputs.input_ids
        input_len = input_ids.shape[1]

        # 2. 生成
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        output_ids = outputs[0]

        # 3. 找到 [ENDOFPROMPT] 对应的 token id
        marker_id = self.tokenizer.convert_tokens_to_ids("[ENDOFPROMPT]")

        # 在输出 token 序列中找到 marker 的位置
        # 这里用 PyTorch 的方式找所有等于 marker_id 的 index
        marker_positions = (output_ids == marker_id).nonzero(as_tuple=True)[0]

        if len(marker_positions) > 0:
            # 取最后一个 marker 之后的内容（稳一点，防止模型又复述了一次 marker）
            start = marker_positions[-1].item() + 1
        else:
            # 如果没找到 marker，就退回到“去掉 prompt 部分”的逻辑
            start = input_len

        # 4. 截取 marker 之后的 token 作为真正的生成
        gen_ids = output_ids[start:]

        # 5. 解码为干净文本（这里可以安全地 skip_special_tokens）
        text = self.tokenizer.decode(
            gen_ids,
            skip_special_tokens=True
        ).strip()

        return text
    '''

    
    def generate(self, prompt: str):
        out = self.pipe(prompt, max_new_tokens=512)
        full_text = out[0]["generated_text"]
        pattern = r"\[ENDOFPROMPT\]\n\n(.*)"
        match = re.search(pattern, full_text, flags=re.DOTALL)
        if match:
            result = match.group(1)
        else:
            result = ""   # or full_text
        return result
    

    def check_answerability(self, query, retrieved_docs):
        docs = "\n\nRetrieved Passages:\n" + "\n".join(
            [f"[{i+1}] {doc}" for i, doc in enumerate(retrieved_docs)]
        ) + "\n\n"
        check_prompt = f"""
        You are an information classifier.

        Task: Decide if the provided context contains any evidence that directly helps answer the question.

        If there is relevant information that could help answer the question, reply "YES".
        If the context is missing key information for answer the question, reply "NO".

        Question:
        {query}

        Context:
        {docs}

        ONLY REPLY YES OR NO, NO ADDITIONAL WORDS:
        """+'[ENDOFPROMPT]\n\n'
        decision = self.generate(check_prompt).strip().upper()
        print(decision)
        if "NO" in decision:
            return False
        return True
