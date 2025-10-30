from pathlib import Path
import json
from vlm_test.s3client import s3Client
from tqdm import tqdm
import time
from qwen_vl_utils import process_vision_info

class QwenInferencePipeline(): 
    def __init__(self, system_prompt, user_prompt, s3_bucket, model, processor,
                 qwen_version, max_response_tokens=4096): 
        self.qwen_version = qwen_version
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.max_response_tokens = max_response_tokens
        self.client = s3Client(buckets=[s3_bucket])
        self.model = model
        self.processor = processor
    
    def _get_prompt(self, vidpath): 
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": [
                {"type": "image", "image": vidpath,},
                {"type": "text", "text": self.user_prompt,}],
            }
        ] 
            
    def _preprocess_msg(self, msg): 
        text = self.processor.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(msg)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return inputs.to("cuda")
    
    def generate(self, vidpath):
        prompt = self._get_prompt(vidpath)
        input = self._preprocess_msg(prompt)
        
        generated_ids = self.model.generate(**input, max_new_tokens=self.max_response_tokens)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(input.input_ids, generated_ids)]
        return self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    def write_model_card(self,
        dst_dir: Path,
        model_id: str,
        dataset_path: Path,
        agg_test_time: float, 
        num_tested_elems: float
    ) -> None:
        card = {
            "model_id": model_id,
            "dataset_version": str(dataset_path),
            "model_version": self.qwen_version,
            "agg_test_time": agg_test_time,
            "avg_test_time_per_img": agg_test_time/num_tested_elems,
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
        }        

        results_key = f"{dst_dir}/model_card.json"
        results_bytes = json.dumps(card, indent=2).encode('utf-8')
        
        return self.client.put_object(str(results_key), results_bytes, content_type='application/json')   

    def iteratively_generate(self, vid_dir:Path, s3_dir:Path): 
        num_elems = 0
        start_time = time.perf_counter()
        paths = list(vid_dir.rglob("*.jpg"))

        for vid in tqdm(paths): 
            res = self.generate(str(vid))
            results_key = f"{s3_dir}/predictions/{vid.stem}.txt"
            results_bytes = bytes(res, 'utf-8')
            
            self.client.put_object(
                results_key,
                results_bytes,
                content_type='application/txt'
            )   
            num_elems+=1
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        
        return num_elems, elapsed_time